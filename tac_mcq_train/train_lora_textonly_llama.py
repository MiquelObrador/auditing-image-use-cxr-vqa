#!/usr/bin/env python3
"""Text-only control paired with the TAC models' backbone: LoRA on Llama-3.2-3B-Instruct, trained and evaluated
WITHOUT images (review, Flaw #6).

Everything as in train_lora.py (models A, B, B2) except the images:
  - same assembly and loader (train_lora.build_model: Libra's Llama-3.2-3B-Instruct, the TAC loaded but frozen and
    never called), same LoRA (all linear layers of the LLM, r 16, alpha 32, dropout 0.05), same data (clean split,
    train_lora.load_rows("train"), make_example formats and option permutations per (qid, epoch)), same optimisation
    (lr 5e-5, cosine, 3% warmup, batch 2 x 8, bf16, seed 0, 1 epoch);
  - prompt: model_tac.build_conv_prompt(..., with_image=False) with the TAC system prompt, tokenised as
    model_tac.TacMcqBaseline.predict does for `blind` (tokenizer(prompt, add_special_tokens=False)); no <image>, no
    pixel values, so the LLM runs a text-only forward (images=None).
Evaluation is then exactly the `blind` path of the TAC wrapper:
  python3 run_eval_clean.py --condition blind --adapter runs/textonly_llama/adapter --output runs/textonly_llama/clean_blind.csv

Usage:
  python3 train_lora_textonly_llama.py --n-train 32 --out /tmp/humo_textonly_llama     # smoke
  python3 train_lora_textonly_llama.py --out runs/textonly_llama                       # 1 epoch, ~4 h estimated
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

import train_lora as TL
from format_targets import make_example, SPLITS_FILE
from model_tac import build_conv_prompt


class TextOnlyDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows, self.tokenizer, self.epoch = rows, tokenizer, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from libra.constants import IGNORE_INDEX
        prompt_text, target, fmt = make_example(self.rows[i], self.epoch)
        p_ids = torch.tensor(self.tokenizer(build_conv_prompt(prompt_text, with_image=False),
                                            add_special_tokens=False).input_ids, dtype=torch.long)
        a_ids = torch.tensor(self.tokenizer(target + "<|eot_id|>", add_special_tokens=False).input_ids,
                             dtype=torch.long)
        return {"input_ids": torch.cat([p_ids, a_ids]),
                "labels": torch.cat([torch.full((len(p_ids),), IGNORE_INDEX, dtype=torch.long), a_ids])}


class TextCollator:
    def __init__(self, pad_token_id):
        self.pad = pad_token_id

    def __call__(self, feats):
        from libra.constants import IGNORE_INDEX
        M = max(len(f["input_ids"]) for f in feats)
        pad = lambda x, v: torch.cat([x, torch.full((M - len(x),), v, dtype=torch.long)])
        return {"input_ids": torch.stack([pad(f["input_ids"], self.pad) for f in feats]),
                "labels": torch.stack([pad(f["labels"], IGNORE_INDEX) for f in feats]),
                "attention_mask": torch.stack([torch.cat([torch.ones(len(f["input_ids"]), dtype=torch.long),
                                                          torch.zeros(M - len(f["input_ids"]), dtype=torch.long)])
                                               for f in feats])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--assembly", default=TL.DEFAULT_ASSEMBLY)
    args = ap.parse_args()
    from transformers import Trainer, TrainingArguments, TrainerCallback

    rows = TL.sample_by_patient(TL.load_rows("train"), args.n_train)
    n_pat = len({d["cur_image_path"].split("/")[2] for d in rows})
    print(f"text-only training: {len(rows)} rows | {n_pat} patients | {args.epochs} epoch(s)")
    tokenizer, model, _, tac = TL.build_model(args.assembly, args.rank, args.alpha, args.dropout)
    tac.eval()
    ds = TextOnlyDataset(rows, tokenizer)

    class EpochCallback(TrainerCallback):
        def on_epoch_begin(self, a, state, control, **kw):
            ds.epoch = int(state.epoch or 0)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.03, bf16=True, gradient_checkpointing=False, logging_steps=10, save_strategy="no",
        report_to=[], remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=args.workers,
        seed=TL.SEED, data_seed=TL.SEED)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=TextCollator(tokenizer.pad_token_id),
                      callbacks=[EpochCallback()])
    t0 = time.perf_counter()
    res = trainer.train()
    dt = time.perf_counter() - t0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    sps = len(rows) * args.epochs / dt
    stats = {"splits_file": SPLITS_FILE.name, "n_train": len(rows), "n_patients": n_pat, "epochs": args.epochs,
             "train_runtime_s": dt, "samples_per_s": sps, "pico_vram_GiB": torch.cuda.max_memory_allocated() / 2**30,
             "train_loss": res.training_loss, "images": "none (text-only)",
             "loss_curve": [[h["step"], h["loss"]] for h in trainer.state.log_history if "loss" in h],
             "hparams": {k: getattr(args, k) for k in ("lr", "rank", "alpha", "dropout", "batch", "accum")}}
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"done: loss {res.training_loss:.4f}, {dt/3600:.2f} h, {sps:.2f} samples/s, "
          f"peak VRAM {stats['pico_vram_GiB']:.1f} GiB")


if __name__ == "__main__":
    main()
