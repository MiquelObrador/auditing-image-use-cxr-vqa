#!/usr/bin/env python3
"""Ablation de SOLO TEXTO: mismo checkpoint (Qwen2.5-VL-3B-Instruct -- el MISMO que el
baseline generico visual, pesos de partida byte-identicos), mismos datos/formatos/
permutacion/pasos/hiperparametros LoRA, pero SIN NINGUNA entrada visual, ni en entrenamiento
ni en evaluacion.

Por que este checkpoint y no Llama-3.2-3B-Instruct: usar el MISMO checkpoint que el baseline
visual aisla una unica variable (presencia/ausencia de imagen). Llama-3.2 introduciria un LLM
distinto (otro pretraining, otro tokenizer, otra escala de instruction-tuning), confundiendo
cualquier diferencia con "es un LLM distinto" en vez de "no vio imagenes".

Responde a la objecion "el baseline veia imagenes al entrenar, quiza por eso `blind` no cae":
si este modelo -- que NUNCA ve una imagen, ni en entrenamiento ni en evaluacion -- alcanza una
accuracy similar al 0.8088 de `blind` del baseline visual, confirma que el atajo esta en el
TEXTO del dataset (pregunta+opciones), no es un artefacto de haber visto imagenes durante el
entrenamiento.

Mucho mas barato que el baseline visual: sin torre de vision, sin pixel_values, secuencia
~150-180 tokens (solo texto) en vez de ~800.

Uso:
  python3 train_lora_textonly.py --out runs/textonly
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TAC_DIR = REPO / "tac_mcq_train"
sys.path.insert(0, str(TAC_DIR))
sys.path.insert(0, str(REPO))
from config import MMXU_DEV_JSONL  # noqa: E402

from format_targets import make_example, SPLITS_FILE   # noqa: E402 -- agnostico de modelo/modalidad

SEED = 0
IGNORE_INDEX = -100
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"   # mismo checkpoint que el baseline visual
EFFECTIVE_BATCH = 16


def load_rows(split):
    ids = set(json.loads(SPLITS_FILE.read_text())[split])
    rows = []
    with open(MMXU_DEV_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in ids:
                rows.append(d)
    return sorted(rows, key=lambda d: d["id"])


class TextOnlyDataset(Dataset):
    """Mismo (prompt_text, target, formato) que el baseline visual (format_targets.make_example
    es agnostico de modalidad), pero SIN insertar ninguna imagen en el mensaje -- ni siquiera
    un placeholder. El chat template de Qwen2.5-VL no exige contenido de tipo imagen."""

    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        d = self.rows[i]
        prompt_text, target, fmt = make_example(d, self.epoch)

        msgs_prompt = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        msgs_full = msgs_prompt + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]

        text_prompt = self.processor.apply_chat_template(
            msgs_prompt, tokenize=False, add_generation_prompt=True)
        text_full = self.processor.apply_chat_template(
            msgs_full, tokenize=False, add_generation_prompt=False)
        assert text_full.startswith(text_prompt), f"prefijo no coincide en qid={d['id']}"

        prompt_ids = self.processor.tokenizer(text_prompt, add_special_tokens=False).input_ids
        full_ids = self.processor.tokenizer(text_full, add_special_tokens=False).input_ids
        prefix_len = len(prompt_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prefix_len] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": labels, "fmt": fmt}


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, feats):
        M = max(len(f["input_ids"]) for f in feats)

        def pad(x, fill):
            return torch.cat([x, torch.full((M - len(x),), fill, dtype=torch.long)])

        return {
            "input_ids": torch.stack([pad(f["input_ids"], self.pad_token_id) for f in feats]),
            "labels": torch.stack([pad(f["labels"], IGNORE_INDEX) for f in feats]),
            "attention_mask": torch.stack([
                torch.cat([torch.ones(len(f["input_ids"]), dtype=torch.long),
                           torch.zeros(M - len(f["input_ids"]), dtype=torch.long)])
                for f in feats]),
        }


def build_model(rank, alpha, dropout):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None).cuda()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    targets = sorted({n.split(".")[-1] for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear) and "visual" not in n
                      and "lm_head" not in n})
    # mismo bug ya encontrado con el baseline visual: exclude_modules como LISTA no basta
    # para excluir un subarbol completo (solo excluye nombre exacto o sufijo ".<nombre>");
    # como STRING se trata como regex con re.fullmatch -- eso si excluye "visual" entero.
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=targets, exclude_modules=r"model\.visual\..*", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    bad = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    assert not bad, f"hay params entrenables en la vision tower: {bad[:3]}"
    return processor, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    assert args.batch * args.accum == EFFECTIVE_BATCH

    from transformers import Trainer, TrainingArguments, TrainerCallback

    rows = load_rows("train")
    print(f"entrenamiento SOLO TEXTO: {len(rows)} filas, {args.epochs} epoca(s)")

    processor, model = build_model(args.rank, args.alpha, args.dropout)
    ds = TextOnlyDataset(rows, processor)

    class EpochCallback(TrainerCallback):
        def on_epoch_begin(self, a, state, control, **kw):
            ds.epoch = int(state.epoch or 0)
            print(f"  [epoca {ds.epoch}] formato y permutacion resorteados")

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=10, save_strategy="no", report_to=[],
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=args.workers, seed=SEED, data_seed=SEED,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(processor.tokenizer.pad_token_id),
                      callbacks=[EpochCallback()])

    t0 = time.perf_counter()
    res = trainer.train()
    dt = time.perf_counter() - t0

    out = Path(args.out)
    model.save_pretrained(out / "adapter")
    stats = {"splits_file": SPLITS_FILE.name,
             "n_train": len(rows), "epochs": args.epochs, "train_runtime_s": dt,
            "samples_per_s": len(rows) * args.epochs / dt,
            "pico_vram_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "train_loss": res.training_loss,
            "hparams": {k: getattr(args, k) for k in ("lr", "rank", "alpha", "dropout", "batch", "accum")}}
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n=== ENTRENAMIENTO SOLO-TEXTO TERMINADO ===")
    print(f"  loss final : {res.training_loss:.4f}")
    print(f"  tiempo     : {dt/60:.1f} min")
    print(f"  pico VRAM  : {stats['pico_vram_GiB']:.2f} GiB")


if __name__ == "__main__":
    main()
