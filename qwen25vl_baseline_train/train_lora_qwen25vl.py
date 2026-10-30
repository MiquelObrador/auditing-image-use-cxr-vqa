#!/usr/bin/env python3
"""Entrenamiento LoRA del baseline generico Qwen2.5-VL-3B-Instruct, SIN modulo de fusion
temporal, sobre exactamente los mismos datos/formatos/permutacion/pasos/hiperparametros que el
modelo TAC (ver tac_mcq_train/PREREGISTRO_2026-09-21.md). Unica diferencia deliberada:
arquitectura (sin TAC) -- ver qwen25vl_baseline_train/PLAN_2026-09-21.md.

Resolucion de ENTRENAMIENTO igualada a la que ve RAD-DINO por imagen (518x518=268324 px,
aspecto conservado por smart_resize) -- NO los MIN/MAX_PIXELS de evaluacion del protocolo del
ISBI (256..1280)*28*28: con esos, la secuencia real es ~2599 tokens/muestra y el throughput
medido da ~63.5h/epoca. La EVALUACION (full/blind/mismatch, etc.) sigue usando los
MIN/MAX_PIXELS estandar -- ver run_eval_qwen25vl.py.

Vision tower + merger congelados (freeze explicito, forzados a .eval()); LoRA solo sobre las
lineales del LLM (mismos target_modules, r, alpha, dropout, lr, schedule que el modelo TAC).
Mismo numero de pasos (6027, calculado desde 96426 filas / batch efectivo 16, igual que el TAC).

Uso:
  python3 train_lora_qwen25vl.py --out runs/full
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TAC_DIR = REPO / "tac_mcq_train"
sys.path.insert(0, str(REPO / "mmxu_eval"))
sys.path.insert(0, str(TAC_DIR))
sys.path.insert(0, str(REPO))
from config import MIMIC_JPG_ROOT, MMXU_DEV_JSONL  # noqa: E402

from format_targets import make_example, SPLITS_FILE   # noqa: E402 -- agnostico de modelo, reutilizado tal cual

IMG_ROOT = MIMIC_JPG_ROOT
SEED = 0
IGNORE_INDEX = -100
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# Resolucion de entrenamiento igualada a RAD-DINO (518x518 px por imagen), NO los limites de
# evaluacion del protocolo. Medido: 320 tokens visuales/imagen tras el merge 2x2 (ver
# throughput_probe_qwen25vl.py). Documentado aqui, no inventado.
TRAIN_PIXELS = 518 * 518
EFFECTIVE_BATCH = 16   # el mismo que el modelo TAC (batch=2 x accum=8)


def load_rows(split):
    ids = set(json.loads(SPLITS_FILE.read_text())[split])
    rows = []
    with open(MMXU_DEV_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in ids:
                rows.append(d)
    return sorted(rows, key=lambda d: d["id"])


def sample_by_patient(rows, n, seed=SEED):
    """Submuestra estratificada POR PACIENTE (no por fila): se cogen pacientes enteros hasta
    llegar a n, para que el clustering por paciente siga significando lo mismo. Duplicado
    desde tac_mcq_train/train_lora.py por convencion del proyecto (no se importa entre
    directorios de fase)."""
    if n is None or n >= len(rows):
        return rows
    by_pat = {}
    for d in rows:
        by_pat.setdefault(d["cur_image_path"].split("/")[2], []).append(d)
    pats = sorted(by_pat)
    order = np.random.default_rng(seed).permutation(len(pats))
    out = []
    for i in order:
        out.extend(by_pat[pats[i]])
        if len(out) >= n:
            break
    return sorted(out[:n], key=lambda d: d["id"])


class MMXUFormatDatasetQwen(Dataset):
    """Cada elemento: un ejemplo tokenizado con el chat template de Qwen2.5-VL, mismo formato
    (A/B/C) y permutacion de opciones por (qid, epoca) que el modelo TAC (format_targets.py es
    agnostico de arquitectura: solo produce (prompt_text, target, formato)).

    Labels: doble llamada al processor (prompt-only vs. prompt+respuesta), verificado byte a
    byte que el prefijo coincide (step_check_qwen25vl.py) -- el corte de longitud del prompt-
    only es seguro para enmascarar con IGNORE_INDEX.
    """

    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from qwen_vl_utils import process_vision_info

        d = self.rows[i]
        cur = Image.open(IMG_ROOT / d["cur_image_path"]).convert("RGB")
        prior = Image.open(IMG_ROOT / d["prior_image_path"]).convert("RGB")
        prompt_text, target, fmt = make_example(d, self.epoch)

        # Orden [prior, current] = [Image-1, Image-2], igual que mmxu_eval/models/qwen25vl.py.
        content = [{"type": "image", "image": prior}, {"type": "image", "image": cur},
                  {"type": "text", "text": prompt_text}]
        msgs_prompt = [{"role": "user", "content": content}]
        msgs_full = msgs_prompt + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]

        text_prompt = self.processor.apply_chat_template(
            msgs_prompt, tokenize=False, add_generation_prompt=True)
        text_full = self.processor.apply_chat_template(
            msgs_full, tokenize=False, add_generation_prompt=False)
        assert text_full.startswith(text_prompt), \
            f"prefijo no coincide en qid={d['id']} -- revisar chat template"

        image_inputs, _ = process_vision_info(msgs_prompt)
        enc_prompt = self.processor(text=[text_prompt], images=image_inputs, return_tensors="pt")
        enc_full = self.processor(text=[text_full], images=image_inputs, return_tensors="pt")
        prefix_len = enc_prompt.input_ids.shape[1]

        input_ids = enc_full.input_ids[0]
        labels = input_ids.clone()
        labels[:prefix_len] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": labels, "fmt": fmt,
               "pixel_values": enc_full.pixel_values, "image_grid_thw": enc_full.image_grid_thw}


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
            "pixel_values": torch.cat([f["pixel_values"] for f in feats], dim=0).to(torch.bfloat16),
            "image_grid_thw": torch.cat([f["image_grid_thw"] for f in feats], dim=0),
        }


def build_model(rank, alpha, dropout):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None).cuda()
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, min_pixels=TRAIN_PIXELS, max_pixels=TRAIN_PIXELS)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    # OJO: la vision tower de Qwen2.5-VL reusa los MISMOS nombres de hoja que el LLM
    # (gate_proj/up_proj/down_proj/q_proj/...), y PEFT empareja target_modules por nombre de
    # hoja en TODO el modelo, no por ruta -- construir el set de nombres excluyendo "visual"
    # NO basta (target_modules acaba conteniendo "gate_proj" etc. porque el LLM tambien lo
    # tiene, y entonces se aplica tambien dentro de `visual`). Hace falta exclude_modules
    # explicito para el submodulo completo.
    targets = sorted({n.split(".")[-1] for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear) and "visual" not in n
                      and "lm_head" not in n})
    # exclude_modules como LISTA solo excluye por nombre EXACTO o sufijo ".<nombre>" (ver
    # peft/tuners/tuners_utils.py:2350-2355) -- no basta para excluir un subarbol entero. Como
    # STRING se trata como regex con re.fullmatch contra la clave completa
    # ("model.visual.blocks.N.mlp.gate_proj", verificado): esto si excluye el subarbol.
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
    ap.add_argument("--frac", type=float, default=None,
                    help="fraccion (0,1] del train split, muestreada por paciente; None = todo")
    args = ap.parse_args()
    assert args.batch * args.accum == EFFECTIVE_BATCH, \
        f"batch({args.batch}) x accum({args.accum}) debe ser {EFFECTIVE_BATCH}, igual que el TAC"

    from transformers import Trainer, TrainingArguments, TrainerCallback

    all_rows = load_rows("train")
    n = int(round(args.frac * len(all_rows))) if args.frac is not None else None
    rows = sample_by_patient(all_rows, n)
    n_pat = len({d["cur_image_path"].split("/")[2] for d in rows})
    frac_txt = f"{args.frac*100:.0f}%" if args.frac is not None else "100%"
    print(f"entrenamiento: {len(rows)}/{len(all_rows)} filas ({frac_txt}), {n_pat} pacientes, "
         f"{args.epochs} epoca(s)")

    processor, model = build_model(args.rank, args.alpha, args.dropout)
    # jerarquia real (confirmada via named_modules(), claves "model.visual...."): PeftModel
    # -> .base_model (LoraModel) -> .model (Qwen2_5_VLForConditionalGeneration original)
    # -> .model (Qwen2_5_VLModel, el compuesto) -> .visual
    visual = model.base_model.model.model.visual
    visual.eval()

    ds = MMXUFormatDatasetQwen(rows, processor)

    class EpochCallback(TrainerCallback):
        def on_epoch_begin(self, a, state, control, **kw):
            ds.epoch = int(state.epoch or 0)
            print(f"  [epoca {ds.epoch}] formato y permutacion resorteados")

    class VisualFrozenTrainer(Trainer):
        def training_step(self, *a, **kw):
            visual.eval()
            return super().training_step(*a, **kw)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=10, save_strategy="no", report_to=[],
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=args.workers, seed=SEED, data_seed=SEED,
    )
    trainer = VisualFrozenTrainer(model=model, args=targs, train_dataset=ds,
                                  data_collator=Collator(processor.tokenizer.pad_token_id),
                                  callbacks=[EpochCallback()])

    t0 = time.perf_counter()
    res = trainer.train()
    dt = time.perf_counter() - t0

    out = Path(args.out)
    model.save_pretrained(out / "adapter")
    n_steps = int(res.global_step) if hasattr(res, "global_step") else None
    stats = {
        "splits_file": SPLITS_FILE.name,
        "n_train": len(rows), "n_patients": n_pat, "frac": args.frac,
        "epochs": args.epochs, "train_runtime_s": dt,
        "samples_per_s": len(rows) * args.epochs / dt,
        "pico_vram_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "train_loss": res.training_loss,
        "train_pixels_per_image": TRAIN_PIXELS,
        "hparams": {k: getattr(args, k) for k in
                    ("lr", "rank", "alpha", "dropout", "batch", "accum")},
    }
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n=== ENTRENAMIENTO TERMINADO ===")
    print(f"  loss final : {res.training_loss:.4f}")
    print(f"  tiempo     : {dt/3600:.2f} h")
    print(f"  pico VRAM  : {stats['pico_vram_GiB']:.2f} GiB")
    print(f"  adaptador  : {out/'adapter'}")


if __name__ == "__main__":
    main()
