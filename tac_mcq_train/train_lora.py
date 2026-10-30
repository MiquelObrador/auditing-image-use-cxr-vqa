#!/usr/bin/env python3
"""Entrenamiento LoRA del ensamblaje RAD-DINO + TAC + Llama-3.2-3B-Instruct sobre MMXU-dev.

RAD-DINO congelado (lo esta por construccion), TAC congelado Y en eval, LoRA solo sobre las
lineales del LLM. Loss unicamente sobre los tokens de respuesta. Ver PREREGISTRO_2026-09-21.md.

Piloto:
  python3 train_lora.py --n-train 4000 --out runs/pilot
Completo (tras decidir el presupuesto con el throughput medido):
  python3 train_lora.py --out runs/full
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from config import LIBRA_DIR, MIMIC_JPG_ROOT, MMXU_DEV_JSONL  # noqa: E402
sys.path.insert(0, str(LIBRA_DIR))
sys.path.insert(0, str(REPO / "mmxu_eval"))

from format_targets import make_example, SPLITS_FILE       # noqa: E402
from model_tac import build_conv_prompt, MODEL_NAME, MODEL_BASE, DEFAULT_ASSEMBLY  # noqa: E402

IMG_ROOT = MIMIC_JPG_ROOT
SEED = 0


# --------------------------------------------------------------------------- datos
class MMXUFormatDataset(Dataset):
    """Cada elemento: un ejemplo ya tokenizado y con las dos imagenes preprocesadas.

    El formato y la permutacion de opciones dependen de (qid, epoca): deterministas, pero
    distintos en cada epoca -- aumentacion real sin inflar el dataset. `epoch` lo actualiza
    EpochCallback al principio de cada epoca (los workers reciben una copia nueva del dataset
    al crearse el iterador, asi que el valor se propaga).
    """

    def __init__(self, rows, tokenizer, image_processor):
        self.rows = rows
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        # process_images solo lee `image_aspect_ratio`; un namespace es picklable y el
        # config completo del modelo no hace falta en los workers.
        self.img_cfg = SimpleNamespace(image_aspect_ratio="pad")
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from libra.mm_utils import tokenizer_image_token, process_images
        from libra.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX

        d = self.rows[i]
        prompt_text, target, fmt = make_example(d, self.epoch)
        prompt = build_conv_prompt(prompt_text, with_image=True)

        p_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt")
        a_ids = torch.tensor(
            self.tokenizer(target + "<|eot_id|>", add_special_tokens=False).input_ids,
            dtype=torch.long)
        input_ids = torch.cat([p_ids, a_ids])
        labels = torch.cat([torch.full((len(p_ids),), IGNORE_INDEX, dtype=torch.long), a_ids])

        imgs = []
        for rel in (d["cur_image_path"], d["prior_image_path"]):   # orden: ACTUAL, PREVIA
            px = process_images([Image.open(IMG_ROOT / rel).convert("RGB")],
                                self.image_processor, self.img_cfg)
            # process_images se traga las excepciones y devuelve None (mm_utils.py:49-51).
            assert px is not None, f"fallo al preprocesar {rel}"
            imgs.append(px[0])
        return {"input_ids": input_ids, "labels": labels,
                "cur": imgs[0], "prior": imgs[1], "fmt": fmt}


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, feats):
        from libra.constants import IGNORE_INDEX
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
            # [2, B, C, H, W]: el eje 0 es (actual, previa), NO el batch.
            "images": torch.stack([torch.stack([f["cur"] for f in feats]),
                                   torch.stack([f["prior"] for f in feats])]).to(torch.bfloat16),
        }


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
    llegar a n. Asi el clustering por paciente sigue significando lo mismo."""
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


# --------------------------------------------------------------------------- modelo
def build_model(assembly, rank, alpha, dropout, unfreeze_tac=False, tac_init="libra",
                tac_state_init=None, stage1_tac_only=False):
    from libra.model.builder import load_pretrained_model
    from peft import LoraConfig, get_peft_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        assembly, MODEL_BASE, MODEL_NAME, device_map=None, device="cuda")
    model = model.to(torch.bfloat16)
    raw = torch.load(Path(assembly) / "mm_projector.bin", map_location="cpu")
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in raw.items()}, strict=False)
    if tac_init == "scratch":
        # Misma inicializacion que la etapa 1 de Libra: un TAC(config) nuevo con la init por
        # defecto de PyTorch (libra_arch.py:101, pretrain_mm_mlp_adapter=None), sin pesos.
        from libra.model.multimodal_projector.builder import TAC
        torch.manual_seed(SEED)
        fresh = TAC(model.get_model().config).state_dict()
        tac_mod = model.get_model().mm_projector
        tac_mod.load_state_dict({k: v.to(torch.bfloat16) for k, v in fresh.items()})
        pref = "model.mm_projector."
        assert all(pref + k in raw for k in fresh), "claves del TAC no coinciden con mm_projector.bin"
        same = [k for k, v in tac_mod.state_dict().items()
                if torch.equal(v.cpu(), raw[pref + k].to(torch.bfloat16))]
        assert not same, f"tensores del TAC todavia iguales a los de Libra: {same[:5]}"
        print(f"TAC reinicializado desde cero: {len(fresh)} tensores, ninguno igual a Libra")
    if tac_state_init is not None:
        # B2 etapa 2: el TAC viene YA entrenado de la etapa 1 (un tac_state_dict.bin propio,
        # sin el prefijo "model.mm_projector." de mm_projector.bin -- son las mismas claves
        # que produce tac.state_dict() en el guardado de main()). Se congela mas abajo.
        state = torch.load(tac_state_init, map_location="cpu")
        model.get_model().mm_projector.load_state_dict(
            {k: v.to(torch.bfloat16) for k, v in state.items()})
        print(f"TAC inicializado desde {tac_state_init}")
    model = model.cuda()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for p in model.parameters():
        p.requires_grad = False
    # Con PEFT + checkpointing, sin enable_input_require_grads() la entrada del primer bloque
    # no requiere grad y el checkpointing corta el grafo: no llegaria ningun gradiente.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    if stage1_tac_only:
        # B2 etapa 1: sin LoRA. RAD-DINO y el LLM se quedan congelados; el TAC es el UNICO
        # modulo entrenable, asi que la loss solo puede bajar si el TAC aporta informacion de
        # la imagen -- no hay LoRA que pueda absorber el atajo textual en su lugar.
        model.get_model().mm_projector.requires_grad_(True)
        tac = model.get_model().mm_projector
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {n_tr:,} || all params: {n_all:,} || "
              f"trainable%: {100*n_tr/n_all:.4f}")
        bad = [n for n, p in model.named_parameters()
               if p.requires_grad and "mm_projector" not in n]
        assert not bad, f"hay params entrenables fuera del TAC: {bad[:3]}"
        return tokenizer, model, image_processor, tac

    targets = sorted({n.split(".")[-1] for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear)
                      and "vision_tower" not in n and "mm_projector" not in n
                      and "lm_head" not in n})
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=targets, task_type="CAUSAL_LM"))

    # get_peft_model() vuelve a congelar TODO menos la LoRA (_mark_only_adapters_as_trainable),
    # asi que descongelar el TAC solo tiene efecto si se hace DESPUES de esta llamada, no antes.
    if unfreeze_tac:
        model.base_model.model.get_model().mm_projector.requires_grad_(True)
    model.print_trainable_parameters()

    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and "vision_tower" in n]
    assert not bad, f"hay params entrenables en la vision tower: {bad[:3]}"
    if not unfreeze_tac:
        bad = [n for n, p in model.named_parameters()
               if p.requires_grad and "mm_projector" in n]
        assert not bad, f"hay params entrenables en el TAC sin --unfreeze-tac: {bad[:3]}"
    tac = model.base_model.model.get_model().mm_projector
    return tokenizer, model, image_processor, tac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=None, help="submuestra por paciente; None = todo")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--assembly", default=DEFAULT_ASSEMBLY)
    ap.add_argument("--unfreeze-tac", action="store_true",
                    help="descongela el TAC con su propio grupo de LR (--tac-lr)")
    ap.add_argument("--tac-lr", type=float, default=None,
                    help="LR del TAC descongelado; por defecto lr/10")
    ap.add_argument("--tac-init", choices=["libra", "scratch"], default="libra",
                    help="scratch = TAC con init aleatoria de Libra (exige --unfreeze-tac)")
    ap.add_argument("--stage1-tac-only", action="store_true",
                    help="B2 etapa 1: sin LoRA, solo el TAC entrenable "
                         "(exige --unfreeze-tac --tac-init scratch)")
    ap.add_argument("--tac-state-init", default=None,
                    help="B2 etapa 2: cargar el TAC desde este tac_state_dict.bin "
                         "(salida de --stage1-tac-only); se congela como de costumbre")
    args = ap.parse_args()
    assert args.tac_init == "libra" or args.unfreeze_tac, "--tac-init scratch exige --unfreeze-tac"
    assert not args.stage1_tac_only or (args.unfreeze_tac and args.tac_init == "scratch"), \
        "--stage1-tac-only exige --unfreeze-tac --tac-init scratch"
    assert args.tac_state_init is None or args.tac_init == "libra", \
        "--tac-state-init sustituye la init; no lo combines con --tac-init scratch"
    if args.unfreeze_tac and args.tac_lr is None:
        args.tac_lr = args.lr / 10

    from transformers import Trainer, TrainingArguments, TrainerCallback

    rows = sample_by_patient(load_rows("train"), args.n_train)
    n_pat = len({d["cur_image_path"].split("/")[2] for d in rows})
    print(f"entrenamiento: {len(rows)} filas | {n_pat} pacientes | {args.epochs} epoca(s)"
         + (f" | TAC descongelado, tac_lr={args.tac_lr:g}" if args.unfreeze_tac else "")
         + (" | ETAPA 1 de B2: sin LoRA" if args.stage1_tac_only else ""))

    tokenizer, model, image_processor, tac = build_model(
        args.assembly, args.rank, args.alpha, args.dropout, unfreeze_tac=args.unfreeze_tac,
        tac_init=args.tac_init, tac_state_init=args.tac_state_init,
        stage1_tac_only=args.stage1_tac_only)
    if not args.unfreeze_tac:
        tac.eval()

    ds = MMXUFormatDataset(rows, tokenizer, image_processor)

    class EpochCallback(TrainerCallback):
        def on_epoch_begin(self, a, state, control, **kw):
            ds.epoch = int(state.epoch or 0)
            print(f"  [epoca {ds.epoch}] formato y permutacion resorteados")

    class TacFrozenTrainer(Trainer):
        """Trainer.train() pone TODO en .train(), y el TAC tiene dropout 0.1. Congelado no es
        lo mismo que en eval: sin esto inyectaria ruido en los embeddings visuales."""
        def training_step(self, *a, **kw):
            tac.eval()
            return super().training_step(*a, **kw)

    class TacUnfrozenTrainer(Trainer):
        """TAC descongelado con su propio grupo de LR (mas bajo que el de la LoRA). Sin
        modo eval forzado: entrena en modo normal, dropout incluido, como el resto."""
        def __init__(self, *a, tac_lr=None, tac_module=None, **kw):
            self._tac_lr = tac_lr
            self._tac_param_ids = {id(p) for p in tac_module.parameters()}
            super().__init__(*a, **kw)

        def create_optimizer(self):
            if self.optimizer is None:
                opt_model = self.model
                main_params, tac_params = [], []
                for n, p in opt_model.named_parameters():
                    if not p.requires_grad:
                        continue
                    (tac_params if id(p) in self._tac_param_ids else main_params).append(p)
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model)
                base_lr = optimizer_kwargs.pop("lr", self.args.learning_rate)
                groups = [g for g in ({"params": main_params, "lr": base_lr},
                                      {"params": tac_params, "lr": self._tac_lr}) if g["params"]]
                self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
            return self.optimizer

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=False,   # ya activado a mano junto a enable_input_require_grads
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,    # el dataset devuelve "images", que no es columna estandar
        label_names=["labels"],         # PeftModel oculta la firma del modelo base al Trainer
        dataloader_num_workers=args.workers,
        seed=SEED,
        data_seed=SEED,
    )
    if args.unfreeze_tac:
        trainer = TacUnfrozenTrainer(model=model, args=targs, train_dataset=ds,
                                     data_collator=Collator(tokenizer.pad_token_id),
                                     callbacks=[EpochCallback()],
                                     tac_lr=args.tac_lr, tac_module=tac)
    else:
        trainer = TacFrozenTrainer(model=model, args=targs, train_dataset=ds,
                                   data_collator=Collator(tokenizer.pad_token_id),
                                   callbacks=[EpochCallback()])

    t0 = time.perf_counter()
    res = trainer.train()
    dt = time.perf_counter() - t0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.stage1_tac_only:
        # la etapa 1 de B2 no envuelve el modelo en PEFT (no hay LoRA que guardar).
        model.save_pretrained(out / "adapter")
    if args.unfreeze_tac:
        # el TAC no es un modulo PEFT: save_pretrained() de arriba no incluye sus pesos.
        torch.save(tac.state_dict(), out / "tac_state_dict.bin")
    sps = len(rows) * args.epochs / dt
    stats = {
        "splits_file": SPLITS_FILE.name,
        "n_train": len(rows), "n_patients": n_pat, "epochs": args.epochs,
        "train_runtime_s": dt, "samples_per_s": sps,
        "h_por_epoca_106929": 106929 / sps / 3600,
        "pico_vram_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "train_loss": res.training_loss,
        "loss_curve": [[h["step"], h["loss"]] for h in trainer.state.log_history if "loss" in h],
        "hparams": {k: getattr(args, k) for k in
                    ("lr", "rank", "alpha", "dropout", "batch", "accum",
                     "unfreeze_tac", "tac_lr", "tac_init", "stage1_tac_only", "tac_state_init")},
    }
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n=== ENTRENAMIENTO TERMINADO ===")
    print(f"  loss final      : {res.training_loss:.4f}")
    print(f"  tiempo          : {dt/60:.1f} min  ({sps:.2f} muestras/s)")
    print(f"  h/epoca (106929): {stats['h_por_epoca_106929']:.1f}")
    print(f"  pico VRAM       : {stats['pico_vram_GiB']:.2f} GiB")
    if not args.stage1_tac_only:
        print(f"  adaptador       : {out/'adapter'}")
    if args.unfreeze_tac:
        print(f"  tac_state_dict  : {out/'tac_state_dict.bin'}")


if __name__ == "__main__":
    main()
