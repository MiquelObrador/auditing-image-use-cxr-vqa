#!/usr/bin/env python3
"""Chequeo de sesgo de posicion de letra sobre las 5 condiciones BASE de MMXU-test
(full/hole/blind/mismatch/no_prior), para HuatuoGPT-7B y Qwen2.5-VL-7B.

Motivo: hasta ahora la permutacion de letras solo se aplico a `swap`
(swap_permuted_check.py), porque ahi el patron original resulto parcialmente explicado por
sesgo de posicion (confirmado para HuatuoGPT, descartado para Qwen2.5-VL). Un revisor senala
que si el sesgo basta para invertir el signo de una condicion, no se puede asumir sin
comprobarlo que las demas condiciones del criterio pre-registrado (full/hole/blind/
mismatch/no_prior) estan libres de el.

Misma tecnica EXACTA que swap_permuted_check.py: permutacion determinista de A/B/C/D por
fila via np.random.SeedSequence(0).spawn(n) (no hash() de Python, que no es estable entre
ejecuciones), calculada UNA VEZ sobre la poblacion completa (2992) y COMPARTIDA entre las 5
condiciones y los 2 modelos -- la fila qid=17 usa la misma permutacion en full_permuted,
hole_permuted, etc. La manipulacion de imagen de cada condicion es identica a
oracle_evaluate.py; lo unico que cambia es el orden A/B/C/D del prompt y, en consecuencia,
que letra aloja ahora la respuesta correcta original.

Uso:
    python3 permuted_check.py --model huatuo --condition full --output oracle_results/oracle_full_permuted.csv
    python3 permuted_check.py --model qwen25vl --condition mismatch --output ../qwen25vl_eval/oracle_results/oracle_mismatch_permuted.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from oracle_data import load_oracle_samples
from oracle_evaluate import mismatch_prior_index, MIN_PIXELS, MAX_PIXELS
from oracle_transforms import gray_inside_box
from evaluate import parse_letter
from models import get_model, MODEL_REGISTRY
from prompts import TEMPLATE

REPO_ROOT = Path(__file__).resolve().parent.parent
from data import MMXU_JSONL_PATH  # noqa: E402
LETTERS = ["A", "B", "C", "D"]
CONDITIONS = ("full", "hole", "blind", "mismatch", "no_prior")

CSV_FIELDS = [
    "qid", "pred_raw", "pred_letter", "answer_permuted", "correct", "content_type",
    "region", "cells_max", "condition", "source_qid_prior", "status",
]


def load_raw_entries(jsonl_path: Path) -> dict[int, dict]:
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            out[entry["id"]] = entry
    return out


def build_permutations(qids: list[int]) -> dict[int, list[str]]:
    """IDENTICO a swap_permuted_check.py::build_permutations -- misma semilla, mismo
    mecanismo, para que la permutacion de una fila sea la MISMA en todas las condiciones."""
    child_seeds = np.random.SeedSequence(0).spawn(len(qids))
    perms = {}
    for i, qid in enumerate(qids):
        rng = np.random.default_rng(child_seeds[i])
        perms[qid] = list(rng.permutation(LETTERS))
    return perms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_samples = list(load_oracle_samples())  # SIEMPRE poblacion completa, antes de --limit
    raw = load_raw_entries(MMXU_JSONL_PATH)
    qids = [s["qid"] for s in all_samples]
    perms = build_permutations(qids)
    mismatch_src_idx = mismatch_prior_index(len(all_samples))

    samples = all_samples[: args.limit] if args.limit is not None else all_samples

    model = get_model(args.model, max_new_tokens=args.max_new_tokens,
                       min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model.load()

    n_total = n_correct = n_unparseable = n_error = 0
    with open(args.output, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for i, sample in enumerate(samples):
            qid = sample["qid"]
            entry = raw[qid]
            perm = perms[qid]  # perm[j] = letra ORIGINAL que ahora ocupa la posicion LETTERS[j]
            new_options = {LETTERS[j]: entry["options"][perm[j]] for j in range(4)}
            prompt_text = TEMPLATE.format(
                question=entry["question"],
                option_a=new_options["A"], option_b=new_options["B"],
                option_c=new_options["C"], option_d=new_options["D"],
            )
            answer_permuted = next(l for j, l in enumerate(LETTERS) if perm[j] == entry["answer"])

            row = {
                "qid": qid, "answer_permuted": answer_permuted,
                "content_type": sample["content_type"], "region": sample["region"],
                "cells_max": sample["cells_max"], "condition": f"{args.condition}_permuted",
                "source_qid_prior": "",
            }
            try:
                if args.condition == "blind":
                    pred_raw = model.predict(None, None, prompt_text)
                elif args.condition == "no_prior":
                    image_current = Image.open(sample["image_current_path"]).convert("RGB")
                    pred_raw = model.predict(None, image_current, prompt_text)
                elif args.condition == "mismatch":
                    src = all_samples[mismatch_src_idx[i]]
                    row["source_qid_prior"] = src["qid"]
                    image_prior_src = Image.open(src["image_prior_path"]).convert("RGB")
                    image_current = Image.open(sample["image_current_path"]).convert("RGB")
                    pred_raw = model.predict(image_prior_src, image_current, prompt_text)
                elif args.condition == "hole":
                    image_prior = Image.open(sample["image_prior_path"]).convert("RGB")
                    image_current = Image.open(sample["image_current_path"]).convert("RGB")
                    image_prior = gray_inside_box(image_prior, sample["bbox_prior"])
                    image_current = gray_inside_box(image_current, sample["bbox_cur"])
                    pred_raw = model.predict(image_prior, image_current, prompt_text)
                else:  # full
                    image_prior = Image.open(sample["image_prior_path"]).convert("RGB")
                    image_current = Image.open(sample["image_current_path"]).convert("RGB")
                    pred_raw = model.predict(image_prior, image_current, prompt_text)
            except Exception as exc:  # noqa: BLE001
                row.update(pred_raw="", pred_letter="", correct=False, status="error")
                print(f"[error] qid={qid}: {exc}", file=sys.stderr)
                n_error += 1
                n_total += 1
                writer.writerow(row)
                out_f.flush()
                continue

            pred_letter = parse_letter(pred_raw)
            if pred_letter is None:
                row.update(pred_raw=pred_raw, pred_letter="", correct=False, status="unparseable")
                n_unparseable += 1
            else:
                correct = pred_letter == answer_permuted
                row.update(pred_raw=pred_raw, pred_letter=pred_letter, correct=correct, status="ok")
                n_correct += correct
            n_total += 1
            writer.writerow(row)
            out_f.flush()

            if n_total % 500 == 0:
                print(f"  ...{n_total}/{len(samples)}")

    print(f"\n=== {args.condition}_permuted, modelo={args.model} ({n_total} muestras) ===")
    if n_total:
        print(f"accuracy: {n_correct}/{n_total} = {n_correct/n_total:.3f}")
    print(f"unparseable: {n_unparseable}/{n_total}  error: {n_error}/{n_total}")


if __name__ == "__main__":
    main()
