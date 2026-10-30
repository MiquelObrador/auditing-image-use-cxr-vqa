#!/usr/bin/env python3
"""Diagnóstico: ¿el patrón de `swap` (cae MÁS contra la etiqueta invertida que contra la
original, en los dos VLM genéricos de MMXU-test) es señal real de que el modelo no distingue
dirección, o es el sesgo de posición de letra ya documentado (InternVL2 predice A el 46.6% de
las veces vs. 24.8% esperado, Qwen2-VL en menor medida) actuando sobre la letra que construye
"invertida", que en el dataset original no está distribuida al azar entre A/B/C/D?

Se baraja el orden A/B/C/D de las 4 opciones de cada pregunta -- permutación determinista por
qid, MISMA técnica que biovilt_eval/qwen2vl_evaluate.py usa para ImaGenome
(numpy.random.SeedSequence(0).spawn(n), no hash() de Python, que no es estable entre
ejecuciones). El swap de imágenes (previa<->actual) es idéntico a
oracle_evaluate.py --condition swap -- lo único que cambia es qué letra aloja cada opción.

Lectura: si el patrón (cae más vs. invertida que vs. original) persiste con las letras
barajadas, es señal real -- el sesgo de posición no puede explicarlo, porque ya no hay
ninguna letra fija asociada a "invertida" en todo el dataset. Si desaparece o se invierte,
era el sesgo.

Uso: python3 swap_permuted_check.py --model huatuo --output oracle_results/oracle_swap_permuted.csv
     python3 swap_permuted_check.py --model qwen25vl --output ../qwen25vl_eval/oracle_results/oracle_swap_permuted.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from oracle_data import load_oracle_samples
from oracle_evaluate import label_option, INVERT_CLASS, MIN_PIXELS, MAX_PIXELS
from evaluate import parse_letter
from models import get_model, MODEL_REGISTRY
from prompts import TEMPLATE

REPO_ROOT = Path(__file__).resolve().parent.parent
from data import MMXU_JSONL_PATH  # noqa: E402
LETTERS = ["A", "B", "C", "D"]

CSV_FIELDS = [
    "qid", "pred_raw", "pred_letter", "answer_permuted", "answer_inverted_permuted",
    "correct_vs_original", "correct_vs_inverted", "content_type", "region", "cells_max",
    "status", "condition",
]


def load_raw_options(jsonl_path: Path) -> dict[int, dict]:
    """qid -> entry crudo del jsonl (para 'options' y 'question'), solo lo necesario aquí."""
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            out[entry["id"]] = entry
    return out


def build_permutations(qids: list[int]) -> dict[int, list[str]]:
    """qid -> lista de 4 letras originales en el orden en que ahora ocupan A,B,C,D. Mismo
    mecanismo que qwen2vl_evaluate.py (SeedSequence.spawn), aplicado en el mismo orden en que
    load_oracle_samples() produce los qids, para que sea reproducible sin más contexto."""
    child_seeds = np.random.SeedSequence(0).spawn(len(qids))
    perms = {}
    for i, qid in enumerate(qids):
        rng = np.random.default_rng(child_seeds[i])
        perms[qid] = list(rng.permutation(LETTERS))
    return perms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = list(load_oracle_samples())
    raw = load_raw_options(MMXU_JSONL_PATH)
    qids = [s["qid"] for s in samples]
    perms = build_permutations(qids)  # SIEMPRE sobre la población completa, antes de --limit

    if args.limit is not None:
        samples = samples[: args.limit]

    model = get_model(args.model, max_new_tokens=args.max_new_tokens,
                       min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model.load()

    n_total = n_error = n_unparseable = 0
    n_inv_excluded = 0
    with open(args.output, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for sample in samples:
            qid = sample["qid"]
            entry = raw[qid]
            perm = perms[qid]  # perm[i] = letra ORIGINAL que ahora ocupa la posición LETTERS[i]
            new_options = {LETTERS[i]: entry["options"][perm[i]] for i in range(4)}
            new_labels = {letter: label_option(txt) for letter, txt in new_options.items()}

            orig_answer_letter_new = next(l for i, l in enumerate(LETTERS) if perm[i] == entry["answer"])
            orig_class = new_labels[orig_answer_letter_new]
            answer_inverted_new = ""
            if orig_class in INVERT_CLASS:
                target = INVERT_CLASS[orig_class]
                matches = [l for l, lab in new_labels.items() if lab == target]
                if len(matches) == 1:
                    answer_inverted_new = matches[0]
            if not answer_inverted_new:
                n_inv_excluded += 1

            prompt_text = TEMPLATE.format(
                question=entry["question"],
                option_a=new_options["A"], option_b=new_options["B"],
                option_c=new_options["C"], option_d=new_options["D"],
            )

            row = {
                "qid": qid, "answer_permuted": orig_answer_letter_new,
                "answer_inverted_permuted": answer_inverted_new,
                "content_type": sample["content_type"], "region": sample["region"],
                "cells_max": sample["cells_max"], "condition": "swap_permuted",
                "correct_vs_original": "", "correct_vs_inverted": "",
            }
            try:
                image_prior_real = Image.open(sample["image_prior_path"]).convert("RGB")
                image_current_real = Image.open(sample["image_current_path"]).convert("RGB")
                # mismo swap que oracle_evaluate.py --condition swap: la previa real entra
                # como argumento "actual" y la actual real como argumento "previa".
                pred_raw = model.predict(image_current_real, image_prior_real, prompt_text)
            except Exception as exc:  # noqa: BLE001
                row.update(pred_raw="", pred_letter="", status="error")
                print(f"[error] qid={qid}: {exc}", file=sys.stderr)
                n_error += 1
                n_total += 1
                writer.writerow(row)
                out_f.flush()
                continue

            pred_letter = parse_letter(pred_raw)
            if pred_letter is None:
                row.update(pred_raw=pred_raw, pred_letter="", status="unparseable")
                n_unparseable += 1
            else:
                row.update(pred_raw=pred_raw, pred_letter=pred_letter, status="ok",
                           correct_vs_original=pred_letter == orig_answer_letter_new,
                           correct_vs_inverted=(pred_letter == answer_inverted_new)
                           if answer_inverted_new else "")
            n_total += 1
            writer.writerow(row)
            out_f.flush()

            if n_total % 500 == 0:
                print(f"  ...{n_total}/{len(samples)}")

    print(f"\n=== swap_permuted, modelo={args.model} ({n_total} muestras) ===")
    print(f"unparseable: {n_unparseable}/{n_total}  error: {n_error}/{n_total}")
    print(f"etiqueta invertida no construible: {n_inv_excluded}/{n_total}")
    print(f"resultados: {args.output}")


if __name__ == "__main__":
    main()
