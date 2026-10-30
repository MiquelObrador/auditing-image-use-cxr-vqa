#!/usr/bin/env python3
"""Evalúa un VLMBaseline en zero-shot sobre MMXU-test (sin historical records).

Uso:
    python3 evaluate.py --model qwen2vl --limit 10 --output results_qwen2vl_dryrun.csv

No carga nada hasta el primer predict(): construir el objeto del modelo con
get_model() es barato, load() es lo que descarga/mueve pesos a GPU.
"""
import argparse
import csv
import re
import sys
import collections
from pathlib import Path

from PIL import Image

from data import load_samples
from models import get_model, MODEL_REGISTRY

LETTER_RE = re.compile(r"\b([A-Da-d])\b")

CSV_FIELDS = ["qid", "pred_raw", "pred_letter", "answer", "correct", "content_type", "status"]


def parse_letter(raw: str) -> str | None:
    """Primera letra A-D (case-insensitive) que aparezca como palabra suelta.

    None si no hay ninguna -- no se fuerza nunca una respuesta.
    """
    m = LETTER_RE.search(raw)
    return m.group(1).upper() if m else None


def run(model_name: str, model_kwargs: dict, limit: int | None, output_path: Path) -> None:
    model = get_model(model_name, **model_kwargs)
    model.load()

    n_total = n_correct = n_unparseable = n_error = 0
    by_content_type = collections.defaultdict(lambda: {"n": 0, "correct": 0})

    with open(output_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for i, sample in enumerate(load_samples()):
            if limit is not None and i >= limit:
                break

            row = {
                "qid": sample["qid"],
                "answer": sample["answer"],
                "content_type": sample["content_type"],
            }
            try:
                image_prior = Image.open(sample["image_prior_path"]).convert("RGB")
                image_current = Image.open(sample["image_current_path"]).convert("RGB")
                pred_raw = model.predict(image_prior, image_current, sample["prompt_text"])
            except Exception as exc:  # noqa: BLE001 -- una muestra rota no debe tumbar las otras 2999
                row.update(pred_raw="", pred_letter="", correct=False, status="error")
                print(f"[error] qid={sample['qid']}: {exc}", file=sys.stderr)
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
                correct = pred_letter == sample["answer"]
                row.update(pred_raw=pred_raw, pred_letter=pred_letter, correct=correct, status="ok")
                if correct:
                    n_correct += 1
                    by_content_type[sample["content_type"]]["correct"] += 1

            by_content_type[sample["content_type"]]["n"] += 1
            n_total += 1
            writer.writerow(row)
            out_f.flush()  # progreso a salvo si se corta a mitad de las 3000

    n_parseable = n_total - n_unparseable - n_error
    print(f"\n=== resumen ({n_total} muestras) ===")
    print(f"accuracy global (unparseable+error cuentan como incorrecto): "
          f"{n_correct}/{n_total} = {n_correct/n_total:.3f}" if n_total else "n_total=0")
    if n_parseable:
        print(f"accuracy solo sobre lo parseable: {n_correct}/{n_parseable} = {n_correct/n_parseable:.3f}")
    print(f"unparseable: {n_unparseable}/{n_total} ({n_unparseable/n_total:.1%})" if n_total else "")
    print(f"error (imagen/generación): {n_error}/{n_total} ({n_error/n_total:.1%})" if n_total else "")
    print("\naccuracy por content_type:")
    for ct, stats in sorted(by_content_type.items()):
        acc = stats["correct"] / stats["n"] if stats["n"] else float("nan")
        print(f"  {ct:12s} n={stats['n']:4d}  accuracy={acc:.3f}")
    print(f"\nresultados por muestra: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2vl", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--model-name", default=None, help="override del checkpoint HF (p.ej. otra revisión)")
    parser.add_argument("--limit", type=int, default=None, help="evaluar solo las primeras N muestras")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    args = parser.parse_args()

    model_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.model_name:
        model_kwargs["model_name"] = args.model_name

    run(args.model, model_kwargs, args.limit, args.output)


if __name__ == "__main__":
    main()
