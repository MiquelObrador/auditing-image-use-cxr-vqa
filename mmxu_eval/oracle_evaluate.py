#!/usr/bin/env python3
"""Experimento oráculo: completa / solo-la-caja (recortada) / solo-la-caja (en su sitio) /
solo-fuera-de-la-caja / blind.

Reutiliza el pipeline de evaluate.py (mismo modelo, mismo parse_letter, misma
lógica de errores) pero sobre las 2992 muestras con bbox válida
(oracle_data.py) y aplicando una transformación de imagen según --condition.

Cinco condiciones:
  full  -- ambas imágenes completas.
  box   -- recortada exactamente a la bbox (imagen pequeña, sin contexto).
  keep  -- bbox en su posición/escala originales, resto tapado de gris. Con `box` es un
           control gratis: si coinciden, el encuadre no importa; si difieren, escala y
           posición sí. (Esta condición se llamó "mask" hasta que se descubrió que la
           documentación del experimento describía justo lo contrario de lo que hacía el
           código -- ver ../context.md, hallazgos consolidados.)
  hole  -- complemento exacto de `keep`: bbox tapada de gris, resto visible. Es la única
           condición que prueba si la región es NECESARIA (las demás, como mucho, si es
           SUFICIENTE).
  blind -- entrada visual a cero (sin abrir ningún fichero de imagen): mide cuánto usa el
           modelo la imagen en general, no solo la región. El prompt no cambia respecto a las
           otras condiciones (sigue mencionando Image-1/Image-2).

Las condiciones full/box/keep/hole fijan min_pixels/max_pixels estándar de Qwen2-VL
(256*28*28 .. 1280*28*28) -- necesario para que los recortes pequeños no caigan
por debajo del suelo de resolución, y para que "completa" sea comparable (no se
reusa results.csv, que se generó sin este límite explícito). Ver
../context.md / mmxu_eval/results_summary.md para el porqué. No aplica a
"blind" (sin imágenes, el límite de resolución es irrelevante) pero se deja
fijado igual por uniformidad de configuración del modelo entre condiciones.

Uso:
    python3 oracle_evaluate.py --condition full  --output oracle_full.csv
    python3 oracle_evaluate.py --condition box   --output oracle_box.csv
    python3 oracle_evaluate.py --condition keep  --output oracle_keep.csv
    python3 oracle_evaluate.py --condition hole  --output oracle_hole.csv
    python3 oracle_evaluate.py --condition blind --output oracle_blind.csv
"""
import argparse
import csv
import json
import re
import sys
import collections
from pathlib import Path

import numpy as np
from PIL import Image

from oracle_data import load_oracle_samples
from data import MMXU_JSONL_PATH
from oracle_transforms import crop_to_box, gray_outside_box, gray_inside_box
from hole_random_transforms import hole_random_bbox
from evaluate import parse_letter
from models import get_model, MODEL_REGISTRY

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

CSV_FIELDS = [
    "qid", "pred_raw", "pred_letter", "answer", "correct", "content_type",
    "status", "region", "cells_max", "condition",
    # solo rellenos para swap/mismatch/duplicate, vacíos en el resto:
    "answer_inverted", "correct_vs_inverted", "pred_is_no_change", "source_qid_prior",
]

# swap/mismatch/duplicate: NO editan píxeles, solo reasignan qué imagen ocupa qué papel
# (previa/actual) -- estructura TEMPORAL, no espacial. Ver TASK_temporal.md / plan de sesión.
TEMPORAL_CONDITIONS = ("swap", "mismatch", "duplicate")

# --- léxico opción->clase, copiado verbatim de bbox_analysis/analyze_region_prior.py (no se
# importa ese script: es un análisis con efectos secundarios al cargarse, no un módulo) --
# necesario para construir la etiqueta INVERTIDA de `swap` (worsen<->improved, no change
# igual). "worsen", no "worsened": mismo token que usa `content_type` en MMXU-test.jsonl.
IMPROVED_RE = re.compile(
    r"improv|resolv|better|decreas|reduc|clear(ed|ing)?\b|less |diminish|smaller|regress", re.I)
WORSENED_RE = re.compile(
    r"worse|worsen|increas|progress|larger|develop|new |appear|greater|more |enlarg|deteriorat", re.I)
NOCHANGE_RE = re.compile(
    r"no change|unchanged|no significant|stable|same|remain|not chang|no observ|persist|similar|no differ", re.I)
INVERT_CLASS = {"improved": "worsen", "worsen": "improved", "no change": "no change"}


def label_option(text: str) -> str:
    i = bool(IMPROVED_RE.search(text))
    w = bool(WORSENED_RE.search(text))
    nc = bool(NOCHANGE_RE.search(text))
    hits = [lab for lab, flag in (("no change", nc), ("improved", i), ("worsen", w)) if flag]
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        return "NONE"
    return "AMBIG"


def build_qid_lexicon(jsonl_path: Path) -> tuple[dict[int, dict[str, str]], dict[int, str | None]]:
    """Por qid: (a) `option_labels` -- {letra: clase} de sus 4 opciones vía léxico, usado por
    `duplicate` para comprobar si la letra predicha mapea a "no change" sin necesitar una
    única "letra de no-change" canónica; (b) letra correcta bajo la etiqueta INVERTIDA de
    `swap` (worsen<->improved, no change igual), o None si se excluye (0 o >1 opciones mapean
    a la clase invertida -- no se adivina un gold)."""
    option_labels_by_qid, inverted = {}, {}
    n_excluded = 0
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            options = entry["options"]
            labels = {letter: label_option(txt) for letter, txt in options.items()}
            option_labels_by_qid[entry["id"]] = labels
            orig_class = labels.get(entry["answer"])
            if orig_class not in INVERT_CLASS:
                inverted[entry["id"]] = None
                n_excluded += 1
                continue
            target = INVERT_CLASS[orig_class]
            matches = [letter for letter, lab in labels.items() if lab == target]
            inverted[entry["id"]] = matches[0] if len(matches) == 1 else None
            n_excluded += len(matches) != 1
    print(f"[swap] etiqueta invertida construible para {len(inverted) - n_excluded}/{len(inverted)} "
          f"preguntas ({n_excluded} excluidas: AMBIG/NONE/empate en la opción original o invertida)")
    return option_labels_by_qid, inverted


def mismatch_prior_index(n: int) -> np.ndarray:
    """Desplazamiento cíclico sin puntos fijos -- misma técnica que
    internvl_eval/shuffled_pairs_check.py, aquí solo aplicada a la PREVIA (la actual se queda
    real): `mismatch` prueba si la previa importa como referencia real, no si el modelo lee
    contenido de imagen en absoluto (esa prueba, más básica, ya la hace shuffled_pairs_check.py)."""
    return np.roll(np.arange(n), 1)


def apply_condition(image: Image.Image, bbox: tuple[int, int, int, int], condition: str) -> Image.Image:
    if condition == "full":
        return image
    if condition == "box":
        return crop_to_box(image, bbox)
    if condition == "keep":
        return gray_outside_box(image, bbox)
    if condition == "hole":
        return gray_inside_box(image, bbox)
    raise ValueError(f"condición desconocida: {condition}")


CONDITIONS = ["full", "box", "keep", "hole", "hole_random", "blind", "no_prior", "swap", "mismatch", "duplicate"]


def run(condition: str, model_name: str, model_kwargs: dict, limit: int | None, output_path: Path) -> None:
    model_kwargs = {**model_kwargs, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
    model = get_model(model_name, **model_kwargs)
    model.load()

    n_total = n_correct = n_unparseable = n_error = 0
    by_content_type = collections.defaultdict(lambda: {"n": 0, "correct": 0})

    is_temporal = condition in TEMPORAL_CONDITIONS
    option_labels_by_qid = inverted_by_qid = None
    all_samples = mismatch_src_idx = None
    if is_temporal:
        # el split cíclico de `mismatch` y la etiqueta invertida de `swap` se calculan SIEMPRE
        # sobre la población COMPLETA (2992), nunca sobre una versión truncada por --limit --
        # si no, con menos filas el desplazamiento cíclico apuntaría a otras preguntas.
        all_samples = list(load_oracle_samples())
        mismatch_src_idx = mismatch_prior_index(len(all_samples))
        if condition == "swap":
            option_labels_by_qid, inverted_by_qid = build_qid_lexicon(
                MMXU_JSONL_PATH)
        elif condition == "duplicate":
            option_labels_by_qid, _ = build_qid_lexicon(
                MMXU_JSONL_PATH)

    with open(output_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for i, sample in enumerate(load_oracle_samples()):
            if limit is not None and i >= limit:
                break

            row = {
                "qid": sample["qid"],
                "answer": sample["answer"],
                "content_type": sample["content_type"],
                "region": sample["region"],
                "cells_max": sample["cells_max"],
                "condition": condition,
                "answer_inverted": "", "correct_vs_inverted": "", "pred_is_no_change": "",
                "source_qid_prior": "",
            }
            try:
                if condition == "blind":
                    # Entrada visual a cero: ni se abren los ficheros de imagen.
                    pred_raw = model.predict(None, None, sample["prompt_text"])
                elif condition == "no_prior":
                    # Solo la actual: la previa NO se envia (image_prior=None), pero
                    # sample["prompt_text"] no se toca -- sigue diciendo literalmente
                    # "Previous chest X-RAY is Image-1. And current chest X-RAY is Image-2.",
                    # mencionando una Image-1 que no existe. Mismo criterio que "blind" (texto
                    # intacto, entrada visual la que cambia) -- NO es una reformulacion del
                    # prompt para avisar de que falta la previa.
                    image_current = Image.open(sample["image_current_path"]).convert("RGB")
                    pred_raw = model.predict(None, image_current, sample["prompt_text"])
                elif condition == "swap":
                    # Se intercambian los PAPELES: la previa real entra como argumento
                    # "actual" y la actual real como argumento "previa" -- el prompt no
                    # cambia (sigue diciendo "Previous... is Image-1. And current... is
                    # Image-2."), así que si el modelo lee contenido real, debería fallar
                    # contra la etiqueta original y acertar contra la invertida.
                    image_prior_real = Image.open(sample["image_prior_path"]).convert("RGB")
                    image_current_real = Image.open(sample["image_current_path"]).convert("RGB")
                    pred_raw = model.predict(image_current_real, image_prior_real, sample["prompt_text"])
                elif condition == "duplicate":
                    image_current = Image.open(sample["image_current_path"]).convert("RGB")
                    pred_raw = model.predict(image_current, image_current, sample["prompt_text"])
                elif condition == "mismatch":
                    src = all_samples[mismatch_src_idx[i]]
                    row["source_qid_prior"] = src["qid"]
                    image_prior_src = Image.open(src["image_prior_path"]).convert("RGB")
                    image_current = Image.open(sample["image_current_path"]).convert("RGB")
                    pred_raw = model.predict(image_prior_src, image_current, sample["prompt_text"])
                elif condition == "hole_random":
                    # Control espacial: tapa un parche del mismo tamaño que la bbox real pero
                    # en la posición homóloga contraria (o desplazada, regiones sin pareja),
                    # por imagen -- bbox_prior/bbox_cur se recalculan cada uno sobre su propio
                    # tamaño de imagen real (prior y actual pueden diferir en resolución).
                    image_prior = Image.open(sample["image_prior_path"]).convert("RGB")
                    image_current = Image.open(sample["image_current_path"]).convert("RGB")
                    bbox_prior_rand = hole_random_bbox(sample["region"], sample["bbox_prior"], image_prior.size)
                    bbox_cur_rand = hole_random_bbox(sample["region"], sample["bbox_cur"], image_current.size)
                    image_prior = gray_inside_box(image_prior, bbox_prior_rand)
                    image_current = gray_inside_box(image_current, bbox_cur_rand)
                    pred_raw = model.predict(image_prior, image_current, sample["prompt_text"])
                else:
                    image_prior = Image.open(sample["image_prior_path"]).convert("RGB")
                    image_current = Image.open(sample["image_current_path"]).convert("RGB")
                    image_prior = apply_condition(image_prior, sample["bbox_prior"], condition)
                    image_current = apply_condition(image_current, sample["bbox_cur"], condition)
                    pred_raw = model.predict(image_prior, image_current, sample["prompt_text"])
            except Exception as exc:  # noqa: BLE001 -- una muestra rota no debe tumbar las demás
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

                if condition == "swap":
                    inv_letter = inverted_by_qid.get(sample["qid"])
                    if inv_letter is not None:
                        row["answer_inverted"] = inv_letter
                        row["correct_vs_inverted"] = pred_letter == inv_letter
                elif condition == "duplicate":
                    opts = option_labels_by_qid[sample["qid"]]
                    row["pred_is_no_change"] = opts.get(pred_letter) == "no change"

            by_content_type[sample["content_type"]]["n"] += 1
            n_total += 1
            writer.writerow(row)
            out_f.flush()

    n_parseable = n_total - n_unparseable - n_error
    print(f"\n=== resumen condición={condition} ({n_total} muestras) ===")
    if n_total:
        print(f"accuracy global (unparseable+error cuentan como incorrecto): "
              f"{n_correct}/{n_total} = {n_correct/n_total:.3f}")
    if n_parseable:
        print(f"accuracy solo sobre lo parseable: {n_correct}/{n_parseable} = {n_correct/n_parseable:.3f}")
    if n_total:
        print(f"unparseable: {n_unparseable}/{n_total} ({n_unparseable/n_total:.1%})")
        print(f"error (imagen/generación): {n_error}/{n_total} ({n_error/n_total:.1%})")
    print("\naccuracy por content_type:")
    for ct, stats in sorted(by_content_type.items()):
        acc = stats["correct"] / stats["n"] if stats["n"] else float("nan")
        print(f"  {ct:12s} n={stats['n']:4d}  accuracy={acc:.3f}")
    print(f"\nresultados por muestra: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--model", default="qwen2vl", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.model_name:
        model_kwargs["model_name"] = args.model_name

    run(args.condition, args.model, model_kwargs, args.limit, args.output)


if __name__ == "__main__":
    main()
