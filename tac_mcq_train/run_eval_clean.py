#!/usr/bin/env python3
"""Evalua TacMcqBaseline sobre el conjunto limpio (data_ids/clean_set_ids_2026-09-24.csv, filas
usables: 3.022 de MMXU-dev + 74 de MMXU-test, 331 pacientes del split oficial validate/test de
MIMIC-CXR) con el binario EXISTENTE mmxu_eval/oracle_evaluate.py, sin modificarlo: se
sustituyen en caliente dos globales del modulo que run() resuelve al llamarlas.

  * load_oracle_samples -> muestras del conjunto limpio, en el mismo formato. Prompt con el
    mismo prompts.build_prompt. qid: el de MMXU-test tal cual; MMXU-dev + DEV_QID_OFFSET (los
    qid de ambos ficheros colisionan). Las bbox (region de la pregunta en las dos imagenes) solo
    se cargan para hole/hole_random o con --box-subset: salen de clean_eval/clean_bbox_table.csv
    (clean_eval/build_clean_bbox_table.py, mismas reglas que bbox_analysis/mmxu_bbox_table.csv) y
    restringen la evaluacion a las filas con caja en las dos imagenes (3.064 de 3.096).
  * mismatch_prior_index -> derangement ALEATORIO POR PACIENTE (semilla 0): cada paciente
    recibe las previas de otro paciente distinto, y cada fila una previa al azar de ese
    paciente. El desplazamiento de una fila del protocolo de MMXU-test no sirve aqui: las
    filas de MMXU-dev vienen agrupadas (~10 por paciente) y casi todas recibirian una previa
    del mismo paciente.

Uso:
  python3 run_eval_clean.py --condition full --adapter runs/X/adapter --output runs/X/clean_full.csv
  python3 run_eval_clean.py --condition hole --adapter runs/X/adapter --output runs/X/clean_hole.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from config import LIBRA_DIR, MIMIC_JPG_ROOT, MMXU_DEV_JSONL, MMXU_TEST_JSONL, CLEAN_SET_CSV  # noqa: E402
sys.path.insert(0, str(LIBRA_DIR))
sys.path.insert(0, str(REPO / "mmxu_eval"))

CLEAN_CSV = CLEAN_SET_CSV
IMG_ROOT = MIMIC_JPG_ROOT
BOX_CSV = REPO / "clean_eval" / "clean_bbox_table.csv"
DEV_QID_OFFSET = 1_000_000
SEED = 0
BOX_CONDITIONS = ("hole", "hole_random")


def clean_samples():
    from prompts import build_prompt

    keep = {(r["source"], int(r["qid"])) for r in csv.DictReader(open(CLEAN_CSV))
            if r["usable"] == "True"}
    out = []
    for source, fname in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
        with open(fname) as f:
            for line in f:
                e = json.loads(line)
                if (source, e["id"]) not in keep:
                    continue
                out.append({
                    "qid": e["id"] + (DEV_QID_OFFSET if source == "dev" else 0),
                    "prompt_text": build_prompt(e),
                    "image_prior_path": IMG_ROOT / e["prior_image_path"],
                    "image_current_path": IMG_ROOT / e["cur_image_path"],
                    "answer": e["answer"],
                    "content_type": e["content_type"],
                    "options": e["options"],
                    "region": "", "cells_max": "",
                    # apply_condition las recibe tambien en `full`, que las ignora.
                    "bbox_prior": (0, 0, 0, 0), "bbox_cur": (0, 0, 0, 0),
                    "patient": e["cur_image_path"].split("/")[2],
                })
    assert len(out) == len(keep), f"{len(out)} muestras vs {len(keep)} usables en el CSV"
    return out


def with_boxes(samples):
    """Filas con la caja de la region en las dos imagenes, con region, bbox_prior y bbox_cur
    (pixeles, recortadas a la imagen) rellenas; el resto de la fila, intacta."""
    tab = {int(r["qid"]): r for r in csv.DictReader(open(BOX_CSV))}
    box = lambda r, p: tuple(int(r[f"bbox_{p}_{k}"]) for k in ("x1", "y1", "x2", "y2"))
    return [{**s, "region": tab[s["qid"]]["region"], "bbox_cur": box(tab[s["qid"]], "cur"),
             "bbox_prior": box(tab[s["qid"]], "prior")} for s in samples if s["qid"] in tab]


def patient_derangement(samples, seed=SEED):
    rng = np.random.default_rng(seed)
    by_pat = {}
    for i, s in enumerate(samples):
        by_pat.setdefault(s["patient"], []).append(i)
    pats = sorted(by_pat)
    order = [pats[i] for i in rng.permutation(len(pats))]
    target = {order[k]: order[(k + 1) % len(order)] for k in range(len(order))}
    idx = np.array([by_pat[target[s["patient"]]][rng.integers(len(by_pat[target[s["patient"]]]))]
                    for s in samples])
    assert all(samples[j]["patient"] != s["patient"] for s, j in zip(samples, idx))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tacmcq", choices=["tacmcq", "qwen25vl_baseline"],
                    help="qwen25vl_baseline: correr en el env qwen25vl_baseline_train")
    ap.add_argument("--condition", required=True,
                    choices=["full", "blind", "mismatch", "both_other", "current_other", "swap",
                             "duplicate", "hole", "hole_random"])
    ap.add_argument("--box-subset", action="store_true",
                    help="restringe a las filas con caja (lo hacen siempre hole y hole_random); para full")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tac-state", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="solo para smoke test")
    ap.add_argument("--pixels", type=int, default=None,
                    help="fija min_pixels=max_pixels (p.ej. 268324 = 518^2, la resolucion de entrenamiento del generico); por defecto, los del protocolo")
    args = ap.parse_args()

    import models
    if args.model == "tacmcq":
        from model_tac import TacMcqBaseline
        models.MODEL_REGISTRY["tacmcq"] = TacMcqBaseline
        model_kwargs = {"adapter_path": args.adapter, "tac_state_path": args.tac_state,
                        "max_new_tokens": args.max_new_tokens}
    else:
        assert args.tac_state is None, "--tac-state solo aplica al modelo TAC"
        sys.path.insert(0, str(REPO / "qwen25vl_baseline_train"))
        from model_qwen25vl import Qwen25VLTrainBaseline
        models.MODEL_REGISTRY["qwen25vl_baseline"] = Qwen25VLTrainBaseline
        model_kwargs = {"adapter_path": args.adapter, "max_new_tokens": args.max_new_tokens}
    import oracle_evaluate   # MIN_PIXELS/MAX_PIXELS estandar del protocolo, inyectados por run()
    if args.pixels is not None:
        # run() inyecta los globales MIN_PIXELS/MAX_PIXELS del modulo al llamarse.
        oracle_evaluate.MIN_PIXELS = oracle_evaluate.MAX_PIXELS = args.pixels

    samples = clean_samples()
    src_idx = patient_derangement(samples)
    if args.condition in BOX_CONDITIONS or args.box_subset:
        # el derangement se calcula sobre el conjunto entero y no se usa en estas condiciones
        assert args.condition in BOX_CONDITIONS + ("full",), "--box-subset solo con full"
        samples = with_boxes(samples)
    if args.condition == "both_other":
        # Control de tokens constantes: las DOS imagenes de otro paciente (la fila fuente del
        # mismo derangement que usa mismatch, asi que la previa es la misma que en mismatch y la
        # actual es la de esa misma fila fuente). Mismo numero de tokens visuales, informacion
        # correcta eliminada. Se sustituyen las rutas y la condicion pasa por `full` sin editar.
        samples = [{**s, "image_prior_path": samples[j]["image_prior_path"],
                    "image_current_path": samples[j]["image_current_path"],
                    "source_row": int(j)} for s, j in zip(samples, src_idx)]
        _apply = oracle_evaluate.apply_condition
        oracle_evaluate.apply_condition = (
            lambda img, bbox, cond: img if cond == "both_other" else _apply(img, bbox, cond))
    elif args.condition == "current_other":
        # Complemento de mismatch: la PREVIA real se queda y solo la ACTUAL pasa a ser la de la
        # fila fuente del mismo derangement por paciente (semilla 0) que usan mismatch y
        # both_other. Pasa por `full` sin editar pixeles.
        samples = [{**s, "image_current_path": samples[j]["image_current_path"],
                    "source_row": int(j)} for s, j in zip(samples, src_idx)]
        _apply = oracle_evaluate.apply_condition
        oracle_evaluate.apply_condition = (
            lambda img, bbox, cond: img if cond == "current_other" else _apply(img, bbox, cond))
    oracle_evaluate.load_oracle_samples = lambda: iter(samples)
    oracle_evaluate.mismatch_prior_index = lambda n: src_idx
    # swap/duplicate: build_qid_lexicon() solo lee MMXU-test.jsonl; aqui se construye con la
    # misma regla sobre las opciones de las muestras limpias (qid de dev desplazados).
    by_qid = {s["qid"]: s for s in samples}

    def clean_lexicon(_path):
        labels, inverted = {}, {}
        for q, s in by_qid.items():
            labs = {L: oracle_evaluate.label_option(t) for L, t in s["options"].items()}
            labels[q] = labs
            orig = labs.get(s["answer"])
            if orig not in oracle_evaluate.INVERT_CLASS:
                inverted[q] = None
                continue
            m = [L for L, lab in labs.items() if lab == oracle_evaluate.INVERT_CLASS[orig]]
            inverted[q] = m[0] if len(m) == 1 else None
        return labels, inverted
    oracle_evaluate.build_qid_lexicon = clean_lexicon
    print(f"conjunto limpio: {len(samples)} filas, "
          f"{len({s['patient'] for s in samples})} pacientes")

    oracle_evaluate.run(args.condition, args.model, model_kwargs, args.limit, Path(args.output))


if __name__ == "__main__":
    main()
