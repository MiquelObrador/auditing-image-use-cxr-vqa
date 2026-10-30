#!/usr/bin/env python3
"""Evalua el modelo sobre `val_fast` (2.000 filas de MMXU-dev, disjuntas por paciente).

Se usa dos veces: para la LINEA BASE sin entrenar (referencia exigida por el preregistro,
seccion 3) y para el modelo del piloto. Sin adaptador -> linea base; con --adapter -> entrenado.

Evalua SOLO el formato A (letra sola), con el orden de opciones ORIGINAL y sin permutar: es
el formato del protocolo de evaluacion, y queremos que estos numeros sean comparables con los
de MMXU-test.

Uso:
  python3 eval_valfast.py --out baseline_valfast.csv
  python3 eval_valfast.py --adapter runs/pilot/adapter --out pilot_valfast.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from config import MIMIC_JPG_ROOT, MMXU_DEV_JSONL  # noqa: E402
sys.path.insert(0, str(REPO / "mmxu_eval"))

from prompts import build_prompt        # noqa: E402
from evaluate import parse_letter       # noqa: E402
from model_tac import TacMcqBaseline    # noqa: E402
from format_targets import SPLITS_FILE  # noqa: E402

IMG_ROOT = MIMIC_JPG_ROOT
FIELDS = ["qid", "pred_raw", "pred_letter", "answer", "correct", "content_type", "status"]


def load_rows(split="val_fast"):
    ids = set(json.loads(SPLITS_FILE.read_text())[split])
    rows = []
    with open(MMXU_DEV_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in ids:
                rows.append(d)
    assert len(rows) == len(ids), f"faltan filas: {len(rows)} != {len(ids)}"
    return sorted(rows, key=lambda d: d["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="ruta del adaptador LoRA; sin esto, linea base")
    ap.add_argument("--tac-state", default=None, help="tac_state_dict.bin de una carrera con TAC entrenado")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val_fast")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    args = ap.parse_args()

    rows = load_rows(args.split)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} filas | adaptador: {args.adapter or 'NINGUNO (linea base sin entrenar)'}")

    model = TacMcqBaseline(adapter_path=args.adapter, tac_state_path=args.tac_state,
                           max_new_tokens=args.max_new_tokens)
    model.load()
    print("modelo cargado\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_correct = n_unparseable = n_error = 0
    t0 = time.perf_counter()

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, d in enumerate(rows):
            row = {"qid": d["id"], "answer": d["answer"], "content_type": d["content_type"]}
            try:
                cur = Image.open(IMG_ROOT / d["cur_image_path"]).convert("RGB")
                prior = Image.open(IMG_ROOT / d["prior_image_path"]).convert("RGB")
                raw = model.predict(prior, cur, build_prompt(d))
            except Exception as e:                      # noqa: BLE001
                row.update(pred_raw=f"{type(e).__name__}: {e}", pred_letter="",
                           correct=False, status="error")
                n_error += 1
                w.writerow(row); fh.flush()
                continue
            letter = parse_letter(raw)
            if letter is None:
                row.update(pred_raw=raw, pred_letter="", correct=False, status="unparseable")
                n_unparseable += 1
            else:
                correct = letter == d["answer"]
                row.update(pred_raw=raw, pred_letter=letter, correct=correct, status="ok")
                n_ok += 1
                n_correct += correct
            w.writerow(row); fh.flush()
            if (i + 1) % 200 == 0:
                el = time.perf_counter() - t0
                print(f"  {i+1}/{len(rows)}  acc={n_correct/(i+1):.3f}  "
                      f"{el/(i+1):.2f}s/muestra  ETA {(len(rows)-i-1)*el/(i+1)/60:.0f} min")

    n = len(rows)
    print(f"\n=== {args.split} | adaptador: {args.adapter or 'ninguno'} ===")
    print(f"  accuracy global (unparseable+error = incorrecto): {n_correct}/{n} = {n_correct/n:.3f}")
    if n_ok:
        print(f"  accuracy solo sobre parseable                  : {n_correct}/{n_ok} = {n_correct/n_ok:.3f}")
    print(f"  parseo estricto  : {n_ok}/{n} = {n_ok/n:.3f}")
    print(f"  unparseable      : {n_unparseable}/{n}")
    print(f"  error            : {n_error}/{n}")
    print(f"  tiempo total     : {(time.perf_counter()-t0)/60:.1f} min")
    print(f"\n  referencias: azar=0.250 | suelo de nombre de region=0.395")
    print(f"  resultados por muestra: {out_path}")


if __name__ == "__main__":
    main()
