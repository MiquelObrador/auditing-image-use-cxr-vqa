#!/usr/bin/env python3
"""Puerta del piloto del TAC desde cero (PREREGISTRO_SPLIT_LIMPIO_2026-09-24, §B.2).

Pasa solo si se cumplen las tres; si no, sale con codigo 1 y la cadena se detiene:
  (a) loss finita en todo el piloto;
  (b) la loss baja: media de los 5 ultimos registros < media de los 5 primeros;
  (c) parseo estricto en val_fast >= 95% (el TAC aleatorio no rompe la salida del LLM).
La accuracy se reporta, no se exige.

Uso: python3 pilot_gate.py --dir runs/tac_scratch_clean_pilot --valfast runs/.../valfast.csv
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

MIN_PARSE = 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--valfast", required=True)
    args = ap.parse_args()

    st = json.loads((Path(args.dir) / "train_stats.json").read_text())
    losses = [l for _, l in st["loss_curve"]]
    rows = list(csv.DictReader(open(args.valfast)))
    parse = sum(r["status"] == "ok" for r in rows) / len(rows)
    acc = sum(r["correct"] == "True" for r in rows) / len(rows)

    a = (len(losses) >= 10 and all(math.isfinite(l) for l in losses)
         and math.isfinite(st["train_loss"]))
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    b = last < first
    c = parse >= MIN_PARSE

    report = [f"# Puerta del piloto — {args.dir}\n",
              f"- (a) loss finita: {'SI' if a else 'NO'} ({len(losses)} registros)",
              f"- (b) loss baja: {'SI' if b else 'NO'} (5 primeros {first:.4f} -> 5 ultimos {last:.4f})",
              f"- (c) parseo estricto val_fast >= {MIN_PARSE:.0%}: {'SI' if c else 'NO'} ({parse:.3f})",
              f"- accuracy val_fast (informativa): {acc:.3f} (n={len(rows)})",
              f"\n**{'PASA' if a and b and c else 'NO PASA'}**"]
    text = "\n".join(report) + "\n"
    (Path(args.dir) / "pilot_gate.md").write_text(text)
    print(text)
    sys.exit(0 if a and b and c else 1)


if __name__ == "__main__":
    main()
