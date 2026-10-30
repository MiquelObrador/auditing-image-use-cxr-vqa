#!/usr/bin/env python3
"""Puerta del piloto de B2 etapa 1 (PREREGISTRO_SPLIT_LIMPIO_2026-09-24, anexo B2).

Etapa 1 no tiene LoRA: el LLM esta intacto (nunca se le enseño el formato de respuesta), asi
que el parseo estricto de eval_valfast.py NO es un chequeo valido aqui (eso es tarea de la
etapa 2). La puerta comprueba solo que el TAC, unico modulo entrenable, este aprendiendo algo:

  (a) loss finita en todo el piloto;
  (b) la loss baja de forma clara: media de los 5 ultimos registros <= 95% de la media de los
      5 primeros (un margen mas estricto que la puerta de B, que no tenia con que contrastar
      la caida salvo el propio parseo).

Uso: python3 pilot_gate_stage1.py --dir runs/tac_b2_stage1_pilot
"""
import argparse
import json
import math
import sys
from pathlib import Path

DROP_FRAC = 0.95  # el promedio final debe quedar a <=95% del inicial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    st = json.loads((Path(args.dir) / "train_stats.json").read_text())
    losses = [l for _, l in st["loss_curve"]]

    a = len(losses) >= 10 and all(math.isfinite(l) for l in losses) and math.isfinite(st["train_loss"])
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    b = last <= DROP_FRAC * first

    report = [f"# Puerta del piloto — B2 etapa 1 — {args.dir}\n",
              f"- (a) loss finita: {'SI' if a else 'NO'} ({len(losses)} registros)",
              f"- (b) loss baja >= {100*(1-DROP_FRAC):.0f}%: {'SI' if b else 'NO'} "
              f"(5 primeros {first:.4f} -> 5 ultimos {last:.4f}, "
              f"caida {100*(1-last/first):.1f}%)",
              f"\n**{'PASA' if a and b else 'NO PASA'}**"]
    text = "\n".join(report) + "\n"
    (Path(args.dir) / "pilot_gate_stage1.md").write_text(text)
    print(text)
    sys.exit(0 if a and b else 1)


if __name__ == "__main__":
    main()
