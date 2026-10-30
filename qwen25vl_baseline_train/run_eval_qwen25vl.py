#!/usr/bin/env python3
"""Evalua el baseline generico Qwen2.5-VL-3B (con o sin adaptador LoRA) sobre MMXU-test con el
binario EXISTENTE mmxu_eval/oracle_evaluate.py -- mismo mecanismo de registro en caliente que
tac_mcq_train/run_eval.py.

A diferencia del entrenamiento (resolucion igualada a RAD-DINO, 518x518), la EVALUACION usa los
MIN/MAX_PIXELS ESTANDAR del protocolo del ISBI (los mismos que oracle_evaluate.py inyecta para
TODOS los VLM, incluido Qwen2.5-VL-7B en produccion) -- el usuario pidio explicitamente que la
evaluacion sea "exactamente con las mismas condiciones" que el resto del protocolo.

Uso:
  python3 run_eval_qwen25vl.py --condition full     --adapter runs/full/adapter --output runs/full/oracle_full.csv
  python3 run_eval_qwen25vl.py --condition blind    --adapter runs/full/adapter --output runs/full/oracle_blind.csv
  python3 run_eval_qwen25vl.py --condition mismatch --adapter runs/full/adapter --output runs/full/oracle_mismatch.csv
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "mmxu_eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["full", "blind", "mismatch", "swap", "hole_random"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="solo para smoke test")
    ap.add_argument("--pixels", type=int, default=None,
                    help="fija min_pixels=max_pixels (p.ej. 268324 = 518^2, la resolucion de entrenamiento del generico); por defecto, los del protocolo")
    args = ap.parse_args()

    import models
    from model_qwen25vl import Qwen25VLTrainBaseline
    models.MODEL_REGISTRY["qwen25vl_baseline"] = Qwen25VLTrainBaseline
    import oracle_evaluate   # MIN_PIXELS/MAX_PIXELS estandar del protocolo, inyectados por run()
    if args.pixels is not None:
        # run() inyecta los globales MIN_PIXELS/MAX_PIXELS del modulo al llamarse.
        oracle_evaluate.MIN_PIXELS = oracle_evaluate.MAX_PIXELS = args.pixels

    model_kwargs = {"adapter_path": args.adapter, "max_new_tokens": args.max_new_tokens}
    oracle_evaluate.run(args.condition, "qwen25vl_baseline", model_kwargs, args.limit, Path(args.output))


if __name__ == "__main__":
    main()
