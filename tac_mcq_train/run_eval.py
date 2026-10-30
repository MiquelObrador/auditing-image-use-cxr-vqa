#!/usr/bin/env python3
"""Evalua TacMcqBaseline sobre MMXU-test con el binario EXISTENTE mmxu_eval/oracle_evaluate.py,
SIN modificar ni un fichero de mmxu_eval: se registra en caliente en MODEL_REGISTRY (mutacion
in-place del dict, visible desde oracle_evaluate.py porque `from models import ... MODEL_REGISTRY`
liga el mismo objeto, no una copia).

Mismo prompt, mismas condiciones, mismo CSV, mismo bootstrap que el resto del protocolo del ISBI.

Uso:
  python3 run_eval.py --condition full     --adapter runs/full/adapter --output runs/full/oracle_full.csv
  python3 run_eval.py --condition blind    --adapter runs/full/adapter --output runs/full/oracle_blind.csv
  python3 run_eval.py --condition mismatch --adapter runs/full/adapter --output runs/full/oracle_mismatch.csv
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from config import LIBRA_DIR  # noqa: E402
sys.path.insert(0, str(LIBRA_DIR))
sys.path.insert(0, str(REPO / "mmxu_eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["full", "blind", "mismatch", "swap", "hole_random"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tac-state", default=None,
                    help="tac_state_dict.bin de una ejecucion con --unfreeze-tac")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="solo para smoke test")
    args = ap.parse_args()

    import models
    from model_tac import TacMcqBaseline
    models.MODEL_REGISTRY["tacmcq"] = TacMcqBaseline   # mutacion in-place, no reasignacion
    import oracle_evaluate

    model_kwargs = {"adapter_path": args.adapter, "tac_state_path": args.tac_state,
                    "max_new_tokens": args.max_new_tokens}
    oracle_evaluate.run(args.condition, "tacmcq", model_kwargs, args.limit, Path(args.output))


if __name__ == "__main__":
    main()
