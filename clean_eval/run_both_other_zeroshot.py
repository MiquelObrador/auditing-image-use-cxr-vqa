#!/usr/bin/env python3
"""Control de tokens constantes (both_other) para los VLM zero-shot en MMXU-test completo.

Las dos imagenes de cada fila se sustituyen por las de una fila de OTRO paciente, con el mismo
procedimiento que en el conjunto limpio (tac_mcq_train/run_eval_clean.py::patient_derangement,
derangement aleatorio por paciente, semilla 0), aplicado a las 2.992 filas de MMXU-test. Mismo
binario (mmxu_eval/oracle_evaluate.run), mismo prompt, mismos pixeles y la misma decodificacion
que las demas condiciones de cada modelo; no se modifica ningun fichero de mmxu_eval.

Uso (en el env de cada modelo):
  python3 clean_eval/run_both_other_zeroshot.py --model qwen2vl --output mmxu_eval/oracle_results/oracle_both_other.csv
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "mmxu_eval"))
sys.path.insert(0, str(REPO / "tac_mcq_train"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen2vl", "internvl2", "huatuo", "qwen25vl"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="solo para smoke test")
    args = ap.parse_args()

    import oracle_evaluate
    from run_eval_clean import patient_derangement

    samples = [{**s, "patient": Path(s["image_current_path"]).parts[-3]}
               for s in oracle_evaluate.load_oracle_samples()]
    assert len(samples) == 2992, len(samples)
    src = patient_derangement(samples)
    samples = [{**s, "image_prior_path": samples[j]["image_prior_path"],
                "image_current_path": samples[j]["image_current_path"]} for s, j in zip(samples, src)]
    assert all(samples[j]["patient"] != s["patient"] for s, j in zip(samples, src))
    oracle_evaluate.load_oracle_samples = lambda: iter(samples)
    _apply = oracle_evaluate.apply_condition
    oracle_evaluate.apply_condition = (
        lambda img, bbox, cond: img if cond == "both_other" else _apply(img, bbox, cond))
    print(f"MMXU-test: {len(samples)} filas, {len({s['patient'] for s in samples})} pacientes; "
          f"ambas imagenes de otro paciente (derangement por paciente, semilla 0)")
    oracle_evaluate.run("both_other", args.model, {"max_new_tokens": args.max_new_tokens},
                        args.limit, Path(args.output))


if __name__ == "__main__":
    main()
