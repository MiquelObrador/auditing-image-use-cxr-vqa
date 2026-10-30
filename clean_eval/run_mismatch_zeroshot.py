#!/usr/bin/env python3
"""mismatch con derangement por paciente para los VLM zero-shot en MMXU-test completo.

Sustituye al mismatch del protocolo original (previa de la fila anterior, np.roll), que deja
filas con la previa de su mismo paciente. Aqui cada fila recibe la previa de una fila de OTRO
paciente, con el mismo procedimiento que en el conjunto limpio
(tac_mcq_train/run_eval_clean.py::patient_derangement, derangement aleatorio por paciente,
semilla 0) y en clean_eval/run_both_other_zeroshot.py, sobre las 2.992 filas en el mismo orden:
la previa es la misma que en both_other y la actual es la real. Mismo binario
(mmxu_eval/oracle_evaluate.run, condicion mismatch), mismo prompt, mismos pixeles y la misma
decodificacion; no se modifica ningun fichero de mmxu_eval.

Uso (en el env de cada modelo):
  python3 clean_eval/run_mismatch_zeroshot.py --model qwen2vl --output mmxu_eval/oracle_results/oracle_mismatch_patient.csv
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
    same = sum(samples[j]["patient"] == s["patient"] for s, j in zip(samples, src))
    assert same == 0, same
    oracle_evaluate.load_oracle_samples = lambda: iter(samples)
    oracle_evaluate.mismatch_prior_index = lambda n: src
    print(f"MMXU-test: {len(samples)} filas, {len({s['patient'] for s in samples})} pacientes; "
          f"previa de otro paciente (derangement por paciente, semilla 0); filas con previa del mismo "
          f"paciente: {same}")
    oracle_evaluate.run("mismatch", args.model, {"max_new_tokens": args.max_new_tokens},
                        args.limit, Path(args.output))


if __name__ == "__main__":
    main()
