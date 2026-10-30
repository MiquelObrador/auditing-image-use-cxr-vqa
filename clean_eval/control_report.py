#!/usr/bin/env python3
"""Control de tokens constantes (both_other) en el conjunto limpio: A, B2 y C.

accuracy de full, blind y both_other, y both_other−full, both_other−blind y blind−full,
pareados por fila. Fraccion del gate que sobrevive con tokens presentes:
(full − both_other) / (full − blind), con IC por bootstrap de pacientes. Bootstrap por paciente,
N=5000, semilla 0, IC95%. Lectura fijada en PREREGISTRO_SPLIT_LIMPIO_2026-09-24.md §4d.

Uso: python3 clean_eval/control_report.py
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MMXU_DEV_JSONL  # noqa: E402
N_BOOT, SEED, OFF = 5000, 0, 1_000_000
T, Q = REPO / "tac_mcq_train/runs", REPO / "qwen25vl_baseline_train/runs"
MODELS = {"A (TAC preentrenado)": T / "tac_pre_clean",
          "B2 (TAC desde cero, 2 etapas)": T / "tac_b2_stage2",
          "C (genérico)": Q / "qwen_clean"}


def patient_map():
    pat = {}
    for src, fn in (("test", MMXU_TEST_JSONL), ("dev", MMXU_DEV_JSONL)):
        for line in open(fn):
            d = json.loads(line)
            pat[d["id"] + (OFF if src == "dev" else 0)] = d["cur_image_path"].split("/")[2]
    return pat


def correct(path):
    return {int(r["qid"]): r["correct"] == "True" for r in csv.DictReader(open(path))}


def main():
    pat = patient_map()
    lines = ["# Control de tokens constantes (both_other) — conjunto limpio\n",
             "Las dos imágenes de otro paciente (misma fila fuente que `mismatch`, derangement por "
             f"paciente, semilla 0). Bootstrap por paciente, N={N_BOOT}, semilla {SEED}, IC95%.\n",
             "| modelo | acc. full | acc. blind | acc. both_other | blind−full (pp) | "
             "both_other−full (pp) | both_other−blind (pp) | fracción del gate que sobrevive |",
             "|---|---|---|---|---|---|---|---|"]
    for m, d in MODELS.items():
        c = {k: correct(d / f"clean_{k}.csv") for k in ("full", "blind", "both_other")}
        qs = sorted(set(c["full"]) & set(c["blind"]) & set(c["both_other"]))
        assert all(len(v) == len(qs) for v in c.values()), m
        f, b, o = (np.array([c[k][q] for q in qs], float) for k in ("full", "blind", "both_other"))
        u, inv = np.unique([pat[q] for q in qs], return_inverse=True)
        W = np.zeros((N_BOOT, len(u)))
        idx = np.random.default_rng(SEED).integers(0, len(u), (N_BOOT, len(u)))
        np.add.at(W, (np.arange(N_BOOT)[:, None], idx), 1.0)
        cnt = np.zeros(len(u)); np.add.at(cnt, inv, 1.0)

        def reps(arr):
            s = np.zeros(len(u)); np.add.at(s, inv, arr)
            return (W @ s) / (W @ cnt)

        def ci(arr, pp=True):
            r = reps(arr); lo, hi = np.percentile(r, [2.5, 97.5]); k = 100 if pp else 1
            return (f"{arr.mean()*k:+.2f} [{lo*k:+.2f},{hi*k:+.2f}]" if pp
                    else f"{arr.mean():.3f} [{lo:.3f},{hi:.3f}]")

        gate, lost = reps(f - b), reps(f - o)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_r = lost / gate
        pt = (f - o).mean() / (f - b).mean() if (f - b).mean() != 0 else float("nan")
        lo, hi = np.nanpercentile(frac_r, [2.5, 97.5])
        frac = f"{pt:.2f} [{lo:.2f},{hi:.2f}]" if abs((f - b).mean()) > 0.02 else "— (sin gate)"
        lines.append(f"| {m} (n={len(qs)}/{len(u)}) | {ci(f, False)} | {ci(b, False)} | "
                     f"{ci(o, False)} | {ci(b - f)} | {ci(o - f)} | {ci(o - b)} | {frac} |")

    lines += ["", "Fracción del gate que sobrevive = (full − both_other) / (full − blind): 1 = quitar "
              "la información correcta cuesta tanto como quitar los tokens; 0 = con tokens de otro "
              "paciente rinde como con los suyos. Solo se calcula si |blind−full| > 2pp."]
    text = "\n".join(lines) + "\n"
    (HERE / "control_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
