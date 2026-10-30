#!/usr/bin/env python3
"""current_other for A on the clean set: the real prior image stays and only the current image is replaced by the
current image of the source row of the same patient derangement (seed 0) that `mismatch` and `both_other` use
(tac_mcq_train/run_eval_clean.py --condition current_other). Compared with `mismatch` (current real, prior from
another patient) and `both_other`.

Per condition: accuracy, condition − full paired by row, verdict as in Table 1 (Sig. = |Δ| ≥ 5pp with the 95% CI
excluding zero; Eq. = 90% CI inside ±5pp; otherwise Inconcl.) and Δpred (rows whose letter changes against full).
Patient bootstrap (paper_tables.Boot: 5,000 replicates, seed 0), 95% CI, pp. Aggregates only.

Usage: python3 clean_eval/current_other_report.py   -> clean_eval/current_other_report.md
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from paper_tables import Boot, load, patient_maps  # noqa: E402

A_DIR = REPO / "tac_mcq_train/runs/tac_pre_clean"
CONDS = [("current_other", "prior real, current from another patient"),
         ("mismatch", "current real, prior from another patient"),
         ("both_other", "both from another patient")]
DELTA = 0.05


def verdict(t):
    if abs(t[0]) >= DELTA and (t[1] > 0 or t[2] < 0):
        return "Sig."
    if -DELTA < t[3] and t[4] < DELTA:
        return "Eq."
    return "Inconcl."


def main():
    full = load(A_DIR / "clean_full.csv")
    qs = sorted(full)
    bt = Boot(qs, patient_maps()["clean"])
    corr = lambda R: np.array([R[q]["correct"] == "True" for q in qs], float)
    f = corr(full)
    md = ["# current_other for A (clean set)\n",
          "Paired by row, patient bootstrap (5,000, seed 0), 95% CI, pp; verdicts as in Table 1.\n",
          "| condition (A) | acc. | − full | verdict | Δpred |", "|---|---|---|---|---|",
          f"| full | {f.mean():.3f} | — | — | — |"]
    for c, what in CONDS:
        R = load(A_DIR / f"clean_{c}.csv")
        assert set(qs) <= set(R), c
        x = corr(R)
        t = bt.ci(x - f)
        d = bt.ci(np.array([R[q]["pred_letter"] != full[q]["pred_letter"] for q in qs], float))
        md.append(f"| {c} ({what}) | {x.mean():.3f} | {t[0]*100:+.1f} [{t[1]*100:+.1f}, {t[2]*100:+.1f}] | "
                  f"{verdict(t)} | {d[0]*100:.1f}% [{d[1]*100:.1f}, {d[2]*100:.1f}] |")
    md.append("")
    (HERE / "current_other_report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
