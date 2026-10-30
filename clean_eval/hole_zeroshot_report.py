#!/usr/bin/env python3
"""`hole` column of Table 1a: for each zero-shot VLM on MMXU-test (2,992 rows, all with the region's box in both
images), accuracy under `hole` (grey over the region asked about) minus accuracy under `hole_random` (a grey patch of
the same size elsewhere), paired by row. Patient bootstrap (paper_tables.Boot: 5,000 replicates, seed 0), 95% CI and
its half-width, pp; Table 1a gives the zero-shot models no verdict. The fine-tuned models' column comes from
clean_eval/hole_finetuned_report.py.

Usage: python3 clean_eval/hole_zeroshot_report.py   -> clean_eval/hole_zeroshot_report.md
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from paper_tables import Boot, load, patient_maps  # noqa: E402

ZS = [("Qwen2-VL-7B", REPO / "mmxu_eval/oracle_results"), ("InternVL2-8B", REPO / "internvl_eval/oracle_results"),
      ("HuatuoGPT-7B", REPO / "huatuo_eval/oracle_results"), ("Qwen2.5-VL-7B", REPO / "qwen25vl_eval/oracle_results")]


def main():
    pat = patient_maps()["full"]
    md = ["# hole − hole_random, zero-shot VLMs, MMXU-test (Table 1a)\n",
          "| model | n | hole − hole_random (pp) | half-width | 95% CI |", "|---|---|---|---|---|"]
    for name, d in ZS:
        H, R = load(d / "oracle_hole.csv"), load(d / "oracle_hole_random.csv")
        qs = sorted(H)
        assert set(qs) == set(R), name
        c = lambda X: np.array([X[q]["correct"] == "True" for q in qs], float)
        t = Boot(qs, pat).ci(c(H) - c(R))
        md.append(f"| {name} | {len(qs)} | {t[0]*100:+.1f} | {(t[2]-t[1])/2*100:.1f} | "
                  f"[{t[1]*100:+.1f}, {t[2]*100:+.1f}] |")
    md.append("")
    (HERE / "hole_zeroshot_report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
