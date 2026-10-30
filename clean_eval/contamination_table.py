#!/usr/bin/env python3
"""Table 2 of the paper: accuracy of every fine-tuned model on MMXU-test (2,992 rows, 97.6% from official-train
patients) and on the decontaminated set (3,096 rows, 331 held-out patients), and the drop between the two.
A, B, B2 and C under `full`; the two text-only models under `blind` (they never see an image). C at its training
resolution (518², px518/). Accuracy over all rows, an unparsed answer counting as wrong.

Usage: python3 clean_eval/contamination_table.py   -> clean_eval/contamination_table.md
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from paper_tables import load  # noqa: E402

T, Q = REPO / "tac_mcq_train/runs", REPO / "qwen25vl_baseline_train/runs"
ROWS = [("A: pretrained TAC", T / "tac_pre_clean", "full"),
        ("B: TAC from scratch", T / "tac_scratch_clean", "full"),
        ("B2: TAC, two stages", T / "tac_b2_stage2", "full"),
        ("C: generic", Q / "qwen_clean/px518", "full"),
        ("Text only (Llama-3.2-3B)", T / "textonly_llama", "blind"),
        ("Text only (Qwen2.5-VL-3B)", Q / "textonly_clean", "blind")]


def acc(path):
    R = load(path)
    return sum(r["correct"] == "True" for r in R.values()) / len(R), len(R)


def main():
    md = ["# Accuracy on MMXU-test and on the decontaminated set (Table 2)\n",
          "| model | MMXU-test | decontaminated | drop (pp) |", "|---|---|---|---|"]
    for name, d, cond in ROWS:
        (a_t, n_t), (a_c, n_c) = acc(d / f"oracle_{cond}.csv"), acc(d / f"clean_{cond}.csv")
        assert (n_t, n_c) == (2992, 3096), (name, n_t, n_c)
        md.append(f"| {name} | {a_t:.3f} | {a_c:.3f} | {(a_t - a_c)*100:.1f} |")
    md.append("")
    (HERE / "contamination_table.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
