#!/usr/bin/env python3
"""Paper numbers quoted in the text that no table script prints.

  1. Contamination: MMXU-test rows (of the 2,992 evaluated) whose patient is in the official
     MIMIC-CXR train split (paper: 2920, 97.6%).
  2. Qwen2-VL-7B under blind: answers that become correct vs. become wrong relative to full
     (paper: 256 gains against 298 losses).
  3. Text-only model (Qwen2.5-VL-3B fine-tuned without images) on the decontaminated set
     (paper: 0.782), and model C without images on the full MMXU-test (paper: 0.812).

Usage: python3 scripts/paper_numbers.py   (after the evaluation runs; see README)
"""
import csv
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from config import MMXU_TEST_JSONL, MIMIC_SPLIT_CSV, EVAL_QIDS  # noqa: E402


def load(path):
    return {int(r["qid"]): r for r in csv.DictReader(open(path))}


def main():
    qids = {int(r["qid"]) for r in csv.DictReader(open(EVAL_QIDS))}
    split_of = {r["subject_id"]: r["split"] for r in csv.DictReader(gzip.open(MIMIC_SPLIT_CSV, "rt"))}
    train = sum(split_of[json.loads(l)["cur_image_path"].split("/")[2][1:]] == "train"
                for l in open(MMXU_TEST_JSONL) if json.loads(l)["id"] in qids)
    print(f"1. MMXU-test rows from official-train patients: {train}/{len(qids)} ({100*train/len(qids):.1f}%)")

    d = REPO / "mmxu_eval/oracle_results"
    full, blind = load(d / "oracle_full.csv"), load(d / "oracle_blind.csv")
    gains = sum(full[q]["correct"] != "True" and blind[q]["correct"] == "True" for q in full)
    losses = sum(full[q]["correct"] == "True" and blind[q]["correct"] != "True" for q in full)
    print(f"2. Qwen2-VL-7B, blind vs full: {gains} gains, {losses} losses")

    q = REPO / "qwen25vl_baseline_train/runs"
    for label, path in (("text-only, decontaminated set", q / "textonly_clean/clean_blind.csv"),
                        ("C without images, full MMXU-test", q / "qwen_clean/px518/oracle_blind.csv")):
        r = load(path)
        print(f"3. {label}: {sum(x['correct'] == 'True' for x in r.values()) / len(r):.3f} (n={len(r)})")


if __name__ == "__main__":
    main()
