#!/usr/bin/env bash
# One-time setup: Libra code at the pinned commit, the TAC assembly, and the table of Chest
# ImaGenome boxes that defines the 2,992 evaluated MMXU-test rows.
# Data paths: config.py (data/ symlinks or environment variables, see README).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_TAC="${ENV_TAC:-tacmcq_train}"
ENV_MMXU="${ENV_MMXU:-mmxu_eval}"

LIBRA_DIR="${LIBRA_DIR:-$REPO/third_party/Libra}"
if [ ! -d "$LIBRA_DIR/.git" ]; then
  git clone https://github.com/X-iZhang/Libra.git "$LIBRA_DIR"
fi
git -C "$LIBRA_DIR" checkout 0d6c56a44b0f74259bdd8f05929465790b49f6e5   # tag v1.0.3

conda run --no-capture-output -n "$ENV_TAC" python "$REPO/tac_mcq_train/make_assembly.py"

# Boxes from Chest ImaGenome scene graphs -> bbox_analysis/mmxu_bbox_table.csv (2,992 rows)
cd "$REPO/bbox_analysis" && conda run --no-capture-output -n "$ENV_MMXU" python analyze_bbox_size.py
python3 - <<PY
import csv
q = sorted(int(r["id"]) for r in csv.DictReader(open("$REPO/bbox_analysis/mmxu_bbox_table.csv")))
s = sorted(int(r["qid"]) for r in csv.DictReader(open("$REPO/data_ids/mmxu_test_eval_qids.csv")))
assert q == s, f"box table gives {len(q)} rows, expected the 2992 in data_ids/mmxu_test_eval_qids.csv"
print("OK: the box table reproduces the 2992 evaluated rows")
PY
