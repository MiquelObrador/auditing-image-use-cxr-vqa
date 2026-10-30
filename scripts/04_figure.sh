#!/usr/bin/env bash
# Fig. 2 (image benefit from content vs. from visual tokens), from the numbers of Table 1
# (clean_eval/paper_parallel_table.json, written by scripts/03_analysis.sh). Aggregates only.
# Fig. 1 is drawn in LaTeX (TikZ) in the paper and needs no script.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${PY:-python3}" "$REPO/figures/fig_audit_ladder.py"
