#!/usr/bin/env bash
# All statistics, tables and numbers of the paper, from the saved predictions. CPU only,
# a few minutes. Patient-clustered bootstrap (N=5000, seed 0), TOST +-5pp with 90% CIs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"   # needs numpy, scipy, scikit-learn (envs/analysis.yml)
Q="$REPO/qwen25vl_baseline_train/runs/qwen_clean/px518"
cd "$REPO"

# Paper
$PY clean_eval/paper_parallel_table.py            # Table 1 (a: zero-shot, b: fine-tuned), Fig. 2 input
$PY clean_eval/hole_zeroshot_report.py            # Table 1a, hole column (hole - hole_random)
$PY clean_eval/hole_finetuned_report.py smoke     # Table 1b, hole column: 20-row check first
$PY clean_eval/hole_finetuned_report.py report    #   then hole - hole_random on 3,064 rows
$PY clean_eval/contamination_table.py             # Table 2 (MMXU-test vs. decontaminated set)
$PY clean_eval/current_other_report.py            # A: current scan replaced (11.9pp)
$PY clean_eval/swap_direction_audit.py            # swap reversal rate against mismatch (25% vs 14%)
$PY clean_eval/textonly_llama_report.py           # text-only Llama vs. A, B, B2
$PY clean_eval/per_class_report.py                # per-class analyses (blind shift towards "worsened")
$PY shortcut_analysis/options_only_linear.py      # linear model on the options alone (0.67)
$PY scripts/paper_numbers.py                      # contamination, blind revisions, text-only, C without images

# Supplementary and pre-registered reports
for d in tac_mcq_train/runs/tac_pre_clean tac_mcq_train/runs/tac_scratch_clean tac_mcq_train/runs/tac_b2_stage2 "$Q"; do
  $PY clean_eval/criteria_report_clean.py --dir "$d"   # pre-registered criteria, clean set
done
C_DIR="$Q" $PY clean_eval/compare_models.py        # A-C contrasts, difference-in-differences
$PY clean_eval/control_report.py                  # constant-token control
$PY clean_eval/both_other_zeroshot_report.py      # constant-token control, zero-shot VLMs
$PY clean_eval/swapdup_report.py                  # swap turn per class, fine-tuned models
$PY clean_eval/swap_inverted_recompute.py         # swap, unambiguous inverted label (zero-shot)
$PY clean_eval/swap_turn_zeroshot.py              # swap turn per class, zero-shot VLMs
$PY clean_eval/paper_tables.py --c-dir "$Q"       # hard subset (and LaTeX tables)
$PY clean_eval/c_resolution_report.py             # C at protocol vs. training resolution
$PY shortcut_analysis/question_options_linear.py  # options + question
$PY shortcut_analysis/hedge_check.py              # hedged vs absolute wording
