#!/usr/bin/env bash
# Fine-tuned models on the clean split (data_ids/splits_clean_2026-09-24.json): Table 1b,
# Table 2 and the per-class and swap analyses. ~90 GPU hours on one 24 GB GPU.
#   A   pretrained TAC (Libra), frozen; LoRA on Llama-3.2-3B         ~12 h
#   B   TAC from scratch, trained jointly with the LoRA (pilot+gate)  ~13 h
#   B2  TAC from scratch, two stages: TAC alone, then the LoRA       ~20 h
#   C   generic Qwen2.5-VL-3B-Instruct, LoRA, images at 518^2         ~16 h
#   text-only Qwen2.5-VL-3B (C's model) and Llama-3.2-3B (A/B/B2's),
#   same recipes, never shown an image                               ~4 h each
# Run 00_prepare.sh first. Pre-registration: docs/preregistrations/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$REPO/tac_mcq_train"; Q="$REPO/qwen25vl_baseline_train"
PYT="conda run --no-capture-output -n ${ENV_TAC:-tacmcq_train} python"
PYQ="conda run --no-capture-output -n ${ENV_QWEN:-qwen25vl_baseline_train} python"
PX518=268324   # 518*518: C is trained and evaluated at RAD-DINO's input area
step() { echo; echo "=== $1  $(date '+%F %T') ==="; }

eval_tac() {   # $1 = run dir, $2.. = extra flags (adapter / tac-state)
  local d=$1; shift
  for c in full blind mismatch both_other swap duplicate hole hole_random; do
    step "$(basename $d) clean $c"; $PYT run_eval_clean.py --condition $c "$@" --output $d/clean_$c.csv
  done
  # 20-row check that the box subset reproduces `full` (clean_eval/hole_finetuned_report.py smoke)
  mkdir -p runs/hole_smoke
  $PYT run_eval_clean.py --condition full --box-subset --limit 20 "$@" --output runs/hole_smoke/$(basename $d)_full.csv
  for c in hole hole_random; do
    $PYT run_eval_clean.py --condition $c --limit 20 "$@" --output runs/hole_smoke/$(basename $d)_$c.csv
  done
  for c in full blind mismatch hole_random; do
    step "$(basename $d) MMXU-test $c"; $PYT run_eval.py --condition $c "$@" --output $d/oracle_$c.csv
  done
}

step "Region boxes of the clean set (hole, hole_random)"
$PYT "$REPO/clean_eval/build_clean_bbox_table.py"

cd "$T"
step "A: training"
$PYT train_lora.py --out runs/tac_pre_clean
eval_tac "$T/runs/tac_pre_clean" --adapter runs/tac_pre_clean/adapter
step "A clean current_other (prior real, current from another patient)"
$PYT run_eval_clean.py --condition current_other --adapter runs/tac_pre_clean/adapter --output runs/tac_pre_clean/clean_current_other.csv

step "B: pilot (4,000 rows) and pre-registered gate"
$PYT train_lora.py --n-train 4000 --unfreeze-tac --tac-init scratch --tac-lr 2e-5 --out runs/tac_scratch_clean_pilot
$PYT eval_valfast.py --adapter runs/tac_scratch_clean_pilot/adapter \
  --tac-state runs/tac_scratch_clean_pilot/tac_state_dict.bin --out runs/tac_scratch_clean_pilot/valfast.csv
python3 pilot_gate.py --dir runs/tac_scratch_clean_pilot --valfast runs/tac_scratch_clean_pilot/valfast.csv
step "B: training"
$PYT train_lora.py --unfreeze-tac --tac-init scratch --tac-lr 2e-5 --out runs/tac_scratch_clean
eval_tac "$T/runs/tac_scratch_clean" --adapter runs/tac_scratch_clean/adapter --tac-state runs/tac_scratch_clean/tac_state_dict.bin

step "B2 stage 1: pilot (4,000 rows, TAC alone, language model frozen) and its gate"
$PYT train_lora.py --n-train 4000 --stage1-tac-only --unfreeze-tac --tac-init scratch --tac-lr 2e-5 --out runs/tac_b2_stage1_pilot
python3 pilot_gate_stage1.py --dir runs/tac_b2_stage1_pilot
step "B2 stage 1: full (TAC alone)"
$PYT train_lora.py --stage1-tac-only --unfreeze-tac --tac-init scratch --tac-lr 2e-5 --out runs/tac_b2_stage1
step "B2 stage 2: LoRA, stage-1 TAC frozen"
$PYT train_lora.py --tac-state-init runs/tac_b2_stage1/tac_state_dict.bin --out runs/tac_b2_stage2
eval_tac "$T/runs/tac_b2_stage2" --adapter runs/tac_b2_stage2/adapter --tac-state runs/tac_b2_stage1/tac_state_dict.bin

cd "$Q"
step "C: training"
$PYQ train_lora_qwen25vl.py --out runs/qwen_clean
A="--adapter $Q/runs/qwen_clean/adapter"
mkdir -p runs/qwen_clean/px518
cd "$T"
for c in full blind both_other mismatch swap hole hole_random; do   # paper: C at its training resolution
  step "C 518^2 clean $c"; $PYQ run_eval_clean.py --model qwen25vl_baseline --condition $c $A --pixels $PX518 --output $Q/runs/qwen_clean/px518/clean_$c.csv
done
$PYQ run_eval_clean.py --model qwen25vl_baseline --condition full --box-subset --limit 20 $A --pixels $PX518 --output runs/hole_smoke/qwen_clean_full.csv
for c in hole hole_random; do
  $PYQ run_eval_clean.py --model qwen25vl_baseline --condition $c --limit 20 $A --pixels $PX518 --output runs/hole_smoke/qwen_clean_$c.csv
done
for c in full blind mismatch both_other swap duplicate; do   # protocol resolution (supplementary)
  step "C protocol clean $c"; $PYQ run_eval_clean.py --model qwen25vl_baseline --condition $c $A --output $Q/runs/qwen_clean/clean_$c.csv
done
cd "$Q"
for c in full blind; do
  step "C 518^2 MMXU-test $c"; $PYQ run_eval_qwen25vl.py --condition $c $A --pixels $PX518 --output runs/qwen_clean/px518/oracle_$c.csv
done
for c in full blind mismatch hole_random; do
  step "C protocol MMXU-test $c"; $PYQ run_eval_qwen25vl.py --condition $c $A --output runs/qwen_clean/oracle_$c.csv
done

step "Text-only Qwen2.5-VL-3B (C's model): training (no image, ever)"
$PYQ train_lora_textonly.py --out runs/textonly_clean
cd "$T"
$PYQ run_eval_clean.py --model qwen25vl_baseline --condition blind --adapter $Q/runs/textonly_clean/adapter --output $Q/runs/textonly_clean/clean_blind.csv
cd "$Q"
$PYQ run_eval_qwen25vl.py --condition blind --adapter runs/textonly_clean/adapter --output runs/textonly_clean/oracle_blind.csv
cd "$T"
step "Text-only Llama-3.2-3B (A/B/B2's language model): training (no image, ever)"
$PYT train_lora_textonly_llama.py --out runs/textonly_llama
$PYT run_eval_clean.py --condition blind --adapter runs/textonly_llama/adapter --output runs/textonly_llama/clean_blind.csv
$PYT run_eval.py --condition blind --adapter runs/textonly_llama/adapter --output runs/textonly_llama/oracle_blind.csv
step "DONE"
