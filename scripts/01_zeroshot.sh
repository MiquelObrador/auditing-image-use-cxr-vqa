#!/usr/bin/env bash
# Zero-shot VLMs on MMXU-test (2,992 rows): Table 1a. ~20 GPU hours on one 24 GB GPU.
# Every model runs every condition. mismatch gives each row the prior of another patient
# (patient derangement, seed 0); swap permutes the answer letters per question, and its
# baseline is full with the same permutation.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_MMXU="${ENV_MMXU:-mmxu_eval}"          # Qwen2-VL, HuatuoGPT, Qwen2.5-VL
ENV_INTERNVL="${ENV_INTERNVL:-internvl_eval}"   # InternVL2 (transformers 4.44.2)
run() { conda run --no-capture-output -n "$1" python "${@:2}"; }

declare -A OUT=([qwen2vl]=mmxu_eval [internvl2]=internvl_eval [huatuo]=huatuo_eval [qwen25vl]=qwen25vl_eval)
declare -A ENV=([qwen2vl]=$ENV_MMXU [internvl2]=$ENV_INTERNVL [huatuo]=$ENV_MMXU [qwen25vl]=$ENV_MMXU)

cd "$REPO/mmxu_eval"
for m in qwen2vl internvl2 huatuo qwen25vl; do
  d="$REPO/${OUT[$m]}/oracle_results"; mkdir -p "$d"
  for c in full blind hole hole_random; do
    echo "=== $m $c ==="; run "${ENV[$m]}" oracle_evaluate.py --model "$m" --condition "$c" --output "$d/oracle_$c.csv"
  done
  echo "=== $m both_other ==="
  run "${ENV[$m]}" "$REPO/clean_eval/run_both_other_zeroshot.py" --model "$m" --output "$d/oracle_both_other.csv"
done
for m in qwen2vl internvl2 huatuo qwen25vl; do
  d="$REPO/${OUT[$m]}/oracle_results"
  echo "=== $m mismatch: another patient's prior (patient derangement, seed 0) ==="
  run "${ENV[$m]}" "$REPO/clean_eval/run_mismatch_zeroshot.py" --model "$m" --output "$d/oracle_mismatch_patient.csv"
  echo "=== $m swap, letters permuted per question, and full with the same permutation ==="
  run "${ENV[$m]}" permuted_check.py --model "$m" --condition full --output "$d/oracle_full_permuted.csv"
  run "${ENV[$m]}" swap_permuted_check.py --model "$m" --output "$d/oracle_swap_permuted.csv"
done
