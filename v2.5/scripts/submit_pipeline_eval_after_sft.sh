#!/bin/sh
# Submit one-MIG end-to-end evaluation only after both clean-v2 SFT runs pass.
set -eu

if [ "$#" -ne 4 ]; then
  echo "usage: sh $0 PLANNER_LOG CAPTIONER_LOG STATE_DIR INTERVAL_SECONDS" >&2
  exit 2
fi

planner_log=$1
captioner_log=$2
state_dir=$3
interval=$4
submission_file="$state_dir/submission.txt"
preflight_file="$state_dir/adapter_preflight.jsonl"
failure_file="$state_dir/failure.txt"
mkdir -p "$state_dir"

if [ -s "$submission_file" ]; then
  echo "[gate] evaluation already submitted: $submission_file"
  exit 0
fi

while :; do
  timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')
  if grep -Eqi 'Traceback|Non-finite|out of memory|CUDA error' "$planner_log" "$captioner_log"; then
    printf '[%s] training failure marker found; evaluation not submitted\n' "$timestamp" > "$failure_file"
    exit 1
  fi

  if grep -q '\[checkpoint\] saved final LoRA adapter' "$planner_log" && \
     grep -q '\[checkpoint\] saved final LoRA adapter' "$captioner_log"; then
    .venv-genkai/bin/python scripts/verify_lora_adapter.py \
      outputs/newyorker_compact_v2_planner_7b_qlora/best_val_loss \
      outputs/newyorker_compact_v2_planner_7b_qlora/final_lora \
      outputs/newyorker_compact_v2_captioner_3b_qlora/best_val_loss \
      outputs/newyorker_compact_v2_captioner_3b_qlora/final_lora > "$preflight_file"
    submission=$(pjsub jobs/genkai_eval_newyorker_compact_v2_pipeline.pjm)
    printf '[%s] %s\n' "$timestamp" "$submission" > "$submission_file"
    echo "[gate] $submission"
    exit 0
  fi

  sleep "$interval"
done
