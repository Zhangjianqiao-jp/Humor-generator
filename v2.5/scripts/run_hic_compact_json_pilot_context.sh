#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home/zhang.jianqiao/miniconda3/envs/humor/bin/python}"
TRAIN_LIMIT="${TRAIN_LIMIT:-32}"
VAL_LIMIT="${VAL_LIMIT:-8}"
TRAIN_SEED="${TRAIN_SEED:-260704}"
VAL_SEED="${VAL_SEED:-260705}"
WAIT_GPU_FREE_MB="${WAIT_GPU_FREE_MB:-18000}"
WAIT_GPU_INDEX="${WAIT_GPU_INDEX:-0}"
WAIT_GPU_STABLE_CHECKS="${WAIT_GPU_STABLE_CHECKS:-2}"
WAIT_GPU_CHECK_SECONDS="${WAIT_GPU_CHECK_SECONDS:-60}"
MIN_PIXELS="${MIN_PIXELS:-100352}"
MAX_PIXELS="${MAX_PIXELS:-401408}"

COMMON_ARGS=(
  --model-name /home/zhang.jianqiao/models/Qwen2.5-VL-7B-Instruct
  --temperature 0.0
  --max-new-tokens 768
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_PIXELS}"
  --wait-gpu-free-mb "${WAIT_GPU_FREE_MB}"
  --wait-gpu-index "${WAIT_GPU_INDEX}"
  --wait-gpu-stable-checks "${WAIT_GPU_STABLE_CHECKS}"
  --wait-gpu-check-seconds "${WAIT_GPU_CHECK_SECONDS}"
  --no-dedupe-image
  --overwrite
)

"${PY}" scripts/analyze_hic_humor_viewpoints.py \
  --input data/processed/sft_train.jsonl \
  --output-jsonl outputs/analysis/hic_humor_viewpoints_sft_train_pilot.jsonl \
  --summary-json outputs/analysis/hic_humor_viewpoint_summary_sft_train_pilot.json \
  --report-md outputs/analysis/hic_humor_viewpoint_report_sft_train_pilot.md \
  --limit "${TRAIN_LIMIT}" \
  --sample-seed "${TRAIN_SEED}" \
  "${COMMON_ARGS[@]}"

"${PY}" scripts/analyze_hic_humor_viewpoints.py \
  --input data/processed/sft_val.jsonl \
  --output-jsonl outputs/analysis/hic_humor_viewpoints_sft_val_pilot.jsonl \
  --summary-json outputs/analysis/hic_humor_viewpoint_summary_sft_val_pilot.json \
  --report-md outputs/analysis/hic_humor_viewpoint_report_sft_val_pilot.md \
  --limit "${VAL_LIMIT}" \
  --sample-seed "${VAL_SEED}" \
  "${COMMON_ARGS[@]}"

"${PY}" scripts/train_lora_sft_with_features.py \
  --config configs/lora_sft_hic_compact_json_pilot.yaml \
  --debug-data \
  --max-train-samples 3 \
  --max-val-samples 2
