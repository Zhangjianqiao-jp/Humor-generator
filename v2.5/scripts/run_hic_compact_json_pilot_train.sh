#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home/zhang.jianqiao/miniconda3/envs/humor/bin/python}"
CONFIG="${CONFIG:-configs/lora_sft_hic_compact_json_pilot.yaml}"
TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
VAL_LIMIT="${VAL_LIMIT:-128}"
TRAIN_SEED="${TRAIN_SEED:-260704}"
VAL_SEED="${VAL_SEED:-260705}"
WAIT_GPU_FREE_MB="${WAIT_GPU_FREE_MB:-18000}"
WAIT_GPU_INDEX="${WAIT_GPU_INDEX:-0}"
WAIT_GPU_STABLE_CHECKS="${WAIT_GPU_STABLE_CHECKS:-2}"
WAIT_GPU_CHECK_SECONDS="${WAIT_GPU_CHECK_SECONDS:-60}"
MIN_PIXELS="${MIN_PIXELS:-100352}"
MAX_PIXELS="${MAX_PIXELS:-401408}"
RUN_TRAIN="${RUN_TRAIN:-1}"
OVERWRITE_CONTEXT="${OVERWRITE_CONTEXT:-1}"
CONTINUE_ON_OOM="${CONTINUE_ON_OOM:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LOGGING_STEPS="${LOGGING_STEPS:-}"
EVAL_STEPS="${EVAL_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
FIXED_GENERATION_SAMPLES="${FIXED_GENERATION_SAMPLES:-}"

TAG="${TAG:-${TRAIN_LIMIT}}"
TRAIN_CONTEXT="${TRAIN_CONTEXT:-outputs/analysis/hic_humor_viewpoints_sft_train_pilot_${TAG}.jsonl}"
VAL_CONTEXT="${VAL_CONTEXT:-outputs/analysis/hic_humor_viewpoints_sft_val_pilot_${VAL_LIMIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora_sft_hic_compact_json_pilot_${TAG}}"

COMMON_ANALYSIS_ARGS=(
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
)

if [[ "${OVERWRITE_CONTEXT}" == "1" ]]; then
  COMMON_ANALYSIS_ARGS+=(--overwrite)
fi

if [[ "${CONTINUE_ON_OOM}" == "1" ]]; then
  COMMON_ANALYSIS_ARGS+=(--continue-on-oom)
fi

limit_args() {
  local limit_value="$1"
  local seed_value="$2"
  local -n out_args="$3"
  local normalized
  normalized="$(printf '%s' "${limit_value}" | tr '[:upper:]' '[:lower:]')"
  out_args=()
  case "${normalized}" in
    all|full|none|null|0|-1)
      out_args+=(--limit 0)
      ;;
    *)
      out_args+=(--limit "${limit_value}" --sample-seed "${seed_value}")
      ;;
  esac
}

TRAIN_LIMIT_ARGS=()
VAL_LIMIT_ARGS=()
limit_args "${TRAIN_LIMIT}" "${TRAIN_SEED}" TRAIN_LIMIT_ARGS
limit_args "${VAL_LIMIT}" "${VAL_SEED}" VAL_LIMIT_ARGS

echo "[pilot] train_context=${TRAIN_CONTEXT}"
echo "[pilot] val_context=${VAL_CONTEXT}"
echo "[pilot] output_dir=${OUTPUT_DIR}"
echo "[pilot] train_limit=${TRAIN_LIMIT} val_limit=${VAL_LIMIT} overwrite_context=${OVERWRITE_CONTEXT} resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-none}"

"${PY}" scripts/analyze_hic_humor_viewpoints.py \
  --input data/processed/sft_train.jsonl \
  --output-jsonl "${TRAIN_CONTEXT}" \
  --summary-json "outputs/analysis/hic_humor_viewpoint_summary_sft_train_pilot_${TAG}.json" \
  --report-md "outputs/analysis/hic_humor_viewpoint_report_sft_train_pilot_${TAG}.md" \
  "${TRAIN_LIMIT_ARGS[@]}" \
  "${COMMON_ANALYSIS_ARGS[@]}"

"${PY}" scripts/analyze_hic_humor_viewpoints.py \
  --input data/processed/sft_val.jsonl \
  --output-jsonl "${VAL_CONTEXT}" \
  --summary-json "outputs/analysis/hic_humor_viewpoint_summary_sft_val_pilot_${VAL_LIMIT}.json" \
  --report-md "outputs/analysis/hic_humor_viewpoint_report_sft_val_pilot_${VAL_LIMIT}.md" \
  "${VAL_LIMIT_ARGS[@]}" \
  "${COMMON_ANALYSIS_ARGS[@]}"

"${PY}" scripts/train_lora_sft_with_features.py \
  --config "${CONFIG}" \
  --debug-data \
  --max-train-samples 3 \
  --max-val-samples 2 \
  --override-data.train_path "${TRAIN_CONTEXT}" \
  --override-data.val_path "${VAL_CONTEXT}" \
  --override-data.train_context_path "${TRAIN_CONTEXT}" \
  --override-data.val_context_path "${VAL_CONTEXT}" \
  --override-output.output_dir "${OUTPUT_DIR}"

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "[pilot] RUN_TRAIN=${RUN_TRAIN}; stopping after context and debug-data."
  exit 0
fi

TRAIN_ARGS=(
  --config "${CONFIG}"
  --override-data.train_path "${TRAIN_CONTEXT}"
  --override-data.val_path "${VAL_CONTEXT}"
  --override-data.train_context_path "${TRAIN_CONTEXT}"
  --override-data.val_context_path "${VAL_CONTEXT}"
  --override-output.output_dir "${OUTPUT_DIR}"
  --override-output.latest_adapter_dir "${OUTPUT_DIR}/latest"
  --override-output.best_adapter_dir "${OUTPUT_DIR}/best_val_loss"
  --override-output.final_adapter_dir "${OUTPUT_DIR}/final_lora"
  --override-output.tensorboard_dir "${OUTPUT_DIR}/tensorboard"
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${LOGGING_STEPS}" ]]; then
  TRAIN_ARGS+=(--override-training.logging_steps "${LOGGING_STEPS}")
fi
if [[ -n "${EVAL_STEPS}" ]]; then
  TRAIN_ARGS+=(--override-training.eval_steps "${EVAL_STEPS}")
fi
if [[ -n "${SAVE_STEPS}" ]]; then
  TRAIN_ARGS+=(--override-training.save_steps "${SAVE_STEPS}")
fi
if [[ -n "${SAVE_TOTAL_LIMIT}" ]]; then
  TRAIN_ARGS+=(--override-training.save_total_limit "${SAVE_TOTAL_LIMIT}")
fi
if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
  TRAIN_ARGS+=(--override-evaluation.max_eval_samples "${MAX_EVAL_SAMPLES}")
fi
if [[ -n "${FIXED_GENERATION_SAMPLES}" ]]; then
  TRAIN_ARGS+=(--override-evaluation.fixed_generation_samples "${FIXED_GENERATION_SAMPLES}")
fi

env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" scripts/train_lora_sft_with_features.py \
  "${TRAIN_ARGS[@]}"
