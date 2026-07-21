#!/usr/bin/env bash
set -euo pipefail

cd /home/zhang.jianqiao/projects/v2.5

PY=${PY:-/home/zhang.jianqiao/miniconda3/envs/humor/bin/python}
CONFIG=${CONFIG:-configs/vlm_guided_generation.yaml}
LIMIT=${LIMIT:-800}
PER_VIEWPOINT=${PER_VIEWPOINT:-100}
SEED=${SEED:-20260710}
WAIT_GPU_FREE_MB=${WAIT_GPU_FREE_MB:-14000}
WAIT_GPU_INDEX=${WAIT_GPU_INDEX:-0}
OVERWRITE=${OVERWRITE:-0}

SOURCE_JSONL=outputs/analysis/hic_humor_viewpoints_pairs_10000_random_minview.jsonl
SUBSET=outputs/annotations/hic_region_annotation_subset_${LIMIT}.jsonl
SUBSET_SUMMARY_JSON=outputs/annotations/hic_region_annotation_subset_${LIMIT}_summary.json
SUBSET_SUMMARY_MD=outputs/annotations/hic_region_annotation_subset_${LIMIT}_summary.md
ANNOTATIONS=outputs/annotations/hic_region_annotations_${LIMIT}.jsonl
RENDERED=outputs/annotations/hic_region_annotations_${LIMIT}_rendered.jsonl
OVERLAY_DIR=outputs/annotations/hic_region_overlays_${LIMIT}
CROP_DIR=outputs/annotations/hic_region_crops_${LIMIT}
REVIEW_HTML=outputs/reviews/hic_region_annotations_${LIMIT}.html
EVAL_PREFIX=outputs/evaluations/hic_region_annotation_ablation_${LIMIT}

METHODS=(
  hic-compact-json
  hic-compact-json-region
  hic-compact-json-overlay
  hic-compact-json-crop
)

mkdir -p outputs/annotations outputs/generations outputs/reviews outputs/evaluations

overwrite_arg=()
if [ "$OVERWRITE" = "1" ]; then
  overwrite_arg=(--overwrite)
fi

echo "[hic-region-ablation] $(date -Is) limit=${LIMIT} per_viewpoint=${PER_VIEWPOINT} overwrite=${OVERWRITE}"

if [ "$OVERWRITE" = "1" ] || [ ! -s "$SUBSET" ]; then
  echo "[hic-region-ablation] $(date -Is) preparing subset"
  "$PY" scripts/prepare_hic_viewpoint_annotation_subset.py \
    --input-jsonl "$SOURCE_JSONL" \
    --per-viewpoint "$PER_VIEWPOINT" \
    --output-jsonl "$SUBSET" \
    --summary-json "$SUBSET_SUMMARY_JSON" \
    --summary-md "$SUBSET_SUMMARY_MD" \
    --seed "$SEED"
else
  echo "[hic-region-ablation] $(date -Is) subset exists: $SUBSET"
fi

if [ "$OVERWRITE" = "1" ] || [ ! -s "$ANNOTATIONS" ]; then
  echo "[hic-region-ablation] $(date -Is) annotating regions"
  env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/annotate_hic_humor_regions.py \
      --input-jsonl "$SUBSET" \
      --output-jsonl "$ANNOTATIONS" \
      --limit "$LIMIT" \
      --wait-gpu-free-mb "$WAIT_GPU_FREE_MB" \
      --wait-gpu-index "$WAIT_GPU_INDEX" \
      --wait-gpu-stable-checks 2 \
      --wait-gpu-check-seconds 30 \
      "${overwrite_arg[@]}"
else
  echo "[hic-region-ablation] $(date -Is) annotations exist: $ANNOTATIONS"
fi

if [ "$OVERWRITE" = "1" ] || [ ! -s "$RENDERED" ]; then
  echo "[hic-region-ablation] $(date -Is) rendering overlays and crop sheets"
  "$PY" scripts/render_hic_region_overlays.py \
    --input-jsonl "$ANNOTATIONS" \
    --output-jsonl "$RENDERED" \
    --overlay-dir "$OVERLAY_DIR" \
    --crop-dir "$CROP_DIR" \
    --review-html "$REVIEW_HTML" \
    "${overwrite_arg[@]}"
else
  echo "[hic-region-ablation] $(date -Is) rendered JSONL exists: $RENDERED"
fi

generation_jsonls=()
for method in "${METHODS[@]}"; do
  name=${method//-/_}
  output_jsonl=outputs/generations/hic_region_ablation_${name}_${LIMIT}.jsonl
  output_html=outputs/reviews/hic_region_ablation_${name}_${LIMIT}.html
  output_log=outputs/generations/hic_region_ablation_${name}_${LIMIT}.log
  generation_jsonls+=("$output_jsonl")
  if [ "$OVERWRITE" != "1" ] && [ -s "$output_jsonl" ]; then
    echo "[hic-region-ablation] $(date -Is) generation exists method=${method}: $output_jsonl"
    continue
  fi
  echo "[hic-region-ablation] $(date -Is) generating method=${method}"
  env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/generate_guided_lora_candidates.py \
      --config "$CONFIG" \
      --method "$method" \
      --input-jsonl "$RENDERED" \
      --context-jsonl "$RENDERED" \
      --output-jsonl "$output_jsonl" \
      --review-html "$output_html" \
      --limit "$LIMIT" \
      --num-candidates 8 \
      --seed "$SEED" \
      --wait-gpu-free-mb "$WAIT_GPU_FREE_MB" \
      --wait-gpu-index "$WAIT_GPU_INDEX" \
      --wait-gpu-stable-checks 2 \
      --wait-gpu-check-seconds 30 \
      --min-pixels 200704 \
      --max-pixels 802816 \
      "${overwrite_arg[@]}" \
      > "$output_log" 2>&1
done

judge_args=()
for path in "${generation_jsonls[@]}"; do
  judge_args+=(--generation-jsonl "$path")
done

judge_resume_arg=(--resume)
if [ "$OVERWRITE" = "1" ]; then
  judge_resume_arg=(--overwrite)
fi

echo "[hic-region-ablation] $(date -Is) judging region ablation methods"
env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" scripts/judge_hic_guidance_methods_gold.py \
    "${judge_args[@]}" \
    --output-jsonl "${EVAL_PREFIX}_judgments.jsonl" \
    --summary-json "${EVAL_PREFIX}_summary.json" \
    --report-md "${EVAL_PREFIX}_report.md" \
    --limit "$LIMIT" \
    --wait-gpu-free-mb "$WAIT_GPU_FREE_MB" \
    --wait-gpu-index "$WAIT_GPU_INDEX" \
    --wait-gpu-stable-checks 2 \
    --wait-gpu-check-seconds 30 \
    --min-pixels 200704 \
    --max-pixels 802816 \
    "${judge_resume_arg[@]}" \
    > "${EVAL_PREFIX}_judge.log" 2>&1

echo "[hic-region-ablation] $(date -Is) done"
