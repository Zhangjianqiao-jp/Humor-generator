#!/usr/bin/env bash
set -euo pipefail

cd /home/zhang.jianqiao/projects/v2.5
mkdir -p outputs/generations outputs/reviews outputs/evaluations

PY=/home/zhang.jianqiao/miniconda3/envs/humor/bin/python
CONFIG=configs/vlm_guided_generation.yaml
SUBSET=outputs/analysis/hic_viewpoint_ablation_120.jsonl
SEED=250704
METHODS=(plain hic-humor-point hic-viewpoint-tags hic-anchor-viewpoint hic-compact-json)

for method in "${METHODS[@]}"; do
  name=${method//-/_}
  echo "[hic-ablation] $(date -Is) generating method=${method}"
  set +e
  env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/generate_guided_lora_candidates.py \
      --config "$CONFIG" \
      --method "$method" \
      --input-jsonl "$SUBSET" \
      --context-jsonl "$SUBSET" \
      --output-jsonl "outputs/generations/hic_${name}_120.jsonl" \
      --review-html "outputs/reviews/hic_${name}_120.html" \
      --num-candidates 8 \
      --seed "$SEED" \
      --wait-gpu-free-mb 18000 \
      --wait-gpu-stable-checks 3 \
      --wait-gpu-check-seconds 60 \
      --overwrite \
      > "outputs/generations/hic_${name}_120.log" 2>&1
  code=$?
  set -e
  echo "[hic-ablation] $(date -Is) done method=${method} code=${code}"
  if [ "$code" -ne 0 ]; then
    tail -n 80 "outputs/generations/hic_${name}_120.log" || true
    exit "$code"
  fi
done

echo "[hic-ablation] $(date -Is) judging methods"
set +e
env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" scripts/judge_hic_guidance_methods_gold.py \
    --generation-jsonl outputs/generations/hic_plain_120.jsonl \
    --generation-jsonl outputs/generations/hic_hic_humor_point_120.jsonl \
    --generation-jsonl outputs/generations/hic_hic_viewpoint_tags_120.jsonl \
    --generation-jsonl outputs/generations/hic_hic_anchor_viewpoint_120.jsonl \
    --generation-jsonl outputs/generations/hic_hic_compact_json_120.jsonl \
    --output-jsonl outputs/evaluations/hic_guidance_gold_judgments_120.jsonl \
    --summary-json outputs/evaluations/hic_guidance_gold_summary_120.json \
    --report-md outputs/evaluations/hic_guidance_gold_report_120.md \
    --wait-gpu-free-mb 22000 \
    --wait-gpu-stable-checks 3 \
    --wait-gpu-check-seconds 60 \
    --overwrite \
    > outputs/evaluations/hic_guidance_gold_judge_120.log 2>&1
code=$?
set -e
echo "[hic-ablation] $(date -Is) judge done code=${code}"
if [ "$code" -ne 0 ]; then
  tail -n 80 outputs/evaluations/hic_guidance_gold_judge_120.log || true
  exit "$code"
fi

echo "[hic-ablation] $(date -Is) all done"
