#!/usr/bin/env bash
set -euo pipefail

cd /home/zhang.jianqiao/projects/v2.5
mkdir -p outputs/analysis

PY=/home/zhang.jianqiao/miniconda3/envs/humor/bin/python
HIC_ROOT=/home/zhang.jianqiao/datasets/hic-data
LIMIT=${LIMIT:-10000}
SAMPLE_SEED=${SAMPLE_SEED:-250708}
WAIT_GPU_FREE_MB=${WAIT_GPU_FREE_MB:-22000}

OUT_JSONL="outputs/analysis/hic_humor_viewpoints_pairs_${LIMIT}_random_minview.jsonl"
OUT_SUMMARY="outputs/analysis/hic_humor_viewpoint_summary_pairs_${LIMIT}_random_minview.json"
OUT_REPORT="outputs/analysis/hic_humor_viewpoint_report_pairs_${LIMIT}_random_minview.md"
TAX_SUMMARY="outputs/analysis/hic_viewpoint_taxonomy_pairs_${LIMIT}_random_minview_summary.json"
TAX_REPORT="outputs/analysis/hic_viewpoint_taxonomy_pairs_${LIMIT}_random_minview.md"

echo "[taxonomy-10k] $(date -Is) starting viewpoint classification limit=${LIMIT} sample_seed=${SAMPLE_SEED}"
env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" scripts/analyze_hic_humor_viewpoints.py \
    --input "$HIC_ROOT" \
    --output-jsonl "$OUT_JSONL" \
    --summary-json "$OUT_SUMMARY" \
    --report-md "$OUT_REPORT" \
    --limit "$LIMIT" \
    --sample-seed "$SAMPLE_SEED" \
    --temperature 0 \
    --max-new-tokens 768 \
    --wait-gpu-free-mb "$WAIT_GPU_FREE_MB" \
    --wait-gpu-stable-checks 3 \
    --wait-gpu-check-seconds 60

echo "[taxonomy-10k] $(date -Is) writing taxonomy report"
"$PY" scripts/report_hic_viewpoint_taxonomy.py \
  --input-jsonl "$OUT_JSONL" \
  --summary-json "$TAX_SUMMARY" \
  --report-md "$TAX_REPORT"

echo "[taxonomy-10k] $(date -Is) all done"
echo "[taxonomy-10k] taxonomy report: $TAX_REPORT"
