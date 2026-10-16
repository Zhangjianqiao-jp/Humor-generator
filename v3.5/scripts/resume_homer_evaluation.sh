#!/usr/bin/env bash
# Context-free start/status/once entrypoint for the HOMER account-agent track.
# It is intentionally idempotent: the Python controller owns the evaluation
# lock and resumes only missing, validated packet batches.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
DEFAULT_ROOT="${PROJECT_DIR}/outputs/caption_judgement/pretrained_bridge_comparison_20260911"
ROOT="${HOMER_EVAL_ROOT:-${DEFAULT_ROOT}}"
MODE="${1:-start}"

case "${MODE}" in
  start|status|once)
    shift || true
    ;;
  *)
    echo "usage: $0 [start|status|once] [controller options]" >&2
    exit 64
    ;;
esac

exec "${PROJECT_DIR}/.venv/bin/python" \
  "${PROJECT_DIR}/scripts/homer_evaluation_controller.py" \
  --root "${ROOT}" --mode "${MODE}" "$@"
