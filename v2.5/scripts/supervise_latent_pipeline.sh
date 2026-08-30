#!/bin/sh
# Idempotently keep the two latent-pipeline monitors alive until their gates finish.
set -eu

root=${1:-"$(pwd)"}
full_job_id=${2:-6632115}
interval=${SUPERVISOR_INTERVAL_SECONDS:-120}
cd "$root"
status_dir=results/latent_communication/pipeline_status
mkdir -p "$status_dir"
log="$status_dir/supervisor.log"

while :; do
  now=$(date -Iseconds)
  if test ! -s "$status_dir/eval_submission.txt"; then
    if ! tmux has-session -t latent-bridge-monitor 2>/dev/null; then
      tmux new-session -d -s latent-bridge-monitor \
        "cd '$root' && sh scripts/monitor_latent_bridge_pipeline.sh '$full_job_id' '$root'"
      printf '%s restarted latent-bridge-monitor\n' "$now" >> "$log"
    fi
  fi

  if test ! -s "$status_dir/next_gate.txt"; then
    if ! tmux has-session -t latent-5h-review 2>/dev/null; then
      tmux new-session -d -s latent-5h-review \
        "cd '$root' && sh scripts/five_hour_latent_pipeline_review.sh '$root' '$full_job_id'"
      printf '%s restarted latent-5h-review\n' "$now" >> "$log"
    fi
  fi

  printf '%s heartbeat bridge=%s review=%s\n' "$now" \
    "$(tmux has-session -t latent-bridge-monitor 2>/dev/null && echo up || echo done)" \
    "$(tmux has-session -t latent-5h-review 2>/dev/null && echo up || echo done)" >> "$log"

  if test -s "$status_dir/next_gate.txt"; then
    printf '%s supervisor_complete next_gate=%s\n' "$now" "$(cat "$status_dir/next_gate.txt")" >> "$log"
    exit 0
  fi
  sleep "$interval"
done
