#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
: "${CACHE_JOB_ID:?set CACHE_JOB_ID to the already submitted strict-cache job}"
session=formal_bridge_gate
if tmux has-session -t "$session" 2>/dev/null; then
  echo "monitor already active: $session"
  exit 0
fi
tmux new-session -d -s "$session" \
  ".venv/bin/python scripts/formal_pipeline_monitor.py --initial-job-id '$CACHE_JOB_ID' >> data/cache/formal_pipeline_monitor.log 2>&1"
tmux has-session -t "$session"
echo "monitor started: $session"
