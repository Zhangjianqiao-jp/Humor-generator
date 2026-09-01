#!/bin/sh
# Record SFT job state every 20 minutes without reserving GPU resources.
set -eu

if [ "$#" -ne 4 ]; then
  echo "usage: $0 PLANNER_JOB CAPTIONER_JOB OUTPUT_DIR INTERVAL_SECONDS" >&2
  exit 2
fi

planner_job=$1
captioner_job=$2
output_dir=$3
interval=$4
mkdir -p "$output_dir"
log_file="$output_dir/monitor.log"

while :; do
  timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')
  {
    echo "[$timestamp] planner=$planner_job captioner=$captioner_job"
    pjstat -v "$planner_job" "$captioner_job" 2>&1 || true
    planner_log=$(find . -maxdepth 1 -type f -name "*.$planner_job.out" -print -quit)
    captioner_log=$(find . -maxdepth 1 -type f -name "*.$captioner_job.out" -print -quit)
    for file in "$planner_log" "$captioner_log"; do
      if [ -z "$file" ]; then
        echo "--- job log not created"
        continue
      fi
      if [ -f "$file" ]; then
        echo "--- $file"
        tail -n 25 "$file"
      else
        echo "--- $file (not created)"
      fi
    done
  } >> "$log_file"

  planner_log=$(find . -maxdepth 1 -type f -name "*.$planner_job.out" -print -quit)
  captioner_log=$(find . -maxdepth 1 -type f -name "*.$captioner_job.out" -print -quit)
  if { [ -n "$planner_log" ] && grep -Eqi 'Traceback|Non-finite|out of memory' "$planner_log"; } || \
     { [ -n "$captioner_log" ] && grep -Eqi 'Traceback|Non-finite|out of memory' "$captioner_log"; }; then
    echo "[$timestamp] stopping monitor: failure marker found" >> "$log_file"
    exit 1
  fi
  if { [ -n "$planner_log" ] && grep -q '\[checkpoint\] saved final LoRA adapter' "$planner_log"; } && \
     { [ -n "$captioner_log" ] && grep -q '\[checkpoint\] saved final LoRA adapter' "$captioner_log"; }; then
    echo "[$timestamp] stopping monitor: both final adapters saved" >> "$log_file"
    exit 0
  fi
  sleep "$interval"
done
