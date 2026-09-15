#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 SMOKE_JOB_ID" >&2
  exit 2
fi

smoke_job_id=$1
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
state_dir="$root/.auto-research/phase-a3-monitor"
log="$state_dir/monitor.log"
formal_id_file="$state_dir/formal_job_id.txt"
mkdir -p "$state_dir"

last_state=
absent_observations=0
while :; do
  listing_file="$state_dir/pjstat.latest"
  if ! pjstat "$smoke_job_id" > "$listing_file" 2>/dev/null; then
    sleep 60
    continue
  fi
  current=$(awk 'NR==2 {print $4}' "$listing_file")
  if [ -n "$current" ]; then
    absent_observations=0
    if [ "$current" != "$last_state" ]; then
      date -Is | awk -v state="$current" '{print $0, "smoke_state=" state}' >> "$log"
      last_state=$current
    fi
    sleep 300
    continue
  fi
  absent_observations=$((absent_observations + 1))
  if [ "$absent_observations" -lt 3 ]; then
    sleep 60
    continue
  fi
  break
done

cd "$root"
. .venv/bin/activate
python scripts/validate_phase_a3_smoke.py \
  --config configs/pilot/cross_attention_semantic_phase_a3.yaml \
  --report results/engineering_smoke/cross_attention_semantic_phase_a3.json >> "$log" 2>&1

if [ -s "$formal_id_file" ]; then
  date -Is | awk '{print $0, "formal_already_submitted"}' >> "$log"
  exit 0
fi
if [ -e outputs/pilot/cross_attention_semantic_phase_a3/complete.json ]; then
  date -Is | awk '{print $0, "formal_already_complete"}' >> "$log"
  exit 0
fi
if [ -d outputs/pilot/cross_attention_semantic_phase_a3 ]; then
  date -Is | awk '{print $0, "refuse_partial_output_directory"}' >> "$log"
  exit 3
fi

submission=$(pjsub -N v35_phase_a3 jobs/cross_attention_phase_a3.pjm)
formal_job_id=$(printf '%s\n' "$submission" | awk '/submitted/ {print $(NF-1)}')
case "$formal_job_id" in
  ''|*[!0-9]*)
    printf '%s\n' "$submission" >> "$log"
    exit 4
    ;;
esac
printf '%s\n' "$formal_job_id" > "$formal_id_file"
date -Is | awk -v job="$formal_job_id" '{print $0, "formal_submitted=" job}' >> "$log"
