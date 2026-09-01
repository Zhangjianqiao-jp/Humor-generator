#!/bin/sh
# Submit validation Group-of-3 only after a successful full Bridge run.
set -eu

job_id=${1:?usage: monitor_latent_bridge_pipeline.sh FULL_JOB_ID}
root=${2:-"$(pwd)"}
cd "$root"
status_dir=results/latent_communication/pipeline_status
mkdir -p "$status_dir"
log="$status_dir/monitor_${job_id}.log"
submission_file="$status_dir/eval_submission.txt"
absent_checks=0

while :; do
  if ! snapshot=$(pjstat 2>/dev/null); then
    printf '%s pjstat unavailable; retaining monitor state\n' "$(date -Iseconds)" >> "$log"
    sleep 60
    continue
  fi
  if printf '%s\n' "$snapshot" | awk -v id="$job_id" '$1 == id { found=1 } END { exit !found }'; then
    absent_checks=0
    printf '%s full job %s active\n' "$(date -Iseconds)" "$job_id" >> "$log"
    sleep 60
    continue
  fi
  absent_checks=$((absent_checks + 1))
  printf '%s full job %s absent check=%s/3\n' "$(date -Iseconds)" "$job_id" "$absent_checks" >> "$log"
  test "$absent_checks" -ge 3 && break
  sleep 60
done

manifest=outputs/latent_communication/bridge_sft/run_manifest.json
checkpoint=outputs/latent_communication/bridge_sft/best.pt
if [ ! -s "$manifest" ] || [ ! -s "$checkpoint" ]; then
  printf '%s full job ended without required artifacts; evaluation not submitted\n' "$(date -Iseconds)" >> "$log"
  exit 1
fi

if ! python - "$manifest" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
assert doc.get("smoke") is False
assert doc.get("bridge_only") is True
assert doc.get("policies_frozen") is True
assert doc.get("history")
PY
then
  printf '%s manifest gate failed; evaluation not submitted\n' "$(date -Iseconds)" >> "$log"
  exit 1
fi

if [ -s "$submission_file" ]; then
  printf '%s evaluation submission already recorded; refusing duplicate\n' "$(date -Iseconds)" >> "$log"
  exit 0
fi

submission=$(pjsub jobs/genkai_eval_7b_latent_group3.pjm)
printf '%s %s\n' "$(date -Iseconds)" "$submission" | tee -a "$log"
printf '%s\n' "$submission" > "$submission_file"
