#!/bin/sh
set -eu

reference_job="${1:?usage: monitor_quality64_pipeline.sh REFERENCE_JOB_ID [ROOT]}"
root="${2:-.}"
cd "$root"

log=results/7b_generator/dpo_quality64_step_matched/pipeline_monitor.log
partial=data/processed/newyorker_published_dpo_reference_7b_generator_quality64/dpo_train.partial.jsonl
complete=data/processed/newyorker_published_dpo_reference_7b_generator_quality64/dpo_train.jsonl
reference_out="genkai_7b_quality64_reference_only.pjm.${reference_job}.out"
gate=results/7b_generator/dpo_quality64_step_matched/validation_gate.json
mkdir -p "$(dirname "$log")"
dpo_job="${DPO_JOB_ID:-}"
full_job_file=results/7b_generator/dpo_quality64_step_matched/full_job_id.txt
interval_seconds="${MONITOR_INTERVAL_SECONDS:-300}"
full_job=""
test -f "$full_job_file" && full_job="$(tr -dc '0-9' < "$full_job_file")"

while :; do
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  reference_state="$(pjstat "$reference_job" 2>/dev/null | awk -v id="$reference_job" '$1 == id {print $4}' || true)"
  partial_rows=0
  complete_rows=0
  test -f "$partial" && partial_rows="$(wc -l < "$partial")"
  test -f "$complete" && complete_rows="$(wc -l < "$complete")"
  printf '%s reference_job=%s state=%s partial=%s complete=%s\n' \
    "$now" "$reference_job" "${reference_state:-finished}" "$partial_rows" "$complete_rows" >> "$log"

  if test -z "$dpo_job" && test -f "$reference_out"; then
    dpo_job="$(sed -n 's/.*pjsub Job \([0-9][0-9]*\) submitted.*/\1/p' "$reference_out" | tail -1)"
  fi
  if test -n "$dpo_job"; then
    dpo_state="$(pjstat "$dpo_job" 2>/dev/null | awk -v id="$dpo_job" '$1 == id {print $4}' || true)"
    printf '%s dpo_job=%s state=%s gate=%s\n' \
      "$now" "$dpo_job" "${dpo_state:-finished}" "$(test -f "$gate" && echo present || echo absent)" >> "$log"
  fi

  if test -f "$gate"; then
    decision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$gate")"
    if test "$decision" = "NO_GO_FULL_QUALITY64"; then
      printf '%s pipeline_complete decision=%s full_dpo=not_submitted\n' "$now" "$decision" >> "$log"
      exit 0
    fi
    if test -z "$full_job" && test -n "$dpo_job" && test -z "${dpo_state:-}"; then
      submission="$(pjsub jobs/genkai_7b_quality64_full_dpo.pjm)"
      full_job="$(printf '%s\n' "$submission" | sed -n 's/.*pjsub Job \([0-9][0-9]*\) submitted.*/\1/p')"
      if test -z "$full_job"; then
        printf '%s full_submit_failed output=%s\n' "$now" "$submission" >> "$log"
        exit 1
      fi
      printf '%s\n' "$full_job" > "$full_job_file"
      printf '%s full_submitted job=%s\n' "$now" "$full_job" >> "$log"
    fi
    if test -n "$full_job"; then
      full_state="$(pjstat "$full_job" 2>/dev/null | awk -v id="$full_job" '$1 == id {print $4}' || true)"
      full_checkpoint=outputs/7b-generator-dpo/dpo_mlp_quality64_full/final/adapter_model.safetensors
      printf '%s full_job=%s state=%s checkpoint=%s\n' \
        "$now" "$full_job" "${full_state:-finished}" "$(test -f "$full_checkpoint" && echo present || echo absent)" >> "$log"
      if test -z "$full_state" && test -f "$full_checkpoint"; then
        printf '%s pipeline_complete decision=%s full_dpo=complete\n' "$now" "$decision" >> "$log"
        exit 0
      fi
    fi
  fi
  if test -z "$reference_state" && test "$complete_rows" -ne 17297 && test -z "$dpo_job"; then
    printf '%s pipeline_stopped reference_incomplete\n' "$now" >> "$log"
    exit 1
  fi
  sleep "$interval_seconds"
done
