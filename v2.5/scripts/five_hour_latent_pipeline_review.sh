#!/bin/sh
# Persistent delayed audit for the Bridge -> Group-of-3 pipeline.
set -eu

root=${1:-"$(pwd)"}
full_job_id=${2:-6632115}
delay_seconds=${REVIEW_DELAY_SECONDS:-18000}
poll_seconds=${REVIEW_POLL_SECONDS:-60}
cd "$root"

status_dir=results/latent_communication/pipeline_status
eval_dir=results/latent_communication/validation_group3
review_dir="$eval_dir/automated_review"
log="$status_dir/five_hour_review.log"
target_file="$status_dir/five_hour_review_target_epoch.txt"
mkdir -p "$status_dir" "$review_dir"

if test -n "${REVIEW_NOT_BEFORE_EPOCH:-}"; then
  target_epoch=$REVIEW_NOT_BEFORE_EPOCH
elif test -s "$target_file"; then
  target_epoch=$(tr -dc '0-9' < "$target_file")
else
  target_epoch=$(($(date +%s) + delay_seconds))
fi
case "$target_epoch" in *[!0-9]*|'') echo "invalid review target epoch" >&2; exit 2;; esac
printf '%s\n' "$target_epoch" > "$target_file"
remaining=$((target_epoch - $(date +%s)))
test "$remaining" -gt 0 || remaining=0
printf '%s armed target_epoch=%s remaining=%s full_job=%s\n' \
  "$(date -Iseconds)" "$target_epoch" "$remaining" "$full_job_id" >> "$log"
sleep "$remaining"
printf '%s delayed_review_started\n' "$(date -Iseconds)" >> "$log"

# Do not analyze partial output if the formal Bridge job is still active.
while :; do
  if ! snapshot=$(pjstat 2>/dev/null); then
    printf '%s pjstat unavailable while checking bridge\n' "$(date -Iseconds)" >> "$log"
    sleep "$poll_seconds"
    continue
  fi
  if ! printf '%s\n' "$snapshot" | awk -v id="$full_job_id" '$1 == id {found=1} END {exit !found}'; then
    break
  fi
  printf '%s bridge_still_active job=%s\n' "$(date -Iseconds)" "$full_job_id" >> "$log"
  sleep "$poll_seconds"
done

test -s outputs/latent_communication/bridge_sft/best.pt || {
  printf '%s STOP bridge_checkpoint_missing\n' "$(date -Iseconds)" >> "$log"
  exit 1
}
test -s outputs/latent_communication/bridge_sft/run_manifest.json || {
  printf '%s STOP bridge_manifest_missing\n' "$(date -Iseconds)" >> "$log"
  exit 1
}

# The primary monitor owns evaluation submission. A completed manifest lets a
# restarted watchdog skip scheduler-history reconstruction safely.
if test ! -s "$eval_dir/generation_manifest.json"; then
  waited=0
  while test ! -s "$status_dir/eval_submission.txt" && test "$waited" -lt 900; do
    sleep "$poll_seconds"
    waited=$((waited + poll_seconds))
  done
  test -s "$status_dir/eval_submission.txt" || {
    printf '%s STOP eval_submission_record_missing\n' "$(date -Iseconds)" >> "$log"
    exit 1
  }
  eval_job_id=$(sed -n 's/.*Job \([0-9][0-9]*\) submitted.*/\1/p' "$status_dir/eval_submission.txt" | tail -1)
  test -n "$eval_job_id" || {
    printf '%s STOP eval_job_id_unreadable\n' "$(date -Iseconds)" >> "$log"
    exit 1
  }
  printf '%s eval_job=%s\n' "$(date -Iseconds)" "$eval_job_id" >> "$log"

  while :; do
    if ! snapshot=$(pjstat 2>/dev/null); then
      printf '%s pjstat unavailable while checking eval\n' "$(date -Iseconds)" >> "$log"
      sleep "$poll_seconds"
      continue
    fi
    if ! printf '%s\n' "$snapshot" | awk -v id="$eval_job_id" '$1 == id {found=1} END {exit !found}'; then
      break
    fi
    printf '%s eval_still_active job=%s\n' "$(date -Iseconds)" "$eval_job_id" >> "$log"
    sleep "$poll_seconds"
  done
fi

test -s "$eval_dir/generation_manifest.json" || {
  printf '%s STOP evaluation_manifest_missing\n' "$(date -Iseconds)" >> "$log"
  exit 1
}

.venv-genkai/bin/python scripts/analyze_latent_group3_outputs.py \
  --input-dir "$eval_dir" \
  --output-dir "$review_dir" \
  --seeds 20260828 20260829 20260830 \
  >> "$log" 2>&1

# Produce copy-ready independent LLM judge packets. Private keys remain separate.
for seed in 20260828 20260829 20260830; do
  independent_dir="$eval_dir/independent_raters_seed${seed}"
  packet_dir="$eval_dir/llm_judge_packets_seed${seed}"
  .venv-genkai/bin/python scripts/build_independent_group_rater_packets.py \
    --trials "$eval_dir/blind_group3_seed${seed}.jsonl" \
    --output-dir "$independent_dir" \
    --raters llm_judge_1 llm_judge_2 \
    --seed "$seed" \
    >> "$log" 2>&1
  .venv-genkai/bin/python scripts/build_llm_blind_judge_packages.py \
    --trials "$eval_dir/blind_group3_seed${seed}.jsonl" \
    --private-key "${independent_dir}_private_key.json" \
    --output-dir "$packet_dir" \
    >> "$log" 2>&1
done

printf '%s analysis_complete status=AWAITING_BLIND_RATINGS report=%s\n' \
  "$(date -Iseconds)" "$review_dir/FIVE_HOUR_REVIEW.md" >> "$log"
printf '%s\n' "AWAITING_BLIND_RATINGS" > "$status_dir/next_gate.txt"
