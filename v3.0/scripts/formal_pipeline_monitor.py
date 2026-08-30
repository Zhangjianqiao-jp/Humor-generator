#!/usr/bin/env python3
"""Quota-free, idempotent PJM monitor for trace caching -> bridge matrix.

This process performs no model reasoning and never reads sealed test data. It
only advances through explicit artifact gates and records every submitted job
ID before attempting the next submission.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "data/cache/formal_pipeline_monitor_state.json"
CACHE = ROOT / "data/cache/planner_traces_homer_strict_v1"
DATASET = ROOT / "data/processed/latent_bridge_v3"
JOB_ID = re.compile(r"Job\s+(\d+)\s+submitted")
TRAINING = (
    ("learned_base", "br_l_base", "configs/formal/learned_base.yaml", "outputs/formal/learned_base"),
    ("typed_base", "br_t_base", "configs/formal/typed_base.yaml", "outputs/formal/typed_base"),
    ("learned_sft", "br_l_sft", "configs/formal/learned_sft.yaml", "outputs/formal/learned_sft"),
    ("typed_sft", "br_t_sft", "configs/formal/typed_sft.yaml", "outputs/formal/typed_sft"),
)


def iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_event(state: dict[str, Any], kind: str, **details: Any) -> None:
    state.setdefault("events", []).append({"time": iso(), "kind": kind, **details})
    state["updated_at"] = iso()


def command(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)


def active_job_ids(pjstat_output: str) -> set[str]:
    return {
        line.split()[0]
        for line in pjstat_output.splitlines()
        if line.strip() and line.split()[0].isdigit()
    }


def required_clusters() -> set[str]:
    return {
        row["cluster_id"]
        for split in ("train", "validation")
        for row in read_jsonl(DATASET / f"{split}.jsonl")
    }


def cache_evidence() -> dict[str, Any]:
    required = required_clusters()
    records = read_jsonl(CACHE / "index.jsonl")
    available = {row["cluster_id"] for row in records}
    failures_path = CACHE / "failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.is_file() else None
    return {
        "required": len(required),
        "records": len(records),
        "available": len(available),
        "duplicates": len(records) - len(available),
        "missing": len(required - available),
        "extra": len(available - required),
        "failures_file_present": failures is not None,
        "failures": None if failures is None else len(failures),
        "complete": (
            failures == []
            and available == required
            and len(records) == len(available)
        ),
    }


def parse_submitted_job(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    match = JOB_ID.search(result.stdout + result.stderr)
    if not match:
        raise RuntimeError(f"cannot parse pjsub output: {result.stdout!r} {result.stderr!r}")
    return match.group(1)


def submit_cache(retry_round: int) -> str:
    return parse_submitted_job(command([
        "pjsub", "-N", f"cache_strict{retry_round}",
        "-x", f"TRACE_RETRY_ROUND={retry_round}",
        "jobs/cache_formal_planner_traces.pjm",
    ]))


def submit_training(name: str, job_name: str, config: str, output: str) -> str:
    return parse_submitted_job(command([
        "pjsub", "-N", job_name,
        "-x", f"BRIDGE_CONFIG={config},BRIDGE_OUTPUT={output}",
        "jobs/formal_bridge_train.pjm",
    ]))


def verify_trace_gate() -> None:
    result = command([str(ROOT / ".venv/bin/python"), "scripts/check_trace_completion.py"])
    if result.returncode != 0:
        raise RuntimeError(f"trace gate failed:\n{result.stdout}\n{result.stderr}")


def initialize(state_path: Path, initial_job_id: str, retry_round: int) -> dict[str, Any]:
    if state_path.is_file():
        return json.loads(state_path.read_text())
    state = {
        "schema_version": 1,
        "created_at": iso(),
        "updated_at": iso(),
        "status": "monitoring_cache",
        "cache_job_id": initial_job_id,
        "cache_retry_round": retry_round,
        "training_jobs": {},
        "events": [],
    }
    append_event(state, "monitor_started", cache_job_id=initial_job_id, retry_round=retry_round)
    atomic_json(state_path, state)
    return state


def advance_once(
    state: dict[str, Any],
    state_path: Path,
    *,
    max_retry_rounds: int,
    confirm_absence_seconds: int,
) -> bool:
    """Advance once; return True only when the monitor should terminate."""
    status = command(["pjstat"])
    if status.returncode != 0:
        append_event(state, "pjstat_error", stderr=status.stderr.strip())
        atomic_json(state_path, state)
        return False
    active = active_job_ids(status.stdout)
    cache_job = str(state.get("cache_job_id") or "")
    if cache_job and cache_job in active:
        evidence = cache_evidence()
        state["last_cache_evidence"] = evidence
        state["status"] = "monitoring_cache"
        atomic_json(state_path, state)
        return False

    # A scheduler query can transiently omit jobs. Require a second clean
    # observation before any external submission.
    if confirm_absence_seconds:
        time.sleep(confirm_absence_seconds)
        again = command(["pjstat"])
        if again.returncode != 0:
            append_event(state, "pjstat_confirmation_error", stderr=again.stderr.strip())
            atomic_json(state_path, state)
            return False
        if cache_job and cache_job in active_job_ids(again.stdout):
            append_event(state, "cache_job_reappeared", job_id=cache_job)
            atomic_json(state_path, state)
            return False

    evidence = cache_evidence()
    state["last_cache_evidence"] = evidence
    if evidence["complete"]:
        verify_trace_gate()
        state["status"] = "submitting_bridge_matrix"
        append_event(state, "trace_gate_passed", evidence=evidence)
        atomic_json(state_path, state)
        jobs = state.setdefault("training_jobs", {})
        for name, job_name, config, output in TRAINING:
            if name in jobs:
                continue
            complete = ROOT / output / "complete.json"
            if complete.is_file():
                jobs[name] = {"job_id": None, "already_complete": True, "recorded_at": iso()}
            else:
                job_id = submit_training(name, job_name, config, output)
                jobs[name] = {"job_id": job_id, "already_complete": False, "recorded_at": iso()}
                append_event(state, "training_submitted", experiment=name, job_id=job_id)
            # Persist after every job to make a crash unable to duplicate all
            # subsequent submissions.
            atomic_json(state_path, state)
        state["status"] = "bridge_matrix_submitted"
        append_event(state, "monitor_complete", training_jobs=jobs)
        atomic_json(state_path, state)
        return True

    current_round = int(state.get("cache_retry_round", 0))
    # failures.json exists only after a full traversal. A wall-time-killed
    # partial traversal resumes the same deterministic round.
    next_round = current_round + 1 if evidence["failures_file_present"] else current_round
    if next_round > max_retry_rounds:
        state["status"] = "blocked_retry_limit"
        append_event(state, "retry_limit_reached", evidence=evidence, max_retry_rounds=max_retry_rounds)
        atomic_json(state_path, state)
        return True
    job_id = submit_cache(next_round)
    state.update({
        "status": "monitoring_cache",
        "cache_job_id": job_id,
        "cache_retry_round": next_round,
    })
    append_event(state, "cache_resubmitted", job_id=job_id, retry_round=next_round, evidence=evidence)
    atomic_json(state_path, state)
    return False


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--initial-job-id", required=True)
    parser.add_argument("--retry-round", type=int, default=0)
    parser.add_argument("--max-retry-rounds", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=1200)
    parser.add_argument("--confirm-absence-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.confirm_absence_seconds < 0:
        raise ValueError("invalid monitoring intervals")
    state = initialize(args.state, args.initial_job_id, args.retry_round)
    while True:
        done = advance_once(
            state, args.state,
            max_retry_rounds=args.max_retry_rounds,
            confirm_absence_seconds=args.confirm_absence_seconds,
        )
        if done or args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
