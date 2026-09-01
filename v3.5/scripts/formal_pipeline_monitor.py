#!/usr/bin/env python3
"""Quota-free, idempotent PJM monitor for trace caching -> three SFT-receiver pilots.

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
CACHE = ROOT / "data/cache/planner_traces_homer_strict_v35"
DATASET = ROOT / "data/processed/latent_bridge_v35"
JOB_ID = re.compile(r"Job\s+(\d+)\s+submitted")
TRAINING = (
    ("pilot_learned_sft_kl", "p_l_sft_k", "configs/pilot/learned_sft_kl.yaml", "outputs/pilot/learned_sft_kl"),
    ("pilot_typed_sft_kl", "p_t_sft_k", "configs/pilot/typed_sft_kl.yaml", "outputs/pilot/typed_sft_kl"),
    ("pilot_typed_sft_no_kl", "p_t_sft_n", "configs/pilot/typed_sft_no_kl.yaml", "outputs/pilot/typed_sft_no_kl"),
)
PILOT_EVALUATION_PACKET = ROOT / "outputs/pilot_validation/blind_packet.jsonl"


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


def submit_cache(
    retry_round: int,
    *,
    target_clusters: list[str] | None = None,
    validator_repair: bool = False,
    resource_group: str | None = None,
) -> str:
    environment = [f"TRACE_RETRY_ROUND={retry_round}"]
    if target_clusters:
        environment.append("TRACE_CLUSTER_IDS=" + ":".join(target_clusters))
    if validator_repair:
        environment.append("TRACE_VALIDATOR_REPAIR=1")
    job_file = (
        "jobs/recover_two_planner_traces.pjm"
        if validator_repair and target_clusters == ["nycc_325", "nycc_775"]
        else "jobs/cache_formal_planner_traces.pjm"
    )
    argv = ["pjsub", "-N", f"cache_strict{retry_round}"]
    if resource_group:
        argv.extend(["-L", f"rscgrp={resource_group},elapse=02:00:00,gpu=1"])
    argv.extend(["-x", ",".join(environment), job_file])
    return parse_submitted_job(command(argv))


def submit_training(
    name: str,
    job_name: str,
    config: str,
    output: str,
    *,
    resource_group: str | None = None,
    gpu_count: int = 1,
) -> str:
    argv = ["pjsub", "-N", job_name]
    if resource_group:
        argv.extend(["-L", f"rscgrp={resource_group},elapse=04:00:00,gpu={gpu_count}"])
    argv.extend([
        "-x", f"BRIDGE_CONFIG={config},BRIDGE_OUTPUT={output}",
        "jobs/formal_bridge_train.pjm",
    ])
    return parse_submitted_job(command(argv))


def submit_pilot_evaluation(
    *, resource_group: str | None = None, gpu_count: int = 1,
) -> str:
    argv = ["pjsub", "-N", "pilot_val_gen"]
    if resource_group:
        argv.extend(["-L", f"rscgrp={resource_group},elapse=04:00:00,gpu={gpu_count}"])
    argv.append("jobs/pilot_validation_generation.pjm")
    return parse_submitted_job(command(argv))


def verify_trace_gate() -> None:
    dataset = command([str(ROOT / ".venv/bin/python"), "scripts/verify_clustered_dataset.py"])
    if dataset.returncode != 0:
        raise RuntimeError(f"dataset gate failed:\n{dataset.stdout}\n{dataset.stderr}")
    records = read_jsonl(CACHE / "index.jsonl")
    if any(record.get("schema_version") != 3 for record in records):
        migration = command([
            str(ROOT / ".venv/bin/python"), "scripts/migrate_trace_provenance_v3.py"
        ])
        if migration.returncode != 0:
            raise RuntimeError(
                f"trace provenance migration failed:\n{migration.stdout}\n{migration.stderr}"
            )
    result = command([str(ROOT / ".venv/bin/python"), "scripts/check_trace_completion.py"])
    if result.returncode != 0:
        raise RuntimeError(f"trace gate failed:\n{result.stdout}\n{result.stderr}")


def initialize(
    state_path: Path,
    initial_job_id: str,
    retry_round: int,
    *,
    target_clusters: list[str] | None = None,
    validator_repair: bool = False,
    cache_resource_group: str | None = None,
    training_resource_group: str | None = None,
    training_gpu_count: int = 1,
) -> dict[str, Any]:
    if state_path.is_file():
        return json.loads(state_path.read_text())
    state = {
        "schema_version": 1,
        "created_at": iso(),
        "updated_at": iso(),
        "status": "monitoring_cache",
        "cache_job_id": initial_job_id,
        "cache_retry_round": retry_round,
        "target_clusters": target_clusters or [],
        "validator_repair": validator_repair,
        "cache_resource_group": cache_resource_group,
        "training_resource_group": training_resource_group,
        "training_gpu_count": training_gpu_count,
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
        if not state.get("trace_gate_recorded"):
            append_event(state, "trace_gate_passed", evidence=evidence)
            state["trace_gate_recorded"] = True
        jobs = state.setdefault("training_jobs", {})
        for name, job_name, config, output in TRAINING:
            complete = ROOT / output / "complete.json"
            if complete.is_file():
                if name not in jobs or not jobs[name].get("complete"):
                    jobs[name] = {
                        **jobs.get(name, {}), "complete": True, "completed_at": iso(),
                    }
                    append_event(state, "training_completed", experiment=name)
                continue
            existing = jobs.get(name)
            if existing is not None:
                job_id = str(existing.get("job_id") or "")
                if job_id in active:
                    state["status"] = f"monitoring_{name}"
                    atomic_json(state_path, state)
                    return False
                state["status"] = f"blocked_failed_{name}"
                append_event(state, "training_missing_without_complete", experiment=name, job_id=job_id)
                atomic_json(state_path, state)
                return True
            # At most one formal GPU job exists at a time. Submit the next
            # pilot only after the preceding complete.json passed.
            job_id = submit_training(
                name, job_name, config, output,
                resource_group=state.get("training_resource_group"),
                gpu_count=int(state.get("training_gpu_count", 1)),
            )
            jobs[name] = {
                "job_id": job_id, "complete": False, "recorded_at": iso(),
            }
            state["status"] = f"monitoring_{name}"
            append_event(state, "training_submitted", experiment=name, job_id=job_id)
            atomic_json(state_path, state)
            return False
        if PILOT_EVALUATION_PACKET.is_file():
            state["status"] = "pilot_validation_packet_ready_for_independent_raters"
            append_event(state, "pilot_validation_packet_ready", packet=str(PILOT_EVALUATION_PACKET))
            atomic_json(state_path, state)
            return True
        evaluation = state.get("pilot_evaluation_job")
        if evaluation is not None:
            job_id = str(evaluation.get("job_id") or "")
            if job_id in active:
                state["status"] = "monitoring_pilot_validation_generation"
                atomic_json(state_path, state)
                return False
            state["status"] = "blocked_failed_pilot_validation_generation"
            append_event(state, "pilot_validation_generation_missing", job_id=job_id)
            atomic_json(state_path, state)
            return True
        job_id = submit_pilot_evaluation(
            resource_group=state.get("training_resource_group"),
            gpu_count=int(state.get("training_gpu_count", 1)),
        )
        state["pilot_evaluation_job"] = {"job_id": job_id, "recorded_at": iso()}
        state["status"] = "monitoring_pilot_validation_generation"
        append_event(state, "pilot_validation_generation_submitted", job_id=job_id)
        atomic_json(state_path, state)
        return False

    current_round = int(state.get("cache_retry_round", 0))
    # failures.json exists only after a full traversal. A wall-time-killed
    # partial traversal resumes the same deterministic round.
    next_round = current_round + 1 if evidence["failures_file_present"] else current_round
    if next_round > max_retry_rounds:
        state["status"] = "blocked_retry_limit"
        append_event(state, "retry_limit_reached", evidence=evidence, max_retry_rounds=max_retry_rounds)
        atomic_json(state_path, state)
        return True
    job_id = submit_cache(
        next_round,
        target_clusters=list(state.get("target_clusters", [])),
        validator_repair=bool(state.get("validator_repair", False)),
        resource_group=state.get("cache_resource_group"),
    )
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
    parser.add_argument("--target-clusters", nargs="+")
    parser.add_argument("--validator-repair", action="store_true")
    parser.add_argument("--cache-resource-group", choices=["b-batch", "c-batch"])
    parser.add_argument(
        "--training-resource-group",
        choices=["b-batch", "b-batch-mig", "c-batch"],
    )
    parser.add_argument("--training-gpu-count", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=1200)
    parser.add_argument("--confirm-absence-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.poll_seconds < 1
        or args.confirm_absence_seconds < 0
        or args.training_gpu_count < 1
    ):
        raise ValueError("invalid monitoring intervals")
    state = initialize(
        args.state,
        args.initial_job_id,
        args.retry_round,
        target_clusters=args.target_clusters,
        validator_repair=args.validator_repair,
        cache_resource_group=args.cache_resource_group,
        training_resource_group=args.training_resource_group,
        training_gpu_count=args.training_gpu_count,
    )
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
