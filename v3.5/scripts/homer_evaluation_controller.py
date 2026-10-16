#!/usr/bin/env python3
"""Self-contained, resumable controller for the auxiliary HOMER evaluation.

The controller is deliberately independent of the chat context.  It records a
small immutable run manifest, starts one account-backed runner per evaluator,
and treats every batch JSON file as a checkpoint.  A quota interruption is
therefore just an interrupted worker: rerunning this command validates the
existing checkpoints, starts at the first missing batch, and never overwrites a
valid judgment.

This file never sends the private mapping or blind secret to an evaluator.  It
only uses the public packet shards while workers are running.  The private map
is consumed locally by ``finalize_homer_auxiliary_models.py`` after all three
public judgment sets pass the complete-set gate.

Typical usage (from the v3.5 directory):

    .venv/bin/python scripts/homer_evaluation_controller.py \
      --root outputs/caption_judgement/pretrained_bridge_comparison_20260911 \
      --mode start

Use ``--mode status`` after reconnecting to inspect progress without starting
anything.  The command is safe to run again after a quota reset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable


PROMPT_SHA256 = "98ebb56261262dd77112094330b056417ca133e2adf02ab5b6da60afdc31c524"
SOURCE_PROVENANCE_SHA256 = "44ea9297984ed1040e676a2589706aadc46b89dfa9b5650954a1cd7bec3434b1"
EXPECTED_MODELS = ("gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra")
MODEL_OUTPUTS = {
    "gpt-5.5": "judge_gpt55.jsonl",
    "gpt-5.6-sol": "judge_gpt56sol.jsonl",
    "gpt-5.6-terra": "judge_gpt56terra.jsonl",
}
TERRA_REAUDIT_OUTPUT = "judge_gpt56terra_reaudit.jsonl"
SUSPECT_TERRA = {
    "shard_001201_002400",
    "shard_006001_010800",
    "shard_010801_015600",
    "shard_025201_030000",
    "shard_030001_034800",
}
STATE_DIR_NAMES = {
    "gpt-5.5": "gpt55_account_batches",
    "gpt-5.6-sol": "sol_account_batches",
    "gpt-5.6-terra": "terra_reaudit_batches",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    return rows


def public_ids(path: Path) -> list[str]:
    rows = read_jsonl(path)
    ids = [row.get("packet_id") for row in rows]
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{path}: every public packet needs a non-empty packet_id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate packet_id")
    return ids


def shard_count(shard: Path) -> int:
    return len(public_ids(shard / "public_packets.jsonl"))


def shard_dirs(root: Path) -> list[Path]:
    directory = root / "multimodel_agent_shards"
    shards = sorted(path for path in directory.glob("shard_*") if path.is_dir())
    if not shards:
        raise ValueError(f"no packet shards under {directory}")
    return shards


def validate_public_layout(root: Path) -> tuple[list[Path], int, str]:
    canonical = root / "homer_primary" / "homer_primary_prompts.jsonl"
    provenance = root / "homer_primary" / "provenance.json"
    if not canonical.is_file() or not provenance.is_file():
        raise FileNotFoundError("homer_primary prompts/provenance are required")
    canonical_ids = public_ids(canonical)
    seen: list[str] = []
    for shard in shard_dirs(root):
        seen.extend(public_ids(shard / "public_packets.jsonl"))
    if seen != canonical_ids:
        raise ValueError("shard packets are not an exact ordered partition of canonical packets")
    provenance_obj = json.loads(provenance.read_text(encoding="utf-8"))
    evaluator = provenance_obj.get("evaluator", {})
    if evaluator.get("prompt_sha256") != PROMPT_SHA256:
        raise ValueError("official HOMER prompt hash mismatch")
    if evaluator.get("temperature") != 0:
        raise ValueError("official evaluator temperature must be 0")
    if sha256_file(provenance) != SOURCE_PROVENANCE_SHA256:
        raise ValueError("source provenance hash mismatch")
    return shard_dirs(root), len(canonical_ids), sha256_file(canonical)


def row_valid(row: dict[str, Any], model: str, expected_id: str) -> bool:
    return (
        row.get("packet_id") == expected_id
        and row.get("decision") in {"A", "B"}
        and row.get("model") == model
        and row.get("temperature") == 0
        and row.get("prompt_sha256") == PROMPT_SHA256
        and row.get("source_provenance_sha256") == SOURCE_PROVENANCE_SHA256
    )


def valid_output(path: Path, model: str, ids: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_jsonl(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return len(rows) == len(ids) and all(row_valid(row, model, expected) for row, expected in zip(rows, ids))


def checkpoint_progress(root: Path, model: str) -> dict[str, Any]:
    """Count valid per-batch checkpoints, even while a shard is unfinished."""
    state_dir = root / STATE_DIR_NAMES[model]
    if not state_dir.is_dir():
        return {"rows": 0, "batches": 0, "active": None}
    expected: dict[str, list[str]] = {
        shard.name: public_ids(shard / "public_packets.jsonl") for shard in shard_dirs(root)
    }
    pattern = re.compile(r"^(?P<start>\d+)(?:_(?P<end>\d+))?\.json$")
    candidates: list[tuple[str, int, int, int]] = []
    for path in state_dir.glob("*.json"):
        # Shard names themselves contain numeric ranges.  Parse the known
        # shard prefix first rather than greedily treating those numbers as
        # the batch range.
        matching_shards = [name for name in expected if path.name.startswith(name + "_")]
        if not matching_shards:
            continue
        shard_name = max(matching_shards, key=len)
        match = pattern.match(path.name[len(shard_name) + 1 :])
        if not match:
            continue
        start = int(match.group("start"))
        explicit_end = match.group("end")
        end = int(explicit_end) if explicit_end is not None else -1
        ids = expected[shard_name]
        if ids is None or not (0 <= start < len(ids)):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            decision_rows = payload.get("decisions")
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(decision_rows, list):
            continue
        if end < 0:
            end = start + len(decision_rows)
        if not (start < end <= len(ids)) or len(decision_rows) != end - start:
            continue
        if not all(isinstance(row, dict) for row in decision_rows):
            continue
        if [row.get("packet_id") for row in decision_rows] != ids[start:end]:
            continue
        if any(row.get("decision") not in {"A", "B"} for row in decision_rows):
            continue
        candidates.append((shard_name, start, end, len(decision_rows)))

    # Adaptive evaluator recovery may leave valid child checkpoints alongside
    # a validated parent checkpoint.  Count the largest non-overlapping
    # intervals so progress is not double-counted (e.g. 100 parent rows plus
    # two 50-row children must report 100 rows, one batch).
    selected: list[tuple[str, int, int, int]] = []
    for candidate in sorted(candidates, key=lambda item: (-(item[2] - item[1]), item[0], item[1], item[2])):
        shard_name, start, end, length = candidate
        if any(
            shard_name == chosen[0] and start < chosen[2] and chosen[1] < end
            for chosen in selected
        ):
            continue
        selected.append(candidate)
    rows_total = sum(item[3] for item in selected)
    batches = len(selected)
    latest: tuple[str, int, int] | None = None
    for shard_name, start, end, _ in selected:
        candidate = (shard_name, start, end)
        if latest is None or candidate[0] > latest[0] or (candidate[0] == latest[0] and candidate[1] > latest[1]):
            latest = candidate
    return {
        "rows": rows_total,
        "batches": batches,
        "active": None if latest is None else {"shard": latest[0], "start": latest[1], "end": latest[2]},
    }


def worker_block_path(root: Path, model: str) -> Path:
    return root / STATE_DIR_NAMES[model] / "worker_blocked.json"


def read_worker_block(root: Path, model: str) -> dict[str, Any] | None:
    path = worker_block_path(root, model)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "kind": "malformed_block_marker",
            "retryable": False,
            "path": str(path),
        }
    if not isinstance(value, dict) or value.get("model") != model or not value.get("kind"):
        return {
            "kind": "malformed_block_marker",
            "retryable": False,
            "path": str(path),
        }
    return value


def active_worker_block(root: Path, model: str, log_path: Path | None = None) -> dict[str, Any] | None:
    """Return a block marker that should suppress another costly invocation.

    Quota markers with a parsed future reset time are automatically released
    at that time.  Unsupported-model and expired-auth markers remain paused
    until the user fixes the account/model and removes the marker; this avoids
    an endless two-minute retry loop while preserving every checkpoint.
    """
    value = read_worker_block(root, model)
    if value is None:
        return None
    retry_after = value.get("retry_after_epoch")
    if value.get("kind") == "quota_exhausted" and isinstance(retry_after, (int, float)):
        if time.time() >= float(retry_after):
            with contextlib.suppress(OSError):
                worker_block_path(root, model).unlink()
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{now()} quota wake-up reached for {model}; resuming\n")
            return None
    return value


def all_complete(root: Path, model: str, filename: str | None = None) -> bool:
    filename = filename or MODEL_OUTPUTS[model]
    for shard in shard_dirs(root):
        if not valid_output(shard / filename, model, public_ids(shard / "public_packets.jsonl")):
            return False
    return True


def terra_approved(root: Path) -> bool:
    """Return true only after the five legacy-looking Terra shards were audited.

    A complete-looking ``judge_gpt56terra.jsonl`` is not sufficient: five old
    files were generated by an invalid heuristic path.  The promotion marker
    is written atomically only after every re-audit file passes the same strict
    packet/model/provenance checks and the old file has been backed up.
    """
    marker = root / "terra_reaudit_promotion.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return value.get("status") == "promoted" and all_complete(root, "gpt-5.6-terra")


def progress(root: Path) -> dict[str, Any]:
    shards, total, canonical_sha = validate_public_layout(root)
    models: dict[str, Any] = {}
    for model in EXPECTED_MODELS:
        filename = MODEL_OUTPUTS[model]
        done = sum(
            shard_count(shard)
            for shard in shards
            if valid_output(shard / filename, model, public_ids(shard / "public_packets.jsonl"))
        )
        approved = terra_approved(root) if model == "gpt-5.6-terra" else True
        models[model] = {
            "rows": done,
            "total": total,
            "complete": done == total and approved,
            "checkpoint": checkpoint_progress(root, model),
            "blocked": read_worker_block(root, model),
        }
    # Promotion atomically moves each validated re-audit file to the official
    # Terra output name and leaves only ``terra_reaudit_promotion.json`` as the
    # source-side sentinel.  Once that marker and every promoted target pass,
    # report the five audited shards as complete instead of looking for files
    # that promotion intentionally moved away.
    approved = terra_approved(root)
    terra_reaudit = {
        shard.name: (
            True
            if approved
            else valid_output(
                shard / TERRA_REAUDIT_OUTPUT,
                "gpt-5.6-terra",
                public_ids(shard / "public_packets.jsonl"),
            )
        )
        for shard in shards
        if shard.name in SUSPECT_TERRA
    }
    return {
        "updated_at": now(),
        "canonical_packets": total,
        "canonical_packets_sha256": canonical_sha,
        "models": models,
        "terra_reaudit": terra_reaudit,
        "terra_reaudit_complete": approved or (all(terra_reaudit.values()) if terra_reaudit else False),
    }


def initialize(root: Path, script_dir: Path) -> dict[str, Any]:
    shards, total, canonical_sha = validate_public_layout(root)
    manifest_path = root / "evaluation_manifest.json"
    existing: dict[str, Any] | None = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("canonical_packets") != total or existing.get("prompt_sha256") != PROMPT_SHA256:
            raise ValueError("existing evaluation_manifest.json does not match the packet set")
    manifest = existing or {
        "schema_version": 1,
        "created_at": now(),
        "controller": "scripts/homer_evaluation_controller.py",
        "protocol": "HOMER official anonymous pairwise text evaluator",
        "transport": "codex-exec-account; no API key",
        "models": list(EXPECTED_MODELS),
        "unavailable_models": ["gpt-5.4"],
        "canonical_packets": total,
        "canonical_packets_sha256": canonical_sha,
        "prompt_sha256": PROMPT_SHA256,
        "source_provenance_sha256": SOURCE_PROVENANCE_SHA256,
        "public_only_during_judging": True,
        "private_mapping_used_only_by_local_finalizer": True,
        "temperature_protocol": "requested-0-not-exposed-by-codex-cli",
        "shards": [{"name": shard.name, "rows": shard_count(shard)} for shard in shards],
        "outputs": MODEL_OUTPUTS,
        "terra_reaudit_shards": sorted(SUSPECT_TERRA),
        "source_commit": current_commit(script_dir),
        "state": "initialized",
    }
    manifest["last_initialized_at"] = now()
    # ``source_commit`` identifies the commit that produced the canonical
    # packet/provenance.  Keep it immutable; record controller changes
    # separately so a resumed evaluation remains fully reproducible.
    manifest["controller_commit"] = current_commit(script_dir)
    manifest["last_controller_commit"] = manifest["controller_commit"]
    atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_progress(root, progress(root))
    return manifest


def current_commit(script_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=script_dir.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_progress(root: Path, value: dict[str, Any]) -> None:
    atomic_write(root / "evaluation_progress.json", json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def process_has(marker: str) -> bool:
    current = os.getpid()
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    for row in rows:
        fields = row.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid != current and marker in fields[1]:
            return True
    return False


def worker_process_info(root: Path, model: str) -> dict[str, Any] | None:
    """Return this evaluation's runner handle, never an unrelated model call.

    Matching a bare ``--model`` flag is unsafe on a shared server: another
    experiment using the same account/model could suppress this run forever.
    The runner command must contain both this resolved evaluation root and the
    exact runner script.  Legacy ad-hoc runners are checked separately by
    their unique filename in ``run_once`` for backwards compatibility.
    """
    current = os.getpid()
    root_marker = str(root.resolve())
    model_marker = f"--model {model}"
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    for row in rows:
        fields = row.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        command = fields[1]
        if (
            pid != current
            and "run_homer_account_agent_batches.py" in command
            and root_marker in command
            and model_marker in command
        ):
            return {"pid": pid, "command": command}
    return None


def worker_process_has(root: Path, model: str) -> bool:
    return worker_process_info(root, model) is not None


def record_worker_handle(state: dict[str, Any], model: str, pid: int, command: list[str]) -> None:
    """Persist a lightweight handle so a reconnect can audit what was armed."""
    handles = state.setdefault("worker_handles", {})
    handles[model] = {
        "pid": int(pid),
        "started_at": now(),
        "command": command,
    }


def launch_allowed(state: dict[str, Any], model: str, max_restarts: int) -> bool:
    """Bound repeated launches that make no checkpoint progress.

    ``restart_counts`` is retained as historical telemetry and may already be
    large in an old run.  The safety budget is a separate counter that resets
    whenever a model produces new valid checkpoint rows, so upgrading this
    controller cannot unexpectedly block a healthy in-flight evaluation.
    """
    attempts = int(state.setdefault("no_progress_launches", {}).get(model, 0))
    if attempts < max(1, max_restarts):
        return True
    state.setdefault("restart_limit_models", {})[model] = {
        "kind": "restart_limit_exceeded",
        "attempts_without_progress": attempts,
        "max_restarts": max_restarts,
        "at": now(),
    }
    return False


def account_environment() -> dict[str, str]:
    """Run account-backed Codex without accidentally forwarding API keys."""
    environment = os.environ.copy()
    # Codex account authentication comes from its local login state, not an
    # API credential.  Scrub common provider key names (including variants
    # that may be injected by a parent shell) before spawning any evaluator.
    explicit_key_names = {
        "OPENAI_API_KEY",
        "OPEN_API_KEY",
        "CODEX_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "COHERE_API_KEY",
    }
    for name in list(environment):
        if name in explicit_key_names or name.upper().endswith("_API_KEY"):
            environment.pop(name, None)
    return environment


def start_worker(root: Path, script_dir: Path, model: str, log_path: Path) -> tuple[int, list[str]]:
    python = script_dir.parent / ".venv" / "bin" / "python"
    runner = script_dir / "run_homer_account_agent_batches.py"
    state_dir = root / {
        "gpt-5.5": "gpt55_account_batches",
        "gpt-5.6-sol": "sol_account_batches",
    }.get(model, f"{model.replace('.', '').replace('-', '_')}_account_batches")
    state_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), str(runner), "--root", str(root), "--model", model,
        "--output-name", MODEL_OUTPUTS[model], "--state-dir", str(state_dir),
        "--batch-size", "100", "--shards", *(shard.name for shard in shard_dirs(root)),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{now()} start {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=account_environment(),
        )
    return process.pid, command


def start_terra_worker(root: Path, script_dir: Path, log_path: Path) -> tuple[int, list[str]]:
    """Start the bounded Terra re-audit using the same resumable runner."""
    state_dir = root / "terra_reaudit_batches"
    state_dir.mkdir(parents=True, exist_ok=True)
    python = script_dir.parent / ".venv" / "bin" / "python"
    runner = script_dir / "run_homer_account_agent_batches.py"
    command = [
        str(python), str(runner), "--root", str(root), "--model", "gpt-5.6-terra",
        "--output-name", TERRA_REAUDIT_OUTPUT, "--state-dir", str(state_dir),
        "--batch-size", "100", "--shards", *sorted(SUSPECT_TERRA),
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{now()} start Terra re-audit {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=account_environment(),
        )
    return process.pid, command


def promote_terra(root: Path, log_path: Path) -> bool:
    shards = {shard.name: shard for shard in shard_dirs(root)}
    if any(name not in shards for name in SUSPECT_TERRA):
        raise ValueError("configured Terra re-audit shard is missing")
    if not all(valid_output(shards[name] / TERRA_REAUDIT_OUTPUT, "gpt-5.6-terra", public_ids(shards[name] / "public_packets.jsonl")) for name in SUSPECT_TERRA):
        return False
    for name in sorted(SUSPECT_TERRA):
        shard = shards[name]
        source = shard / TERRA_REAUDIT_OUTPUT
        target = shard / MODEL_OUTPUTS["gpt-5.6-terra"]
        backup = shard / "judge_gpt56terra.heuristic_invalid.jsonl"
        if target.exists() and not backup.exists():
            os.replace(target, backup)
        os.replace(source, target)
    atomic_write(root / "terra_reaudit_promotion.json", json.dumps({"status": "promoted", "at": now(), "shards": sorted(SUSPECT_TERRA)}, indent=2) + "\n")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} promoted audited Terra shards\n")
    return True


def finalization_complete(root: Path, expected_packets: int) -> bool:
    """Validate the final report instead of trusting its filename.

    The finalizer writes JSON/Markdown near the end of a long local
    aggregation.  A process or node interruption can leave a partial JSON
    file at the final name.  Recovery must therefore inspect the completion
    sentinel and the evaluator/packet contract before deciding not to rerun.
    """
    output_dir = root / "multimodel_agent_final"
    summary_path = output_dir / "multimodel_summary.json"
    markdown_path = output_dir / "multimodel_summary.md"
    if not summary_path.is_file() or not markdown_path.is_file() or markdown_path.stat().st_size == 0:
        return False
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if value.get("status") != "complete_auxiliary_multimodel":
        return False
    if value.get("packet_count") != expected_packets:
        return False
    evaluators = value.get("evaluators")
    if not isinstance(evaluators, list):
        return False
    model_ids = {item.get("model_id") for item in evaluators if isinstance(item, dict)}
    if model_ids != set(EXPECTED_MODELS) or len(evaluators) != len(EXPECTED_MODELS):
        return False
    # A complete-looking JSON with only evaluator names is not enough: require
    # the metric payload and the three-way consensus produced by the local
    # finalizer.  This prevents a kill during aggregation from being mistaken
    # for a scientifically complete report.
    for item in evaluators:
        metrics = item.get("metrics") if isinstance(item, dict) else None
        if not isinstance(metrics, dict) or not metrics.get("summaries") or not metrics.get("paired_comparison"):
            return False
    return isinstance(value.get("majority"), dict) and isinstance(value.get("evaluator_model_ids"), list)


def run_finalize(root: Path, script_dir: Path, log_path: Path) -> int:
    python = script_dir.parent / ".venv" / "bin" / "python"
    merged = root / "multimodel_agent_merged"
    final = root / "multimodel_agent_final"
    merge = [
        str(python), str(script_dir / "merge_homer_agent_shards.py"),
        "--packets", str(root / "homer_primary" / "homer_primary_prompts.jsonl"),
        "--shard-dir", str(root / "multimodel_agent_shards"), "--output-dir", str(merged),
    ]
    for model, filename in MODEL_OUTPUTS.items():
        merge.extend(["--models", f"{model}={filename}"])
    merge.append("--require-complete")
    final_cmd = [
        str(python), str(script_dir / "finalize_homer_auxiliary_models.py"),
        "--private-mapping", str(root / "homer_primary" / "homer_primary_private_mapping.jsonl"),
        "--provenance", str(root / "homer_primary" / "provenance.json"),
        "--output-dir", str(final), "--bootstrap-replicates", "2000",
    ]
    for model in EXPECTED_MODELS:
        final_cmd.extend(["--judgments", f"{model}={merged / MODEL_OUTPUTS[model]}"])
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{now()} merge {' '.join(merge)}\n")
        merged_proc = subprocess.run(merge, stdout=handle, stderr=subprocess.STDOUT)
        if merged_proc.returncode != 0:
            return merged_proc.returncode
        handle.write(f"\n{now()} finalize {' '.join(final_cmd)}\n")
        final_proc = subprocess.run(final_cmd, stdout=handle, stderr=subprocess.STDOUT)
    return final_proc.returncode


def checkpoint_rows(value: dict[str, Any], model: str) -> int:
    """Read a model's durable checkpoint count from a previous state file."""
    try:
        item = value.get("models", {}).get(model, {})
        checkpoint = item.get("checkpoint", {})
        return int(checkpoint.get("rows", 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def run_once(root: Path, script_dir: Path, max_restarts: int) -> tuple[dict[str, Any], bool]:
    manifest = initialize(root, script_dir)
    log_path = root / "evaluation_controller.log"
    state_path = root / "evaluation_controller_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "restart_counts": {model: 0 for model in EXPECTED_MODELS}, "finalize_attempts": 0,
    }
    if not isinstance(state.get("restart_counts"), dict):
        state["restart_counts"] = {}
    for model in EXPECTED_MODELS:
        state.setdefault("restart_counts", {}).setdefault(model, 0)
    if not isinstance(state.get("finalize_attempts"), int):
        state["finalize_attempts"] = 0
    if not isinstance(state.get("worker_handles"), dict):
        state["worker_handles"] = {}
    if not isinstance(state.get("no_progress_launches"), dict):
        state["no_progress_launches"] = {model: 0 for model in EXPECTED_MODELS}
    if not isinstance(state.get("restart_limit_models"), dict):
        state["restart_limit_models"] = {}
    previous_progress = state.get("progress", {})
    state["last_cycle_at"] = now()
    if not isinstance(state.get("block_signatures"), dict):
        state["block_signatures"] = {}
    state["blocked_models"] = {}
    # Reconcile workers launched by an older controller instance.  This makes
    # a controller restart observable without requiring the old parent PID to
    # still exist.
    for model in EXPECTED_MODELS:
        info = worker_process_info(root, model)
        if info is not None:
            handle = state["worker_handles"].get(model)
            if not isinstance(handle, dict) or handle.get("pid") != info["pid"]:
                record_worker_handle(state, model, int(info["pid"]), [info["command"]])

    terra_block = active_worker_block(root, "gpt-5.6-terra", log_path)
    if terra_block is not None:
        state["blocked_models"]["gpt-5.6-terra"] = terra_block
        signature = json.dumps(terra_block, sort_keys=True, ensure_ascii=False)
        if state["block_signatures"].get("gpt-5.6-terra") != signature:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{now()} paused gpt-5.6-terra: {terra_block.get('kind')}\n")
            state["block_signatures"]["gpt-5.6-terra"] = signature
    if not terra_approved(root) and terra_block is None:
        # Promote first.  This matters when the worker finished just before a
        # quota interruption: the complete re-audit output is already a valid
        # checkpoint and must not trigger another pass.
        promote_terra(root, log_path)
    if not terra_approved(root):
        if not worker_process_has(root, "gpt-5.6-terra"):
            # Terra uses a separate output name for the five audited legacy
            # shards.  The generic runner is only started for those shards.
            if launch_allowed(state, "gpt-5.6-terra", max_restarts):
                pid, command = start_terra_worker(root, script_dir, log_path)
                record_worker_handle(state, "gpt-5.6-terra", pid, command)
                state["no_progress_launches"]["gpt-5.6-terra"] += 1
                state["restart_counts"]["gpt-5.6-terra"] += 1
    for model, ad_hoc_marker in (("gpt-5.5", "gpt55_batch_runner.py"), ("gpt-5.6-sol", "sol_batch_runner.py")):
        block = active_worker_block(root, model, log_path)
        if block is not None:
            state["blocked_models"][model] = block
            signature = json.dumps(block, sort_keys=True, ensure_ascii=False)
            if state["block_signatures"].get(model) != signature:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{now()} paused {model}: {block.get('kind')}\n")
                state["block_signatures"][model] = signature
            continue
        running = process_has(ad_hoc_marker) or worker_process_has(root, model)
        if not all_complete(root, model) and not running and launch_allowed(state, model, max_restarts):
            pid, command = start_worker(root, script_dir, model, log_path)
            record_worker_handle(state, model, pid, command)
            state["no_progress_launches"][model] += 1
            state["restart_counts"][model] += 1

    current = progress(root)
    # A launch budget is about repeated failures, not the historical number of
    # times older controllers happened to be restarted.  Reset it whenever a
    # durable checkpoint advances; this keeps recovery safe after a controller
    # upgrade while still stopping a persistent crash loop.
    for model in EXPECTED_MODELS:
        old_rows = checkpoint_rows(previous_progress, model)
        new_rows = checkpoint_rows(current, model)
        if new_rows > old_rows:
            state["no_progress_launches"][model] = 0
            state["restart_limit_models"].pop(model, None)
        handle = state["worker_handles"].get(model)
        if isinstance(handle, dict):
            handle["active"] = worker_process_has(root, model)
    write_progress(root, current)
    state["progress"] = current
    state["blocked_models"] = {
        model: item["blocked"]
        for model, item in current["models"].items()
        if item.get("blocked") is not None
    }
    done = all(item["complete"] for item in current["models"].values()) and current["terra_reaudit_complete"]
    if done and not finalization_complete(root, current["canonical_packets"]):
        # A malformed final-name report must not block recovery.  Preserve it
        # as evidence, then let the strict merge/finalizer reconstruct all
        # outputs from the immutable shard judgments.
        summary_path = root / "multimodel_agent_final" / "multimodel_summary.json"
        if summary_path.exists():
            quarantine_invalid_result(summary_path)
            markdown_path = root / "multimodel_agent_final" / "multimodel_summary.md"
            if markdown_path.exists():
                quarantine_invalid_result(markdown_path)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{now()} quarantined incomplete final summary before retry\n")
        if state["finalize_attempts"] >= max_restarts:
            state["state"] = "finalize_retry_limit_exceeded"
            done = False
        else:
            state["finalize_attempts"] += 1
            return_code = run_finalize(root, script_dir, log_path)
            state["last_finalize_returncode"] = return_code
            state["state"] = "complete" if return_code == 0 else "finalize_failed_retryable"
            done = return_code == 0
    else:
        state["state"] = (
            "complete" if done
            else "blocked_waiting" if state["blocked_models"]
            else "restart_limit_exceeded" if state["restart_limit_models"]
            else "running"
        )
    atomic_write(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    return current, done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quota-resumable, no-API-key HOMER agent evaluation controller")
    parser.add_argument("--root", required=True, type=Path, help="evaluation root containing homer_primary/ and multimodel_agent_shards/")
    parser.add_argument("--mode", choices=("status", "once", "start"), default="status")
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means keep running until finalization")
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="status/once output as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    script_dir = Path(__file__).resolve().parent
    try:
        lock_handle = None
        if args.mode != "status":
            lock_path = root / ".evaluation_controller.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_handle = lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"another evaluation controller already owns {lock_path}", file=sys.stderr)
                return 3
        if args.mode == "status":
            manifest = initialize(root, script_dir)
            value = progress(root)
            value["manifest"] = str(root / "evaluation_manifest.json")
            state_path = root / "evaluation_controller_state.json"
            if state_path.is_file():
                try:
                    state_value = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    state_value = {"state_read_error": True}
                if isinstance(state_value, dict):
                    # Keep status context-free but expose only persisted
                    # controller telemetry; packet contents and private map
                    # are never included here.
                    value["controller_state"] = {
                        key: state_value.get(key)
                        for key in (
                            "state",
                            "last_cycle_at",
                            "worker_handles",
                            "no_progress_launches",
                            "restart_limit_models",
                            "finalize_attempts",
                            "blocked_models",
                        )
                        if key in state_value
                    }
            print(json.dumps(value, ensure_ascii=False, indent=2) if args.json else format_status(value))
            return 0
        if args.mode == "once":
            value, done = run_once(root, script_dir, args.max_restarts)
            print(json.dumps(value, ensure_ascii=False, indent=2) if args.json else format_status(value))
            return 0
        cycles = 0
        while True:
            value, done = run_once(root, script_dir, args.max_restarts)
            if not args.json:
                print(format_status(value), flush=True)
            if done:
                if args.json:
                    print(json.dumps(value, ensure_ascii=False, indent=2))
                return 0
            cycles += 1
            if args.max_cycles and cycles >= args.max_cycles:
                if args.json:
                    print(json.dumps(value, ensure_ascii=False, indent=2))
                return 2
            time.sleep(max(10, args.interval_seconds))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if 'lock_handle' in locals() and lock_handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def format_status(value: dict[str, Any]) -> str:
    lines = [f"updated={value.get('updated_at')} packets={value.get('canonical_packets')}"]
    for model, item in value.get("models", {}).items():
        checkpoint = item.get("checkpoint", {})
        checkpoint_text = f", checkpoint={checkpoint.get('rows', 0)} rows/{checkpoint.get('batches', 0)} batches"
        blocked = item.get("blocked")
        state = f"blocked:{blocked.get('kind')}" if isinstance(blocked, dict) else ("complete" if item.get("complete") else "running")
        lines.append(f"{model}: {item.get('rows')}/{item.get('total')} ({state}){checkpoint_text}")
    terra = value.get("terra_reaudit", {})
    if terra:
        lines.append(f"terra re-audit: {sum(terra.values())}/{len(terra)} shards complete")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
