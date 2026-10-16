#!/usr/bin/env python3
"""Run account-backed Codex agent judgments for HOMER public packets.

This runner intentionally reads only public packet shards. It never opens the
private mapping or blind secret files. Each Codex call is constrained by a JSON
schema and saved as batch evidence before the merged shard judgment is written.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROMPT_SHA256 = "98ebb56261262dd77112094330b056417ca133e2adf02ab5b6da60afdc31c524"
SOURCE_PROVENANCE_SHA256 = "44ea9297984ed1040e676a2589706aadc46b89dfa9b5650954a1cd7bec3434b1"
TRANSIENT_RETRY_SECONDS = 300


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "packet_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["packet_id", "decision"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_retry_after(message: str) -> tuple[float | None, str | None]:
    """Parse the reset timestamp printed by Codex usage-limit errors.

    The account message is presented in local account time and may include an
    ordinal suffix (``13th``).  We preserve the original text in the block
    marker and only use this parsed value as a conservative wake-up time.
    """
    match = re.search(
        r"try again at\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None
    value = re.sub(r"(\d{1,2})(?:st|nd|rd|th)", r"\1", match.group(1))
    value = re.sub(r"\s*,\s*", ", ", value)
    parsed = None
    for fmt in ("%B %d, %Y %I:%M %p", "%B %d %Y %I:%M %p", "%b %d, %Y %I:%M %p", "%b %d %Y %I:%M %p"):
        try:
            parsed = datetime.strptime(value, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None, value
    local = parsed.astimezone()
    return local.timestamp(), local.isoformat()


def classify_codex_failure(message: str) -> dict[str, Any] | None:
    """Classify failures that should pause, rather than hot-loop retries."""
    lowered = message.lower()
    if "not supported when using codex with a chatgpt account" in lowered:
        return {
            "kind": "model_unsupported",
            "retryable": False,
            "retry_after_epoch": None,
            "retry_after": None,
        }
    if "token_expired" in lowered or "provided authentication token is expired" in lowered:
        return {
            "kind": "auth_expired",
            "retryable": False,
            "retry_after_epoch": None,
            "retry_after": None,
        }
    if "you've hit your usage limit" in lowered or "you have hit your usage limit" in lowered:
        epoch, timestamp = parse_retry_after(message)
        return {
            "kind": "quota_exhausted",
            "retryable": True,
            "retry_after_epoch": epoch,
            "retry_after": timestamp,
        }
    if "timed out" in lowered or "timeout" in lowered:
        return {
            "kind": "call_timeout",
            "retryable": True,
            "retry_after_epoch": time.time() + TRANSIENT_RETRY_SECONDS,
            "retry_after": None,
        }
    if any(
        marker in lowered
        for marker in (
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "service unavailable",
            "internal server error",
            "connection reset",
            "connection refused",
            "network is unreachable",
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
        )
    ):
        return {
            "kind": "transient_error",
            "retryable": True,
            "retry_after_epoch": time.time() + TRANSIENT_RETRY_SECONDS,
            "retry_after": None,
        }
    return None


def sanitized_environment() -> dict[str, str]:
    """Run account-backed Codex without forwarding provider API keys.

    The controller already scrubs its child environment.  The runner repeats
    the guard so that a user who invokes this file directly cannot accidentally
    leak a key into ``codex exec``.  Account authentication is read from the
    local Codex login state instead.
    """
    environment = os.environ.copy()
    for name in list(environment):
        if name.upper().endswith("_API_KEY"):
            environment.pop(name, None)
    return environment


def write_block_marker(
    state_dir: Path,
    *,
    model: str,
    batch: str,
    failure: dict[str, Any],
    message: str,
) -> None:
    """Persist a machine-readable pause reason for the resumable controller."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "worker_blocked.json"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    payload = {
        "schema_version": 1,
        "model": model,
        "batch": batch,
        "kind": failure["kind"],
        "retryable": bool(failure["retryable"]),
        "retry_after_epoch": failure.get("retry_after_epoch"),
        "retry_after": failure.get("retry_after"),
        "message": message[-2000:],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--call-timeout-seconds",
        type=float,
        default=1800.0,
        help="maximum duration of one account-agent call before a resumable timeout pause",
    )
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--codex-bin", default="codex")
    return parser.parse_args()


def validate_batch(
    rows: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    result_path: Path,
) -> None:
    if len(rows) != len(batch):
        raise RuntimeError(f"Batch count mismatch in {result_path}: {len(rows)} != {len(batch)}")
    row_ids = [row.get("packet_id") for row in rows]
    packet_ids = [packet["packet_id"] for packet in batch]
    if row_ids != packet_ids:
        raise RuntimeError(f"Batch order mismatch in {result_path}")
    bad = [row for row in rows if row.get("decision") not in {"A", "B"}]
    if bad:
        raise RuntimeError(f"Invalid decisions in {result_path}: {bad[:3]}")


def validate_output(
    rows: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    *,
    model: str,
) -> None:
    """Validate an already materialized shard output before reusing it.

    A previous runner can have been killed between versions, leaving a file
    with the expected name but partial or stale content.  Treating mere file
    existence as completion would make a resumed evaluation loop forever.
    Reuse is therefore allowed only for the same ordered packet set and the
    same provenance contract used by the controller/finalizer.
    """
    if len(rows) != len(packets):
        raise RuntimeError(f"Output count mismatch: {len(rows)} != {len(packets)}")
    for row, packet in zip(rows, packets):
        if row.get("packet_id") != packet.get("packet_id"):
            raise RuntimeError("Output packet ID/order mismatch")
        if row.get("decision") not in {"A", "B"}:
            raise RuntimeError("Output decision is not A/B")
        if row.get("model") != model:
            raise RuntimeError("Output model mismatch")
        if row.get("temperature") != 0:
            raise RuntimeError("Output temperature mismatch")
        if row.get("prompt_sha256") != PROMPT_SHA256:
            raise RuntimeError("Output prompt hash mismatch")
        if row.get("source_provenance_sha256") != SOURCE_PROVENANCE_SHA256:
            raise RuntimeError("Output source provenance hash mismatch")


def quarantine_invalid_result(result_path: Path) -> None:
    """Keep a malformed/truncated response as evidence, then allow a retry."""
    if not result_path.exists():
        return
    index = 0
    while True:
        candidate = result_path.with_name(result_path.name + f".invalid{index}")
        if not candidate.exists():
            result_path.rename(candidate)
            return
        index += 1


class RunnerBlocked(RuntimeError):
    """A non-data failure that must pause the resumable worker."""

    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def write_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically materialise a validated checkpoint object."""
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps({"decisions": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_batch_with_recovery(
    *,
    args: argparse.Namespace,
    schema_path: Path,
    state_dir: Path,
    shard_name: str,
    packets: list[dict[str, Any]],
    start: int,
    end: int,
    rubric: str,
) -> tuple[list[dict[str, Any]], Path]:
    """Run one packet range, recursively splitting repeated truncations.

    Account-backed model calls occasionally return one fewer JSON row than a
    large request.  That is a transport/output-capacity failure, not evidence
    about a caption.  After three schema/count failures we bisect the range,
    preserve every invalid response as ``.invalidN`` evidence, and write a
    parent checkpoint only after both children validate.  Existing valid
    checkpoints are always reused first, so recovery never spends calls on
    completed ranges.
    """
    if not (0 <= start < end <= len(packets)):
        raise ValueError(f"invalid packet range [{start}, {end})")
    batch = packets[start:end]
    batch_tag = f"{shard_name}_{start:06d}_{end:06d}"
    result_path = state_dir / f"{batch_tag}.json"
    legacy_result_path = state_dir / f"{shard_name}_{start}.json"
    log_path = state_dir / f"{batch_tag}.log"

    def existing_checkpoint() -> tuple[list[dict[str, Any]], Path] | None:
        for checkpoint_path in (result_path, legacy_result_path):
            if not checkpoint_path.exists():
                continue
            try:
                existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                candidate_rows = existing["decisions"]
                validate_batch(candidate_rows, batch, checkpoint_path)
                return candidate_rows, checkpoint_path
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, RuntimeError, TypeError):
                quarantine_invalid_result(checkpoint_path)
        return None

    def split_range() -> tuple[list[dict[str, Any]], Path]:
        midpoint = start + (end - start) // 2
        if midpoint <= start or midpoint >= end:
            raise RunnerBlocked(2, f"cannot split singleton invalid batch {batch_tag}")
        print(
            f"Splitting malformed evaluator batch {batch_tag} into "
            f"[{start}, {midpoint}) and [{midpoint}, {end})",
            flush=True,
        )
        left_rows, _ = run_batch_with_recovery(
            args=args,
            schema_path=schema_path,
            state_dir=state_dir,
            shard_name=shard_name,
            packets=packets,
            start=start,
            end=midpoint,
            rubric=rubric,
        )
        right_rows, _ = run_batch_with_recovery(
            args=args,
            schema_path=schema_path,
            state_dir=state_dir,
            shard_name=shard_name,
            packets=packets,
            start=midpoint,
            end=end,
            rubric=rubric,
        )
        combined = left_rows + right_rows
        validate_batch(combined, batch, result_path)
        write_checkpoint(result_path, combined)
        return combined, result_path

    checkpoint = existing_checkpoint()
    if checkpoint is not None:
        return checkpoint

    # If a previous process already recorded three or more malformed parent
    # responses, do not waste another large call before applying the recovery
    # policy.  The invalid files remain immutable audit evidence.
    prior_invalid = len(list(state_dir.glob(result_path.name + ".invalid*")))
    if prior_invalid >= 3 and len(batch) > 1:
        return split_range()

    rows: list[dict[str, Any]] | None = None
    evidence_path = result_path
    compact_batch = [
        {"packet_id": packet["packet_id"], "user_prompt": packet["user_prompt"]}
        for packet in batch
    ]
    prompt = (
        "You are an independent anonymous caption evaluator. "
        "Do not use tools, do not read files, do not infer the identities "
        "of caption systems. Evaluate EACH supplied packet independently "
        "using its exact system_prompt and user_prompt. Temperature protocol "
        "requested: 0 (CLI transport does not expose an independent temperature "
        "switch; do not claim enforcement). The output-schema wrapper is only "
        "for transport; give exactly one A/B decision per packet in the supplied "
        "order. No default decisions, no heuristics, no code-generated judgments. "
        "Read and evaluate every pair. Return the JSON object with decisions.\n"
        "Use this exact official HOMER rubric for every item:\n"
        + rubric
        + "\nPackets (evaluate each independently):\n"
        + json.dumps(compact_batch, ensure_ascii=False)
    )
    for attempt in range(1, 4):
        checkpoint = existing_checkpoint()
        if checkpoint is not None:
            return checkpoint
        response_path = result_path.with_name(result_path.name + f".tmp.{os.getpid()}")
        response_path.unlink(missing_ok=True)
        try:
            proc = subprocess.run(
                [
                    args.codex_bin,
                    "exec",
                    "--ephemeral",
                    "-m",
                    args.model,
                    "--skip-git-repo-check",
                    "-C",
                    str(state_dir),
                    "-s",
                    "read-only",
                    "--output-schema",
                    str(schema_path),
                    "-o",
                    str(response_path),
                    "-",
                ],
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=sanitized_environment(),
                timeout=args.call_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            failure_text = (
                f"codex call timed out after {args.call_timeout_seconds}s\n"
                f"stdout={exc.stdout or ''}\n"
                f"stderr={exc.stderr or ''}\n"
            )
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(failure_text)
            failure = classify_codex_failure(failure_text) or {
                "kind": "call_timeout",
                "retryable": True,
                "retry_after_epoch": time.time() + TRANSIENT_RETRY_SECONDS,
                "retry_after": None,
            }
            write_block_marker(
                state_dir,
                model=args.model,
                batch=batch_tag,
                failure=failure,
                message=failure_text,
            )
            response_path.unlink(missing_ok=True)
            raise RunnerBlocked(124, f"BLOCKED {args.model} call_timeout at {batch_tag}")
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n--- attempt {attempt} ---\n")
            log_handle.write(proc.stdout)
            log_handle.write("\n")
            log_handle.write(proc.stderr)
        if proc.returncode != 0:
            response_path.unlink(missing_ok=True)
            failure_text = (proc.stderr or "") + "\n" + (proc.stdout or "")
            failure = classify_codex_failure(failure_text) or {
                "kind": "unclassified_failure",
                "retryable": False,
                "retry_after_epoch": None,
                "retry_after": None,
            }
            write_block_marker(
                state_dir,
                model=args.model,
                batch=batch_tag,
                failure=failure,
                message=failure_text,
            )
            raise RunnerBlocked(
                proc.returncode,
                f"BLOCKED {args.model} {failure['kind']} at {batch_tag}",
            )
        if not response_path.exists():
            failure = {
                "kind": "missing_response",
                "retryable": False,
                "retry_after_epoch": None,
                "retry_after": None,
            }
            write_block_marker(
                state_dir,
                model=args.model,
                batch=batch_tag,
                failure=failure,
                message=f"codex exited successfully but did not create {response_path}",
            )
            raise RunnerBlocked(2, f"missing response file {response_path}")
        os.replace(response_path, result_path)
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            candidate_rows = payload["decisions"]
            validate_batch(candidate_rows, batch, result_path)
            rows = candidate_rows
            evidence_path = result_path
            break
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, RuntimeError, TypeError):
            quarantine_invalid_result(result_path)

    if rows is None:
        if len(batch) > 1:
            return split_range()
        failure = {
            "kind": "invalid_response_exhausted",
            "retryable": False,
            "retry_after_epoch": None,
            "retry_after": None,
        }
        write_block_marker(
            state_dir,
            model=args.model,
            batch=batch_tag,
            failure=failure,
            message=f"three invalid/truncated responses for singleton {result_path}",
        )
        raise RunnerBlocked(2, f"BLOCKED {args.model} invalid_response_exhausted at {batch_tag}")
    return rows, evidence_path


def main() -> int:
    args = parse_args()
    shard_dir = args.root / "multimodel_agent_shards"
    args.state_dir.mkdir(parents=True, exist_ok=True)
    schema_path = args.state_dir / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA, ensure_ascii=False), encoding="utf-8")

    for shard_name in args.shards:
        shard = shard_dir / shard_name
        packet_path = shard / "public_packets.jsonl"
        output_path = shard / args.output_name
        if not packet_path.exists():
            raise FileNotFoundError(packet_path)
        packets = load_jsonl(packet_path)
        if output_path.exists():
            try:
                validate_output(load_jsonl(output_path), packets, model=args.model)
            except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, TypeError):
                # Never let an invalid final-name file block recovery.  Keep
                # it as auditable evidence and reconstruct from checkpoints.
                quarantine_invalid_result(output_path)
                print(f"Quarantined invalid existing output: {output_path}", flush=True)
            else:
                print(f"Existing valid output; skipping: {output_path}", flush=True)
                continue

        all_rows: list[dict[str, Any]] = []
        for start in range(0, len(packets), args.batch_size):
            end = min(start + args.batch_size, len(packets))
            rubric = packets[start]["system_prompt"]
            try:
                rows, evidence_path = run_batch_with_recovery(
                    args=args,
                    schema_path=schema_path,
                    state_dir=args.state_dir,
                    shard_name=shard_name,
                    packets=packets,
                    start=start,
                    end=end,
                    rubric=rubric,
                )
            except RunnerBlocked as exc:
                print(str(exc), flush=True)
                return exc.exit_code
            all_rows.extend(
                {
                    **row,
                    "model": args.model,
                    "temperature": 0,
                    "temperature_protocol": "requested-0-not-exposed",
                    "prompt_sha256": PROMPT_SHA256,
                    "source_provenance_sha256": SOURCE_PROVENANCE_SHA256,
                    "evaluation_transport": "codex-exec-account",
                    "batch_evidence": str(evidence_path),
                }
                for row in rows
            )
            print(
                "Validated",
                shard_name,
                end,
                "/",
                len(packets),
                dict(Counter(row["decision"] for row in rows)),
                flush=True,
            )

        dump_jsonl(output_path, all_rows)
        print(
            "COMPLETE",
            shard_name,
            len(all_rows),
            dict(Counter(row["decision"] for row in all_rows)),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
