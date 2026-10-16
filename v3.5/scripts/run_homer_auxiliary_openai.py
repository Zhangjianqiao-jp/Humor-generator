#!/usr/bin/env python3
"""Run an explicitly labelled auxiliary HOMER evaluator on anonymous packets.

The canonical HOMER runner is intentionally pinned to ``gpt-5-chat-latest``.
This command provides the same packet/provenance checks for an auxiliary model
(for example ``gpt-5.5`` or a locally available ``gpt-5.6-*`` variant) without
changing the primary protocol.  It never reads the private packet mapping.

Use one output file per model.  Judgments are append/resume safe and every row
records the requested model, temperature, prompt hash, and source provenance
hash, so auxiliary results cannot silently be mixed with the canonical table.
The OpenAI client and API key are required only when at least one request is
actually sent; ``--max-requests 0`` is a local protocol smoke.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

try:  # Works both as ``python -m scripts...`` and as a direct script.
    from scripts.run_homer_primary_openai import (
        _client_from_environment,
        canonical_json,
        load_packets,
        read_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - direct CLI import path
    from run_homer_primary_openai import (  # type: ignore
        _client_from_environment,
        canonical_json,
        load_packets,
        read_jsonl,
    )


def load_auxiliary_provenance(path: Path) -> tuple[dict[str, Any], str]:
    """Load canonical packet provenance and return it with its file hash."""
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("provenance must be a JSON object")
    evaluator = value.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("provenance.evaluator is required")
    prompt_hash = evaluator.get("prompt_sha256")
    if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
        raise ValueError("provenance.evaluator.prompt_sha256 must be a SHA-256 string")
    if evaluator.get("temperature") != 0:
        raise ValueError("source provenance must use evaluator temperature 0")
    return value, hashlib.sha256(raw).hexdigest()


def load_existing_auxiliary(
    path: Path,
    *,
    model_id: str,
    temperature: int,
    prompt_hash: str,
    source_provenance_sha256: str,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    existing: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        packet_id = row.get("packet_id")
        if not isinstance(packet_id, str) or not packet_id:
            raise ValueError(f"existing judgment row {index}: packet_id is required")
        if packet_id in existing:
            raise ValueError(f"existing judgments contain duplicate packet_id: {packet_id}")
        if row.get("decision") not in ("A", "B"):
            raise ValueError(f"existing judgment row {index}: decision must be exactly A or B")
        if row.get("model") != model_id:
            raise ValueError(f"existing judgment row {index}: model does not match requested auxiliary model")
        if row.get("temperature") != temperature:
            raise ValueError(f"existing judgment row {index}: temperature does not match requested protocol")
        if row.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"existing judgment row {index}: prompt hash does not match source provenance")
        if row.get("source_provenance_sha256") != source_provenance_sha256:
            raise ValueError(f"existing judgment row {index}: source provenance hash does not match")
        existing[packet_id] = row
    return existing


def judge_one(
    client: Any,
    packet: dict[str, Any],
    *,
    model_id: str,
    temperature: int,
    max_retries: int,
    retry_base_seconds: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": packet["system_prompt"]},
                    {"role": "user", "content": packet["user_prompt"]},
                ],
                max_tokens=1,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            decision = content.strip().upper() if isinstance(content, str) else ""
            if decision in ("A", "B"):
                return decision
            last_error = ValueError(f"invalid evaluator response {content!r}")
        except Exception as exc:  # API/network errors are retried, then surfaced.
            last_error = exc
        if attempt + 1 < max_retries:
            time.sleep(retry_base_seconds * (2**attempt))
    raise RuntimeError(f"packet {packet['packet_id']} failed after {max_retries} attempts: {last_error}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_id.strip():
        raise ValueError("--model-id must not be empty")
    provenance, source_hash = load_auxiliary_provenance(Path(args.provenance))
    evaluator = provenance["evaluator"]
    prompt_hash = evaluator["prompt_sha256"]
    packets = load_packets(Path(args.packets), expected_system_prompt_hash=prompt_hash)
    packet_ids = {packet["packet_id"] for packet in packets}
    output = Path(args.output)
    temperature = 0
    existing = load_existing_auxiliary(
        output,
        model_id=args.model_id,
        temperature=temperature,
        prompt_hash=prompt_hash,
        source_provenance_sha256=source_hash,
    )
    extra = set(existing) - packet_ids
    if extra:
        raise ValueError(f"existing judgments contain {len(extra)} packet IDs not present in public packets")
    remaining = [packet for packet in packets if packet["packet_id"] not in existing]
    if not remaining:
        return {"status": "already_complete", "model": args.model_id, "packets": len(packets), "judgments": len(existing), "output": str(output)}
    if args.max_requests is not None:
        if args.max_requests < 0:
            raise ValueError("--max-requests must be non-negative")
        if args.max_requests == 0:
            return {"status": "partial", "model": args.model_id, "packets": len(packets), "judgments": len(existing), "new_judgments": 0, "output": str(output)}
        pending = remaining[: args.max_requests]
    else:
        pending = remaining

    client = _client_from_environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    delay = max(0.0, float(args.delay_seconds))
    last_request = [0.0]

    def one(packet: dict[str, Any]) -> dict[str, Any]:
        if delay:
            with lock:
                now = time.monotonic()
                wait = delay - (now - last_request[0])
                if wait > 0:
                    time.sleep(wait)
                last_request[0] = time.monotonic()
        decision = judge_one(
            client,
            packet,
            model_id=args.model_id,
            temperature=temperature,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
        )
        return {
            "packet_id": packet["packet_id"],
            "decision": decision,
            "model": args.model_id,
            "temperature": temperature,
            "prompt_sha256": prompt_hash,
            "source_provenance_sha256": source_hash,
        }

    completed: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = {executor.submit(one, packet): packet for packet in pending}
        for future in as_completed(futures):
            row = future.result()
            completed[row["packet_id"]] = row
            print(json.dumps({"event": "judged", "model": args.model_id, "completed": len(completed), "pending": len(pending) - len(completed), "packet_id": row["packet_id"]}, ensure_ascii=False), flush=True)

    with output.open("a", encoding="utf-8") as handle:
        for packet in pending:
            handle.write(canonical_json(completed[packet["packet_id"]]) + "\n")
    final_count = len(existing) + len(completed)
    return {"status": "complete" if final_count == len(packets) else "partial", "model": args.model_id, "packets": len(packets), "judgments": final_count, "new_judgments": len(completed), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an auxiliary model on anonymous HOMER A/B packets")
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_retries < 1:
        raise SystemExit("--max-workers and --max-retries must be positive")
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
