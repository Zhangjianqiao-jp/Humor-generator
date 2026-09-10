#!/usr/bin/env python3
"""Run the pinned HOMER GPT-5 pairwise evaluator on public packets.

This command is deliberately separate from packet construction and aggregation:
it reads only the anonymous public prompt JSONL and writes one decision per line.
It never opens the private mapping, never receives an image/model condition, and
never computes Pass@K.  Set ``OPEN_API_KEY`` (the variable used by the released
HOMER evaluator) or ``OPENAI_API_KEY`` before running.  The optional OpenAI
client dependency is imported only after all local protocol checks pass.

The output is append/resume safe.  An interrupted run can be restarted with the
same arguments; existing rows are checked for exact packet IDs, A/B decisions,
model, temperature, and system-prompt hash before new requests are sent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_provenance(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("provenance must be a JSON object")
    evaluator = value.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("provenance.evaluator is required")
    if evaluator.get("model_id") != "gpt-5-chat-latest":
        raise ValueError("HOMER primary runner requires evaluator model gpt-5-chat-latest")
    if evaluator.get("temperature") != 0:
        raise ValueError("HOMER primary runner requires evaluator temperature 0")
    prompt_hash = evaluator.get("prompt_sha256")
    if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
        raise ValueError("provenance.evaluator.prompt_sha256 must be a SHA-256 string")
    return value


def load_packets(path: Path, *, expected_system_prompt_hash: str) -> list[dict[str, Any]]:
    packets = read_jsonl(path)
    if not packets:
        raise ValueError("public packet file is empty")
    seen: set[str] = set()
    for index, packet in enumerate(packets, 1):
        packet_id = packet.get("packet_id")
        system_prompt = packet.get("system_prompt")
        user_prompt = packet.get("user_prompt")
        if not isinstance(packet_id, str) or not packet_id:
            raise ValueError(f"packet {index}: packet_id is required")
        if packet_id in seen:
            raise ValueError(f"duplicate packet_id: {packet_id}")
        seen.add(packet_id)
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError(f"packet {packet_id}: system_prompt is required")
        if sha256_text(system_prompt) != expected_system_prompt_hash:
            raise ValueError(f"packet {packet_id}: system prompt hash does not match provenance")
        if not isinstance(user_prompt, str) or not user_prompt:
            raise ValueError(f"packet {packet_id}: user_prompt is required")
        if packet.get("response_contract") != "exactly one token: A or B":
            raise ValueError(f"packet {packet_id}: response contract is not the HOMER A/B contract")
        if packet.get("protocol") != "homer_official_humorAI_pairwise":
            raise ValueError(f"packet {packet_id}: unexpected protocol")
    return packets


def load_existing(path: Path, *, model_id: str, temperature: int, prompt_hash: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    existing: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        packet_id = row.get("packet_id")
        decision = row.get("decision")
        if not isinstance(packet_id, str) or not packet_id:
            raise ValueError(f"existing judgment row {index}: packet_id is required")
        if packet_id in existing:
            raise ValueError(f"existing judgments contain duplicate packet_id: {packet_id}")
        if decision not in ("A", "B"):
            raise ValueError(f"existing judgment row {index}: decision must be exactly A or B")
        if row.get("model") != model_id:
            raise ValueError(f"existing judgment row {index}: model does not match provenance")
        if row.get("temperature") != temperature:
            raise ValueError(f"existing judgment row {index}: temperature does not match provenance")
        if row.get("prompt_sha256") != prompt_hash:
            raise ValueError(f"existing judgment row {index}: prompt hash does not match provenance")
        existing[packet_id] = row
    return existing


def _client_from_environment() -> Any:
    key = os.environ.get("OPEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("no API request sent: set OPEN_API_KEY or OPENAI_API_KEY")
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("no API request sent: install the pinned openai client in the evaluation environment") from exc
    return OpenAI(api_key=key, base_url="https://api.openai.com/v1")


def judge_one(client: Any, packet: dict[str, Any], *, max_retries: int, retry_base_seconds: float) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=[
                    {"role": "system", "content": packet["system_prompt"]},
                    {"role": "user", "content": packet["user_prompt"]},
                ],
                max_tokens=1,
                temperature=0,
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
    provenance = load_provenance(Path(args.provenance))
    evaluator = provenance["evaluator"]
    model_id = evaluator["model_id"]
    temperature = evaluator["temperature"]
    packets = load_packets(Path(args.packets), expected_system_prompt_hash=evaluator["prompt_sha256"])
    packet_ids = {packet["packet_id"] for packet in packets}
    output = Path(args.output)
    existing = load_existing(output, model_id=model_id, temperature=temperature, prompt_hash=evaluator["prompt_sha256"])
    extra = set(existing) - packet_ids
    if extra:
        raise ValueError(f"existing judgments contain {len(extra)} packet IDs not present in public packets")
    remaining = [packet for packet in packets if packet["packet_id"] not in existing]
    if not remaining:
        return {"status": "already_complete", "packets": len(packets), "judgments": len(existing), "output": str(output)}
    if args.max_requests is not None:
        if args.max_requests < 0:
            raise ValueError("--max-requests must be non-negative")
        if args.max_requests == 0:
            return {"status": "partial", "packets": len(packets), "judgments": len(existing), "new_judgments": 0, "output": str(output)}
        pending = remaining[: args.max_requests]
    else:
        pending = remaining

    client = _client_from_environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    delay = max(0.0, float(args.delay_seconds))
    last_request = [0.0]

    def one(packet: dict[str, Any]) -> dict[str, Any]:
        # A process-wide minimum delay avoids accidental request bursts when
        # several workers share one API key.  It is a throttle, not a protocol
        # parameter; evaluator temperature/model remain fixed above.
        if delay:
            with lock:
                now = time.monotonic()
                wait = delay - (now - last_request[0])
                if wait > 0:
                    time.sleep(wait)
                last_request[0] = time.monotonic()
        decision = judge_one(client, packet, max_retries=args.max_retries, retry_base_seconds=args.retry_base_seconds)
        return {
            "packet_id": packet["packet_id"],
            "decision": decision,
            "model": model_id,
            "temperature": temperature,
            "prompt_sha256": evaluator["prompt_sha256"],
        }

    completed: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = {executor.submit(one, packet): packet for packet in pending}
        for future in as_completed(futures):
            row = future.result()
            completed[row["packet_id"]] = row
            print(json.dumps({"event": "judged", "completed": len(completed), "pending": len(pending) - len(completed), "packet_id": row["packet_id"]}, ensure_ascii=False), flush=True)

    # Append only after all submitted requests in this invocation succeed.  A
    # failed invocation therefore leaves a clean resumable prefix from earlier
    # invocations instead of a partially written, ambiguous batch.
    with output.open("a", encoding="utf-8") as handle:
        for packet in pending:
            row = completed[packet["packet_id"]]
            handle.write(canonical_json(row) + "\n")
    final_count = len(existing) + len(completed)
    return {"status": "complete" if final_count == len(packets) else "partial", "packets": len(packets), "judgments": final_count, "new_judgments": len(completed), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official HOMER GPT-5 A/B evaluator on anonymous public packets")
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
