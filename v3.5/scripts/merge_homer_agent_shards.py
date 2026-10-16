#!/usr/bin/env python3
"""Merge anonymous collaboration-agent HOMER judgment shards.

This is a resumable assembly step for the no-API-key auxiliary track. Each
shard contains a contiguous slice of the *public* packet file and one decision
file per evaluator. The command verifies that shard packet IDs exactly match
the canonical public file, rejects overlaps/duplicates/provenance drift, and
emits a partial or complete per-model JSONL in canonical packet order. It does
not read the private mapping and it never computes quality metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

try:
    from .aggregate_homer_primary_judgments import read_jsonl
except ImportError:  # pragma: no cover
    from aggregate_homer_primary_judgments import read_jsonl  # type: ignore


PROMPT_SHA256 = "98ebb56261262dd77112094330b056417ca133e2adf02ab5b6da60afdc31c524"
SOURCE_PROVENANCE_SHA256 = "44ea9297984ed1040e676a2589706aadc46b89dfa9b5650954a1cd7bec3434b1"


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "model"


def load_public_packets(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ids = [row.get("packet_id") for row in rows]
    if not rows or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("canonical public packets must be non-empty with unique packet IDs")
    return rows


def verify_decision_rows(rows: list[dict[str, Any]], *, model_id: str, expected_ids: list[str]) -> None:
    if len(rows) != len(expected_ids):
        raise ValueError(f"{model_id}: decision count {len(rows)} does not match shard count {len(expected_ids)}")
    for index, (row, expected_id) in enumerate(zip(rows, expected_ids), 1):
        if row.get("packet_id") != expected_id:
            raise ValueError(f"{model_id}: shard decision row {index} packet ID/order mismatch")
        if row.get("decision") not in ("A", "B"):
            raise ValueError(f"{model_id}: shard decision row {index} is not A/B")
        if row.get("model") != model_id:
            raise ValueError(f"{model_id}: shard decision row {index} has wrong model")
        if row.get("temperature") != 0:
            raise ValueError(f"{model_id}: shard decision row {index} temperature is not 0")
        if row.get("prompt_sha256") != PROMPT_SHA256:
            raise ValueError(f"{model_id}: shard decision row {index} prompt hash mismatch")
        if row.get("source_provenance_sha256") != SOURCE_PROVENANCE_SHA256:
            raise ValueError(f"{model_id}: shard decision row {index} source provenance mismatch")


def discover_shards(shard_dir: Path) -> list[Path]:
    shards = sorted(path for path in shard_dir.glob("shard_*") if path.is_dir())
    if not shards:
        raise ValueError(f"no shard directories found under {shard_dir}")
    return shards


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = []
    for spec in args.models:
        if "=" not in spec:
            raise ValueError(f"--models expects MODEL_ID=FILENAME, got {spec!r}")
        model, filename = spec.split("=", 1)
        model = model.strip()
        filename = filename.strip()
        if not model or not filename or model in models:
            raise ValueError(f"invalid or duplicate model specification: {spec!r}")
        models.append(model)
    public = load_public_packets(Path(args.packets))
    full_ids = [row["packet_id"] for row in public]
    index_by_id = {packet_id: index for index, packet_id in enumerate(full_ids)}
    shards = discover_shards(Path(args.shard_dir))
    collected: dict[str, dict[int, dict[str, Any]]] = {model: {} for model in models}
    shard_reports: list[dict[str, Any]] = []
    for shard in shards:
        packet_path = shard / "public_packets.jsonl"
        if not packet_path.exists():
            raise ValueError(f"{shard}: public_packets.jsonl is missing")
        packet_rows = load_public_packets(packet_path)
        positions = [index_by_id.get(row["packet_id"]) for row in packet_rows]
        if any(position is None for position in positions):
            raise ValueError(f"{shard}: packet ID not present in canonical public packets")
        int_positions = [int(position) for position in positions]
        if int_positions != list(range(int_positions[0], int_positions[-1] + 1)):
            raise ValueError(f"{shard}: shard packet IDs are not a contiguous canonical slice")
        report = {"shard": shard.name, "start_index": int_positions[0], "end_index": int_positions[-1], "rows": len(packet_rows), "models": {}}
        expected_ids = [row["packet_id"] for row in packet_rows]
        for model, filename in [(spec.split("=", 1)[0].strip(), spec.split("=", 1)[1].strip()) for spec in args.models]:
            decision_path = shard / filename
            if not decision_path.exists():
                continue
            rows = read_jsonl(decision_path)
            verify_decision_rows(rows, model_id=model, expected_ids=expected_ids)
            duplicate = set(int_positions) & set(collected[model])
            if duplicate:
                raise ValueError(f"{model}: overlapping shard positions, first={min(duplicate)}")
            for position, row in zip(int_positions, rows):
                collected[model][position] = row
            report["models"][model] = len(rows)
        shard_reports.append(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    progress_models: dict[str, Any] = {}
    complete = True
    for spec in args.models:
        model, filename = spec.split("=", 1)
        model = model.strip()
        rows_by_position = collected[model]
        missing = len(full_ids) - len(rows_by_position)
        if missing:
            complete = False
        output_path = output_dir / filename
        with output_path.open("w", encoding="utf-8") as handle:
            for position in sorted(rows_by_position):
                handle.write(json.dumps(rows_by_position[position], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        outputs[model] = str(output_path)
        progress_models[model] = {"rows": len(rows_by_position), "missing": missing, "complete": missing == 0}
    if args.require_complete and not complete:
        raise ValueError("shard merge is incomplete; rerun without --require-complete only for progress inspection")
    report = {
        "schema_version": 1,
        "status": "complete" if complete else "partial",
        "canonical_packets": len(full_ids),
        "models": progress_models,
        "shards": shard_reports,
        "outputs": outputs,
        "quality_metrics": "not_computed",
    }
    report_path = output_dir / "merge_progress.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": report["status"], "canonical_packets": len(full_ids), "models": progress_models, "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge anonymous HOMER judgment shards")
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--models", required=True, action="append", metavar="MODEL_ID=FILENAME")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
