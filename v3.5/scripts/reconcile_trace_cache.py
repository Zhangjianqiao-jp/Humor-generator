#!/usr/bin/env python3
"""Reconcile an interrupted Planner cache against the current pinned inputs.

Compatible traces are retained after tensor/hash/description validation.
Incompatible or no-longer-required tensors are moved to a timestamped,
recoverable quarantine. The original index and failure report are backed up.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import load_trace, plan_from_record, read_jsonl


def main() -> None:
    dataset = ROOT / "data/processed/latent_bridge_v35"
    cache = ROOT / "data/cache/planner_traces_homer_strict_v35"
    index = cache / "index.jsonl"
    required = {
        row["cluster_id"]: row for row in read_jsonl(dataset / "trace_inputs.jsonl")
    }
    records = read_jsonl(index)
    if len(records) != len({row["cluster_id"] for row in records}):
        raise RuntimeError("refusing to reconcile an index with duplicate cluster records")

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    quarantine = cache / "quarantine" / stamp
    quarantine.mkdir(parents=True, exist_ok=False)
    shutil.copy2(index, quarantine / "index.before.jsonl")
    failures = cache / "failures.json"
    if failures.is_file():
        shutil.copy2(failures, quarantine / "failures.before.json")

    retained, removed = [], []
    for record in records:
        cluster = record["cluster_id"]
        reason = None
        expected = required.get(cluster)
        if expected is None:
            reason = "not_in_current_trace_inputs"
        else:
            try:
                plan = plan_from_record(record["plan"])
                if plan.description.strip() != expected["standard_description"].strip():
                    raise ValueError("standard_description_changed")
                load_trace(ROOT / record["trace_path"], expected_sha256=record["trace_sha256"])
            except Exception as exc:
                reason = f"incompatible:{exc}"
        if reason is None:
            record["split"] = expected["split"]
            retained.append(record)
            continue
        source = ROOT / record["trace_path"]
        destination = quarantine / "states" / source.name
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
        removed.append({"cluster_id": cluster, "reason": reason, "trace": str(destination)})

    temporary = index.with_suffix(".jsonl.reconciled.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained),
        encoding="utf-8",
    )
    os.replace(temporary, index)
    failures.write_text("[]\n")
    available = {row["cluster_id"] for row in retained}
    progress = {
        "requested": len(required), "completed": len(retained), "failed": 0,
        "last_cluster": None,
    }
    (cache / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    report = {
        "schema_version": 1,
        "required": len(required),
        "records_before": len(records),
        "retained": len(retained),
        "removed": len(removed),
        "missing_after": len(set(required) - available),
        "quarantine": str(quarantine.relative_to(ROOT)),
        "removed_records": removed,
    }
    (quarantine / "reconciliation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
