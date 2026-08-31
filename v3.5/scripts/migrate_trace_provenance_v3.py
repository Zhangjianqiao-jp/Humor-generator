#!/usr/bin/env python3
"""Atomically migrate verified schema-v2 Planner traces to input-scoped v3 provenance."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import load_trace, plan_from_record, read_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit() -> str:
    repo = ROOT.parent
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "v3.5"], cwd=repo,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("migration requires a committed, clean v3.5 tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "data/processed/latent_bridge_v35",
    )
    parser.add_argument(
        "--cache", type=Path,
        default=ROOT / "data/cache/planner_traces_homer_strict_v35",
    )
    parser.add_argument("--expected-records", type=int, default=666)
    args = parser.parse_args()
    index = args.cache / "index.jsonl"
    rows = read_jsonl(index)
    if len(rows) != args.expected_records:
        raise RuntimeError(
            f"migration requires {args.expected_records} complete records, found {len(rows)}"
        )
    if len({row["cluster_id"] for row in rows}) != len(rows):
        raise RuntimeError("trace index contains duplicate clusters")

    trace_inputs = {
        row["cluster_id"]: row for row in read_jsonl(args.dataset / "trace_inputs.jsonl")
    }
    if set(trace_inputs) != {row["cluster_id"] for row in rows}:
        raise RuntimeError("trace index and pinned trace-input manifest have different coverage")

    old_provenance: set[str] = set()
    generation_commits: set[str] = set()
    migrated: list[dict[str, Any]] = []
    for row in rows:
        expected = trace_inputs[row["cluster_id"]]
        if row["split"] != expected["split"]:
            raise RuntimeError(f"split mismatch: {row['cluster_id']}")
        plan = plan_from_record(row["plan"])
        if plan.description.strip() != expected["standard_description"].strip():
            raise RuntimeError(f"description mismatch: {row['cluster_id']}")
        load_trace(ROOT / row["trace_path"], expected_sha256=row["trace_sha256"])
        old_provenance.add(json.dumps(row.get("provenance", {}), sort_keys=True))
        generation_commit = row.get("provenance", {}).get("git_commit")
        if not isinstance(generation_commit, str) or len(generation_commit) != 40:
            raise RuntimeError(f"missing generation commit: {row['cluster_id']}")
        generation_commits.add(generation_commit)
        migrated.append(dict(row))
    if len(generation_commits) != 1:
        raise RuntimeError(f"migration requires one generation commit, found {generation_commits}")

    migration_commit = repository_commit()
    provenance = {
        # Keep the commit that actually generated the tensors. The migration
        # commit is recorded separately in the audit report below.
        "git_commit": next(iter(generation_commits)),
        "trace_input_manifest_sha256": sha256(args.dataset / "trace_inputs.jsonl"),
        "homer_prompts_sha256": sha256(ROOT / "src/humor_generator_v35/homer/prompts.py"),
        "adapter_manifest_sha256": sha256(ROOT / "manifests/frozen_7b_adapters.json"),
    }
    for row in migrated:
        row["schema_version"] = 3
        row["provenance"] = provenance

    before = sha256(index)
    temporary = index.with_suffix(".jsonl.v3.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in migrated),
        encoding="utf-8",
    )
    os.replace(temporary, index)
    report = {
        "schema_version": 1,
        "records": len(migrated),
        "validation": [
            "exact cluster coverage", "split equality", "description equality",
            "trace SHA-256 and tensor load", "strict HOMER plan parsing",
        ],
        "old_index_sha256": before,
        "new_index_sha256": sha256(index),
        "old_provenance_values": [json.loads(value) for value in sorted(old_provenance)],
        "new_provenance": provenance,
        "migration_git_commit": migration_commit,
    }
    (args.cache / "provenance_migration_v3.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
