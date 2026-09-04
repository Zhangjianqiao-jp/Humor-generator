#!/usr/bin/env python3
"""Validate the sealed held-out Planner trace cache before outer evaluation."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import load_trace, plan_from_record, read_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/latent_bridge_v35")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_manifest = args.input_manifest.resolve()
    cache = args.cache.resolve()
    expected = {str(item["cluster_id"]): item for item in read_jsonl(input_manifest)}
    records = read_jsonl(cache / "index.jsonl")
    failures_path = cache / "failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.is_file() else []
    expected_hash = sha256(input_manifest)
    prompt_hash = sha256(ROOT / "src/humor_generator_v35/homer/prompts.py")
    adapter_hash = sha256(ROOT / "manifests/frozen_7b_adapters.json")
    errors: list[str] = []
    seen: set[str] = set()
    passed = 0
    for record in records:
        cluster = str(record.get("cluster_id"))
        try:
            if cluster in seen:
                raise ValueError("duplicate cluster record")
            seen.add(cluster)
            if cluster not in expected:
                raise ValueError("cluster is not in sealed input manifest")
            if record.get("schema_version") != 3:
                raise ValueError("trace schema must be version 3")
            if record.get("split") not in {"internal_test", "official_hia_unseen_test"}:
                raise ValueError(f"unexpected split {record.get('split')!r}")
            provenance = record.get("provenance", {})
            if set(provenance) != {
                "git_commit", "trace_input_manifest_sha256", "homer_prompts_sha256",
                "adapter_manifest_sha256",
            }:
                raise ValueError("trace provenance schema mismatch")
            if provenance["trace_input_manifest_sha256"] != expected_hash:
                raise ValueError("trace input manifest hash mismatch")
            if provenance["homer_prompts_sha256"] != prompt_hash:
                raise ValueError("HOMER prompt hash mismatch")
            if provenance["adapter_manifest_sha256"] != adapter_hash:
                raise ValueError("Planner adapter manifest hash mismatch")
            if not re.fullmatch(r"[0-9a-f]{40}", str(provenance["git_commit"])):
                raise ValueError("invalid trace Git commit")
            expected_row = expected[cluster]
            if str(record.get("split")) != str(expected_row.get("split")):
                raise ValueError("trace split differs from sealed input")
            plan_from_record(record["plan"])
            trace_path = (ROOT / record["trace_path"]).resolve(strict=True)
            if ROOT not in trace_path.parents:
                raise ValueError("trace path escapes v3.5 root")
            loaded = load_trace(trace_path, expected_sha256=str(record["trace_sha256"]))
            if any(not torch.isfinite(item.states).all() for item in loaded.values()):
                raise ValueError("trace contains non-finite hidden states")
            if set(record.get("alignment", {})) != {"conflict", "local", "global"}:
                raise ValueError("trace lacks channel alignment evidence")
            passed += 1
        except Exception as exc:
            errors.append(f"{cluster}: {exc}")
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"missing {len(missing)} sealed traces: {missing[:5]}")
    if failures:
        errors.append(f"cache reports {len(failures)} generation failures")
    report = {
        "status": "pass" if not errors else "fail",
        "expected_clusters": len(expected),
        "records": len(records),
        "passed_records": passed,
        "missing_clusters": len(missing),
        "failure_records": len(failures),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": expected_hash,
        "cache": str(cache),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
