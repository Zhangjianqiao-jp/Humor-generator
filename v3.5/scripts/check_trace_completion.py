#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re

from humor_generator_v35.data.traces import load_trace, plan_from_record


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    dataset = ROOT / "data/processed/latent_bridge_v35"
    cache = ROOT / "data/cache/planner_traces_homer_strict_v35"
    required = {
        row["cluster_id"]
        for split in ("train", "validation")
        for row in read_jsonl(dataset / f"{split}.jsonl")
    }
    records = read_jsonl(cache / "index.jsonl")
    available = {record["cluster_id"] for record in records}
    failures = json.loads((cache / "failures.json").read_text())
    missing = sorted(required - available)
    extra = sorted(available - required)
    invalid = []
    provenance_values: set[tuple[str, str, str, str]] = set()
    for record in records:
        try:
            if record.get("schema_version") != 2:
                raise ValueError("formal trace must use provenance schema version 2")
            provenance = record.get("provenance", {})
            required_provenance = {
                "git_commit", "dataset_manifest_sha256", "homer_prompts_sha256",
                "adapter_manifest_sha256",
            }
            if set(provenance) != required_provenance:
                raise ValueError("missing or extra trace provenance fields")
            if not re.fullmatch(r"[0-9a-f]{40}", provenance["git_commit"]):
                raise ValueError("invalid trace Git commit")
            for key in required_provenance - {"git_commit"}:
                if not re.fullmatch(r"[0-9a-f]{64}", provenance[key]):
                    raise ValueError(f"invalid provenance hash: {key}")
            provenance_values.add(tuple(provenance[key] for key in sorted(required_provenance)))
            plan_from_record(record["plan"])
            load_trace(ROOT / record["trace_path"], expected_sha256=record["trace_sha256"])
            if record.get("split") not in {"train", "validation"}:
                raise ValueError(f"invalid split {record.get('split')!r}")
            if set(record.get("alignment", {})) != {"conflict", "local", "global"}:
                raise ValueError("missing channel alignment evidence")
            for channel, evidence in record["alignment"].items():
                replay = evidence.get("replay", {})
                if replay.get("mean_cosine", 0) < 0.98 or replay.get("min_cosine", 0) < 0.90:
                    raise ValueError(f"failed causal replay evidence for {channel}")
                if evidence.get("sampling_mode") not in {"greedy", "sample"}:
                    raise ValueError(f"missing sampling mode for {channel}")
                if evidence.get("communication_state_definition") != "teacher_forced_post_token":
                    raise ValueError(f"invalid communication state definition for {channel}")
        except Exception as exc:
            invalid.append({"cluster_id": record.get("cluster_id"), "error": str(exc)})
    report = {
        "required_clusters": len(required),
        "available_clusters": len(available),
        "failure_records": len(failures),
        "missing_clusters": len(missing),
        "extra_clusters": len(extra),
        "invalid_trace_records": len(invalid),
        "unique_provenance_sets": len(provenance_values),
    }
    print(json.dumps(report, indent=2))
    if failures or missing or extra or invalid or len(records) != len(available) or len(provenance_values) != 1:
        raise SystemExit("formal trace gate failed")


if __name__ == "__main__":
    main()
