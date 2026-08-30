#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    dataset = ROOT / "data/processed/latent_bridge_v3"
    cache = ROOT / "data/cache/planner_traces"
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
    report = {
        "required_clusters": len(required),
        "available_clusters": len(available),
        "failure_records": len(failures),
        "missing_clusters": len(missing),
        "extra_clusters": len(extra),
    }
    print(json.dumps(report, indent=2))
    if failures or missing or extra or len(records) != len(available):
        raise SystemExit("formal trace gate failed")


if __name__ == "__main__":
    main()
