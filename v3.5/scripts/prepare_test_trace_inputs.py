#!/usr/bin/env python3
"""Create a sealed, cluster-level planner-input manifest for held-out traces."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from humor_generator_v35.data.traces import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/latent_bridge_v35")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: dict[str, dict] = {}
    splits = ("internal_test", "official_hia_unseen_test")
    for split in splits:
        split_rows = read_jsonl(args.dataset / f"{split}.jsonl")
        first_by_cluster: dict[str, dict] = {}
        for row in sorted(split_rows, key=lambda item: str(item["row_id"])):
            first_by_cluster.setdefault(str(row["cluster_id"]), row)
        for row in first_by_cluster.values():
            cluster = str(row["cluster_id"])
            value = {
                "cluster_id": cluster,
                "split": split,
                "image_sha256": str(row["image_sha256"]),
                "standard_description": str(row["standard_description"]),
            }
            previous = rows.get(cluster)
            if previous is not None and previous != value:
                raise RuntimeError(f"inconsistent held-out planner input for {cluster}")
            rows[cluster] = value
    if len(rows) != 121:
        raise RuntimeError(f"expected 121 sealed unseen clusters, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(rows[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for key in sorted(rows)
        )
    )
    print(json.dumps({"status": "pass", "clusters": len(rows), "splits": list(splits)}))


if __name__ == "__main__":
    main()
