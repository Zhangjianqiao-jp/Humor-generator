#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/latent_bridge_v3"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    clusters: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        path = DATA / f"{split}.jsonl"
        if digest(path) != manifest["output_sha256"][path.name]:
            raise RuntimeError(f"hash mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        clusters[split] = {row["cluster_id"] for row in rows}
        if len(rows) != manifest["split_rows"][split]:
            raise RuntimeError(f"row count mismatch: {split}")
        if len(clusters[split]) != manifest["split_clusters"][split]:
            raise RuntimeError(f"cluster count mismatch: {split}")
    if clusters["train"] & clusters["validation"] or clusters["train"] & clusters["test"]:
        raise RuntimeError("train leakage")
    if clusters["validation"] & clusters["test"]:
        raise RuntimeError("validation/test leakage")
    print(json.dumps({"status": "pass", "split_clusters": manifest["split_clusters"]}))


if __name__ == "__main__":
    main()
