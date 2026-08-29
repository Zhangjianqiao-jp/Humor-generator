from __future__ import annotations

import json
from pathlib import Path


def test_homer_dataset_is_cluster_isolated() -> None:
    root = Path("data/processed/homer_latent")
    if not root.exists():
        return
    clusters = {}
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        for row in rows:
            previous = clusters.setdefault(row["cluster_id"], split)
            assert previous == split


def test_homer_manifest_reports_rows_and_independent_images_separately() -> None:
    path = Path("data/processed/homer_latent/manifest.json")
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    assert manifest["actual_rows"] == 1044
    assert manifest["independent_image_clusters"] < manifest["actual_rows"]
    assert manifest["cross_source_duplicate_rows"] > 0
