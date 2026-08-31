#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/latent_bridge_v35"
INVALID_CAPTIONS = {"u", "n", "k", "unk", "unknown", "nan", "none", "null"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    clusters: dict[str, set[str]] = {}
    split_names = (
        "train", "validation", "internal_test",
        "official_hia_unseen_test", "official_hia_seen_diagnostic",
    )
    for split in split_names:
        path = DATA / f"{split}.jsonl"
        if digest(path) != manifest["output_sha256"][path.name]:
            raise RuntimeError(f"hash mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        clusters[split] = {row["cluster_id"] for row in rows}
        if len(rows) != manifest["split_rows"][split]:
            raise RuntimeError(f"row count mismatch: {split}")
        if len(clusters[split]) != manifest["split_clusters"][split]:
            raise RuntimeError(f"cluster count mismatch: {split}")
        invalid = [row["row_id"] for row in rows if str(row["caption"]).strip().casefold() in INVALID_CAPTIONS]
        if invalid:
            raise RuntimeError(f"invalid caption sentinels in {split}: {invalid[:10]}")
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            if clusters[left] & clusters[right]:
                raise RuntimeError(f"cluster leakage: {left}/{right}")
    if len(clusters["official_hia_unseen_test"]) != 24:
        raise RuntimeError(
            "expected 24 adapter-unseen official HIA test clusters, got "
            f"{len(clusters['official_hia_unseen_test'])}"
        )
    if len(clusters["official_hia_seen_diagnostic"]) != 23:
        raise RuntimeError("expected 23 adapter-seen official HIA diagnostic clusters")
    official_rows = []
    for split in ("official_hia_unseen_test", "official_hia_seen_diagnostic"):
        official_rows.extend(
            json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()
            if line.strip()
        )
    if any(row["dataset"] != "humor_in_ai" or row["source_split"] != "test" for row in official_rows):
        raise RuntimeError("official HIA test contains a non-official row")
    if any(value for value in manifest["cluster_overlap"].values()):
        raise RuntimeError("manifest reports non-zero cluster overlap")
    trace_input = DATA / manifest["trace_input_manifest"]
    if digest(trace_input) != manifest["trace_input_manifest_sha256"]:
        raise RuntimeError("trace-input manifest hash mismatch")
    trace_rows = [json.loads(line) for line in trace_input.read_text().splitlines() if line.strip()]
    required = clusters["train"] | clusters["validation"]
    if {row["cluster_id"] for row in trace_rows} != required or len(trace_rows) != len(required):
        raise RuntimeError("trace-input manifest does not exactly cover train+validation clusters")
    print(json.dumps({"status": "pass", "split_clusters": manifest["split_clusters"]}))


if __name__ == "__main__":
    main()
