#!/usr/bin/env python3
"""Verify the v2 copy and its byte-level repair lineage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/homer_population_v2.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise RuntimeError("unexpected v2 manifest schema")
    if manifest.get("data_version") != "homer_pretrained_7b_v2":
        raise RuntimeError("unexpected data version")
    recorded_manifest_hash = manifest.get("manifest_sha256")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("manifest_sha256", None)
    recomputed_manifest_hash = hashlib.sha256(
        json.dumps(unhashed_manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if recorded_manifest_hash != recomputed_manifest_hash:
        raise RuntimeError("v2 manifest self-hash mismatch")
    repaired = manifest.get("image_repairs", [])
    for repair in repaired:
        image_paths = repair.get("image_paths", [])
        if len(image_paths) != 1:
            raise RuntimeError("each recorded repair must identify one image path")
        image = Path(image_paths[0])
        if not image.is_file() or sha256(image) != repair["new_sha256"]:
            raise RuntimeError(f"repaired image bytes do not match new hash: {image}")
    rows = 0
    for entry in manifest.get("files", []):
        path = ROOT / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"v2 output hash mismatch: {path}")
        count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if count != entry["rows"]:
            raise RuntimeError(f"v2 output row count mismatch: {path}")
        rows += count
    if rows != manifest.get("rows_total"):
        raise RuntimeError("v2 total row count mismatch")
    print(json.dumps({
        "status": "pass",
        "data_version": manifest["data_version"],
        "rows_total": rows,
        "repairs": len(repaired),
        "allowlist_gate": manifest["formal_route_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
