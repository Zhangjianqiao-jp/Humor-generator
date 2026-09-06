#!/usr/bin/env python3
"""Create an immutable, source-aware HOMER population data version.

This command never edits ``data/processed/latent_bridge_v35`` in place.  It
copies its JSONL outputs to a new version and repairs only image hashes that no
longer match the bytes currently present at the recorded image path.  The
repair is recorded as lineage in the new manifest, so an old trace cannot be
silently presented as a canonical HOMER run.
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/processed/latent_bridge_v35"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/homer_pretrained_7b_v2"
DEFAULT_MANIFEST = ROOT / "manifests/homer_population_v2.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: JSONL row must be an object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build(*, source_dir: Path, output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    old_manifest_path = source_dir / "manifest.json"
    old_population_path = ROOT / "manifests/homer_population_v1.json"
    if not old_manifest_path.is_file() or not old_population_path.is_file():
        raise FileNotFoundError("source manifest or homer_population_v1.json is missing")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a reliable cluster -> current image hash index from source rows.
    cluster_images: dict[tuple[str, str], tuple[str, str]] = {}
    historical_hashes: dict[str, set[str]] = {}
    for source_file in sorted(source_dir.glob("*.jsonl")):
        for row in read_jsonl(source_file):
            cluster = row.get("cluster_id")
            image_value = row.get("image")
            if not cluster or not image_value:
                continue
            image = Path(str(image_value))
            if not image.is_file():
                raise FileNotFoundError(f"recorded image is missing: {image}")
            current_hash = sha256_file(image)
            source_key = (str(row.get("dataset", "")), str(cluster))
            previous = cluster_images.get(source_key)
            pair = (str(image), current_hash)
            if previous is not None and previous != pair:
                raise ValueError(f"cluster has inconsistent image paths or bytes: {source_key}")
            cluster_images[source_key] = pair
            historical_hashes.setdefault(str(row.get("image_sha256", "")), set()).add(current_hash)

    repairs: list[dict[str, Any]] = []
    repair_index: dict[tuple[str, str], dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    total_rows = 0
    for source_file in sorted(source_dir.glob("*.jsonl")):
        rows = read_jsonl(source_file)
        repaired_rows: list[dict[str, Any]] = []
        for row in rows:
            updated = dict(row)
            old_hash = updated.get("image_sha256")
            image_path = updated.get("image")
            if image_path:
                image = Path(str(image_path))
                if not image.is_file():
                    raise FileNotFoundError(f"recorded image is missing: {image}")
                new_hash = sha256_file(image)
            else:
                old_hash_text = str(updated.get("image_sha256", ""))
                candidates = historical_hashes.get(old_hash_text, set())
                if len(candidates) == 1:
                    new_hash = next(iter(candidates))
                else:
                    cluster = str(updated.get("cluster_id", ""))
                    cluster_candidates = {
                        new_hash_value
                        for (dataset, cluster_id), (_, new_hash_value) in cluster_images.items()
                        if cluster_id == cluster
                    }
                    if len(cluster_candidates) != 1:
                        raise ValueError(
                            f"cannot resolve trace image uniquely: cluster={cluster}, "
                            f"historical_hash={old_hash_text}"
                        )
                    new_hash = next(iter(cluster_candidates))
            if isinstance(old_hash, str) and old_hash != new_hash:
                key = (old_hash, new_hash)
                entry = repair_index.get(key)
                if entry is None:
                    entry = {
                        "old_sha256": old_hash,
                        "new_sha256": new_hash,
                        "reason": "current_disk_bytes_are_canonical_for_v2; historical row hash differs",
                        "image_paths": [],
                        "source_files": [],
                        "row_ids": [],
                        "rows_updated": 0,
                    }
                    repair_index[key] = entry
                    repairs.append(entry)
                if image_path and str(image_path) not in entry["image_paths"]:
                    entry["image_paths"].append(str(image_path))
                source_name = relative_path(source_file)
                if source_name not in entry["source_files"]:
                    entry["source_files"].append(source_name)
                if updated.get("row_id") and updated["row_id"] not in entry["row_ids"]:
                    entry["row_ids"].append(updated["row_id"])
                entry["rows_updated"] += 1
                updated["image_sha256"] = new_hash
            repaired_rows.append(updated)
        destination = output_dir / source_file.name
        write_jsonl(destination, repaired_rows)
        total_rows += len(repaired_rows)
        files.append({
            "path": relative_path(destination),
            "source_path": relative_path(source_file),
            "rows": len(repaired_rows),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        })

    old_population = json.loads(old_population_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 2,
        "data_version": "homer_pretrained_7b_v2",
        "population_status": "inventory_complete_benchmark_allowlist_still_required",
        "source": {
            "parent_population_manifest": relative_path(old_population_path),
            "parent_population_manifest_sha256": sha256_file(old_population_path),
            "parent_data_dir": relative_path(source_dir),
            "parent_data_manifest_sha256": sha256_file(old_manifest_path),
            "homer_repository": old_population.get("source", {}).get("repository"),
            "homer_commit": old_population.get("source", {}).get("commit"),
            "paper": old_population.get("source", {}).get("paper"),
        },
        "image_byte_policy": "current_disk_bytes_canonical_for_v2",
        "population_key": ["dataset", "contest_number", "image_sha256"],
        "repair_policy": "copy_rows_and_repair_hash_only; no caption, description, split, or ordering edits",
        "image_repairs": repairs,
        "files": files,
        "rows_total": total_rows,
        "clusters_indexed": len(cluster_images),
        "formal_route_gate": {
            "verified_365_contest_allowlist": False,
            "population_rows": None,
            "trace_index": None,
            "reason": "v2 fixes byte provenance but does not invent the unpublished allow-list",
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    )
    write_json(manifest_path, manifest)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = build(
        source_dir=args.source_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        manifest_path=args.manifest.resolve(),
    )
    print(json.dumps({
        "manifest": relative_path(args.manifest),
        "output_dir": relative_path(args.output_dir),
        "rows_total": result["rows_total"],
        "clusters_indexed": result["clusters_indexed"],
        "repairs": result["image_repairs"],
        "population_status": result["population_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
