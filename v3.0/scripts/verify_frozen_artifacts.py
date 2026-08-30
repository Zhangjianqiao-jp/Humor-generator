#!/usr/bin/env python3
"""Verify every frozen v3.0 adapter byte against its committed manifest."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    checked: dict[str, list[str]] = {}
    for name, spec in manifest["adapters"].items():
        directory = ROOT / spec["local_path"]
        if not directory.is_dir():
            raise FileNotFoundError(f"missing frozen adapter: {directory}")
        expected_files = spec["files"]
        actual_files = sorted(path.name for path in directory.iterdir() if path.is_file())
        if actual_files != sorted(expected_files):
            raise RuntimeError(f"file-set mismatch for {name}: {actual_files}")
        for filename, expected_hash in expected_files.items():
            actual_hash = digest(directory / filename)
            if actual_hash != expected_hash:
                raise RuntimeError(f"SHA-256 mismatch for {name}/{filename}: {actual_hash}")
        checked[name] = actual_files
    return {"status": "pass", "manifest": str(manifest_path), "checked": checked}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "manifests/frozen_7b_adapters.json"
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), indent=2))


if __name__ == "__main__":
    main()
