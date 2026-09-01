#!/usr/bin/env python3
"""Fetch only the pinned public HOMER artifacts and verify every byte.

The upstream repository has no LICENSE file at the pinned commit. Assets are
therefore kept under git-ignored data/external and must not be redistributed.
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests/homer_official_assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path, expected_hash: str, expected_bytes: int) -> str:
    if destination.is_file() and sha256(destination) == expected_hash:
        return "verified-existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "humor-generator-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if temporary.stat().st_size != expected_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch for {url}")
    actual_hash = sha256(temporary)
    if actual_hash != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {url}: {actual_hash}")
    temporary.replace(destination)
    return "downloaded-and-verified"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source = manifest["source"]
    commit = source["commit"]
    base = f"https://raw.githubusercontent.com/Shang-hub/HOMER-Official-Implementation/{commit}"
    assets = {"joke_corpus": manifest["joke_corpus"], **manifest["standard_descriptions"]}
    results: dict[str, str] = {}
    for name, asset in assets.items():
        results[name] = fetch(
            f"{base}/{asset['upstream_path']}",
            ROOT / asset["local_path"],
            asset["sha256"],
            asset["bytes"],
        )
    print(json.dumps({"source_commit": commit, "assets": results}, indent=2))


if __name__ == "__main__":
    main()
