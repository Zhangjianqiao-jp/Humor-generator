#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import nltk


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "artifacts/nltk_data"
MANIFEST = ROOT / "manifests/nltk_assets.json"
REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
PACKAGES = {
    "punkt": "packages/tokenizers/punkt.zip",
    "punkt_tab": "packages/tokenizers/punkt_tab.zip",
    "averaged_perceptron_tagger": "packages/taggers/averaged_perceptron_tagger.zip",
    "averaged_perceptron_tagger_eng": "packages/taggers/averaged_perceptron_tagger_eng.zip",
    "stopwords": "packages/corpora/stopwords.zip",
    "wordnet": "packages/corpora/wordnet.zip",
    "omw-1.4": "packages/corpora/omw-1.4.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    checkout = ROOT / "artifacts/nltk_data_upstream"
    if not (checkout / ".git").is_dir():
        subprocess.run([
            "git", "clone", "--filter=blob:none", "--no-checkout",
            "https://github.com/nltk/nltk_data.git", str(checkout),
        ], check=True)
    subprocess.run(["git", "fetch", "--depth", "1", "origin", REVISION], cwd=checkout, check=True)
    subprocess.run(["git", "sparse-checkout", "init", "--no-cone"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "sparse-checkout", "set", *PACKAGES.values()], cwd=checkout, check=True
    )
    subprocess.run(["git", "checkout", "--detach", REVISION], cwd=checkout, check=True)
    for _package, relative in PACKAGES.items():
        source = checkout / relative
        category = source.parent.name
        archive = DESTINATION / category / source.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(archive.parent)
    archives = sorted(DESTINATION.rglob("*.zip"))
    payload = {
        "schema_version": 1,
        "nltk_version": nltk.__version__,
        "upstream_repository": "https://github.com/nltk/nltk_data.git",
        "upstream_revision": REVISION,
        "packages": list(PACKAGES),
        "archives": {
            str(path.relative_to(DESTINATION)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in archives
        },
        "note": "Extracted runtime files are git-ignored; zip archives are byte-verified.",
    }
    if len(payload["archives"]) != len(PACKAGES):
        raise RuntimeError("one immutable NLTK zip archive is required per package")
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
