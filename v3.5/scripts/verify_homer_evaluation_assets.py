#!/usr/bin/env python3
"""Fail-closed verification for the pinned HOMER evaluation assets.

The HIA paper population (365 contests) and the released HOMER evaluator file
(385 records) are different observable objects.  This checker validates bytes
and schema while refusing to collapse those counts into a single benchmark
number.
"""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(entry: dict[str, Any], errors: list[str], *, root: Path) -> dict[str, Any]:
    path = root / str(entry["local_path"])
    if not path.is_file():
        errors.append(f"missing evaluation asset: {path}")
        return {"path": str(path), "status": "missing"}
    actual_sha = sha256(path)
    actual_bytes = path.stat().st_size
    if actual_sha != entry.get("sha256"):
        errors.append(f"sha256 mismatch: {path}")
    if actual_bytes != entry.get("bytes"):
        errors.append(f"byte-size mismatch: {path}")
    return {"path": str(path), "sha256": actual_sha, "bytes": actual_bytes, "status": "pass"}


def verify(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    assets = payload.get("evaluation_assets", {})
    expected = {
        "hia": (assets.get("humor_in_ai", {}), "1-9", 385),
        "es": (assets.get("electronic_sheep", {}), "group A", 198),
    }
    files: dict[str, Any] = {}
    for name, (entry, key, count) in expected.items():
        if not entry:
            errors.append(f"manifest entry missing: evaluation_assets.{name}")
            continue
        files[name] = check_file(entry, errors, root=ROOT)
        path = ROOT / str(entry.get("local_path", ""))
        if not path.is_file():
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or len(records) != count:
                errors.append(f"{name} evaluator record count is not {count}")
            missing = sum(key not in row for row in records) if isinstance(records, list) else count
            if missing:
                errors.append(f"{name} evaluator has {missing} records without {key!r}")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"cannot parse {name} evaluator: {exc}")

    # Verify the local protocol sources referenced by the provenance manifest.
    # The upstream source hashes identify the pinned public repository; these
    # three local hashes ensure that a later edit cannot silently change the
    # supposedly exact-compatible route.
    local_sources = {
        "local_official_prompts.py": ROOT / "src/humor_generator_v35/homer/official_prompts.py",
        "local_public_code_pipeline.py": ROOT / "src/humor_generator_v35/homer/public_code_pipeline.py",
        "local_homer_retrieval.py": ROOT / "src/humor_generator_v35/homer/retrieval.py",
    }
    source_hashes = payload.get("source_code_hashes", {})
    source_reports: dict[str, Any] = {}
    for name, path in local_sources.items():
        expected_hash = source_hashes.get(name)
        if not path.is_file() or not expected_hash:
            errors.append(f"missing local source/hash: {name}")
            source_reports[name] = {"status": "missing", "path": str(path)}
            continue
        actual_hash = sha256(path)
        source_reports[name] = {"status": "pass" if actual_hash == expected_hash else "fail", "sha256": actual_hash}
        if actual_hash != expected_hash:
            errors.append(f"local source sha256 mismatch: {name}")

    model_contract = assets.get("model_contract", {})
    required_contract = {
        "hia_group_overall": "GPT-4-Turbo with Hessel descriptions",
        "hia_group_best_pick": "GPT-4o-vision with raw cartoon images",
        "homer_primary_humor_evaluator": "GPT-5",
        "homer_primary_humor_evaluator_model_id": "gpt-5-chat-latest",
        "homer_evaluator_temperature": 0.0,
        "homer_candidates_per_image": 5,
        "homer_repeated_trials": 5,
    }
    for key, value in required_contract.items():
        if model_contract.get(key) != value:
            errors.append(f"evaluator model contract mismatch for {key}")
    return {
        "status": "pass" if not errors else "fail",
        "manifest": str(manifest_path),
        "files": files,
        "source_files": source_reports,
        "model_contract": model_contract,
        "errors": errors,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/homer_official_assets.json")
    args = parser.parse_args()
    report = verify(args.manifest.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
