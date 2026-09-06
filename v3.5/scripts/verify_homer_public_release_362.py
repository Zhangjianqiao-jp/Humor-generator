#!/usr/bin/env python3
"""Fail-closed verification for the pinned 362-contest public HIA release."""
from __future__ import annotations

from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests/homer_population_public_release_362.json"
EXPECTED_HF_REVISION = "1cd70477b6a99a473690a25a2fed359f75184c64"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_digest(ids: list[int]) -> str:
    return hashlib.sha256("\n".join(str(v) for v in sorted(ids)).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def fail(message: str) -> None:
    raise RuntimeError(message)


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        fail("unexpected public-release manifest schema")
    if manifest.get("data_version") != "homer_pretrained_7b_public_release_362":
        fail("unexpected public-release data version")
    unhashed = dict(manifest)
    recorded_hash = unhashed.pop("manifest_sha256", None)
    recomputed = hashlib.sha256(
        json.dumps(unhashed, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if recorded_hash != recomputed:
        fail("manifest self-hash mismatch")

    allowlist_path = root_path(str(manifest["allowlist"]))
    if not allowlist_path.is_file() or sha256(allowlist_path) != manifest["allowlist_sha256"]:
        fail("allow-list hash mismatch")
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    if allowlist.get("source_revision") != EXPECTED_HF_REVISION:
        fail("allow-list source revision is not the pinned 40-character HF commit")
    source = manifest.get("source", {})
    if source.get("public_dataset_revision") != EXPECTED_HF_REVISION:
        fail("population manifest source revision is not the pinned HF commit")
    ids = [int(value) for value in allowlist.get("contest_ids", [])]
    if len(ids) != 362 or len(set(ids)) != 362:
        fail("allow-list must contain exactly 362 unique contests")
    if id_digest(ids) != manifest["allowlist_contest_ids_sha256"]:
        fail("allow-list ID digest mismatch")
    if id_digest(ids) != "02e6596e8745569fe944a4a1a846a355758fc11230398280c72d7bfc4b931c62":
        fail("allow-list ID set is not the pinned public release")
    if sorted(set(range(530, 896)) - set(ids)) != [605, 644, 657, 890]:
        fail("unexpected missing IDs in the public numeric range")

    for split, info in allowlist["description_files"].items():
        path = root_path(str(info["path"]))
        if not path.is_file() or sha256(path) != info["sha256"]:
            fail(f"description hash mismatch: {split}")
        if len(read_jsonl(path)) != int(info["rows"]):
            fail(f"description row count mismatch: {split}")

    file_entries = {str(item["path"]): item for item in manifest.get("files", [])}
    for relative, entry in file_entries.items():
        path = root_path(relative)
        if not path.is_file() or sha256(path) != entry["sha256"]:
            fail(f"generated file hash mismatch: {relative}")
        if len(read_jsonl(path)) != int(entry["rows"]):
            fail(f"generated file row count mismatch: {relative}")

    pop_path = root_path(manifest["population_rows"])
    trace_input_path = root_path(manifest["trace_input_manifest"])
    population = read_jsonl(pop_path)
    trace_inputs = read_jsonl(trace_input_path)
    if [int(row["contest_number"]) for row in population] != ids:
        fail("population rows are not exactly the sorted allow-list")
    if len(trace_inputs) != len(population):
        fail("trace input count does not equal population count")
    if [int(row["contest_number"]) for row in trace_inputs] != ids:
        fail("trace input ordering does not equal the allow-list")

    expected_splits = {"train": 271, "validation": 44, "test": 47}
    for split, expected_count in expected_splits.items():
        path = root_path(f"data/processed/homer_pretrained_7b_public_release_362/{split}.jsonl")
        rows = read_jsonl(path)
        if len(rows) != expected_count:
            fail(f"{split} row count mismatch")
        if any(row.get("homer_description_split") != split for row in rows):
            fail(f"{split} contains a row from another description split")

    source_path = root_path("data/processed/homer_pretrained_7b_public_release_362/source_rows.jsonl")
    source_rows = read_jsonl(source_path)
    if len(source_rows) != 362 * 3:
        fail("source_rows.jsonl must retain exactly three captions per contest")
    source_counts: dict[int, int] = {}
    for row in source_rows:
        contest = int(row["contest_number"])
        source_counts[contest] = source_counts.get(contest, 0) + 1
    if set(source_counts) != set(ids) or set(source_counts.values()) != {3}:
        fail("source caption population is not exactly three rows per contest")

    seen: set[int] = set()
    for row in population:
        contest = int(row["contest_number"])
        if contest in seen:
            fail(f"duplicate population contest {contest}")
        seen.add(contest)
        image = root_path(str(row["image"]))
        if not image.is_file() or sha256(image) != row["image_sha256"]:
            fail(f"image hash mismatch for contest {contest}")
        ranking = root_path(str(row["ranking_path"]))
        if not ranking.is_file() or sha256(ranking) != row["ranking_sha256"]:
            fail(f"ranking hash mismatch for contest {contest}")
        with ranking.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            captions = [r for r in reader if str(r.get("caption", "")).strip()]
        if len(captions) != int(row["ranking_caption_rows"]):
            fail(f"ranking caption count mismatch for contest {contest}")
        if row.get("source_caption_count") != 3:
            fail(f"source caption count mismatch for contest {contest}")

    if read_jsonl(root_path("data/processed/homer_pretrained_7b_public_release_362/test.jsonl")) != read_jsonl(
        root_path("data/processed/homer_pretrained_7b_public_release_362/official_hia_unseen_test.jsonl")
    ):
        fail("test alias is not byte-equivalent at JSON-record level")
    gate = manifest["formal_route_gate"]
    if gate.get("verified_public_release_allowlist") is not True:
        fail("public release gate is not marked verified")
    if gate.get("verified_365_contest_allowlist") is not False:
        fail("canonical 365 gate must remain false")
    return {
        "status": "pass",
        "data_version": manifest["data_version"],
        "contests": len(ids),
        "split_counts": expected_splits,
        "source_caption_rows": len(source_rows),
        "canonical_365_verified": gate["verified_365_contest_allowlist"],
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
