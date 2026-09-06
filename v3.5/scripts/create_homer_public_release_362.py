#!/usr/bin/env python3
"""Materialise the exact 362-contest public HIA description release.

The HOMER paper reports a 365-contest HIA population, but neither the paper nor
the released files expose a machine-readable 365-ID allow-list.  The current
public dataset revision *does* expose three description splits.  This script
turns that release into a sealed, source-aware data version without guessing
the missing three contests or editing any upstream bytes.

The generated directory is ignored by git; its hashes and provenance are
recorded in ``manifests/homer_population_public_release_362.json``.  The
result is valid for an ``adapted_public_release_362`` evaluation claim, not a
canonical 365-contest HOMER claim.
"""
from __future__ import annotations

from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HOMER_COMMIT = "d1334f295cc1a8f8f6dc67ba7e846c5939dddcec"
HF_DATASET = "https://huggingface.co/datasets/yguooo/newyorker_caption_ranking"
# This is the 40-character commit returned by the Hugging Face dataset API.
# Keep the full revision: the 39-character look-alike is not a valid HF ref.
HF_REVISION = "1cd70477b6a99a473690a25a2fed359f75184c64"
DESCRIPTION_ROOT = (
    ROOT
    / "data/external/homer_official"
    / HOMER_COMMIT
    / "data/datasets/humorbench/gpt4o_description"
)
PARENT_DIR = ROOT / "data/processed/homer_pretrained_7b_v2"
OUTPUT_DIR = ROOT / "data/processed/homer_pretrained_7b_public_release_362"
ALLOWLIST_PATH = ROOT / "manifests/homer_public_release_362_allowlist.json"
MANIFEST_PATH = ROOT / "manifests/homer_population_public_release_362.json"

EXPECTED_SPLIT_COUNTS = {"train": 271, "validation": 44, "test": 47}
EXPECTED_MISSING_IN_NUMERIC_RANGE = [605, 644, 657, 890]
EXPECTED_ID_DIGEST = "02e6596e8745569fe944a4a1a846a355758fc11230398280c72d7bfc4b931c62"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_digest(ids: list[int]) -> str:
    return sha256_bytes("\n".join(str(value) for value in sorted(ids)).encode("utf-8"))


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_description_splits() -> tuple[dict[int, dict[str, Any]], dict[str, list[int]], dict[str, str]]:
    descriptions: dict[int, dict[str, Any]] = {}
    split_ids: dict[str, list[int]] = {}
    file_hashes: dict[str, str] = {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        path = DESCRIPTION_ROOT / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        file_hashes[split] = sha256_file(path)
        rows = read_jsonl(path)
        if len(rows) != expected_count:
            raise RuntimeError(f"{path}: expected {expected_count} rows, found {len(rows)}")
        ids: list[int] = []
        for item in rows:
            contest = int(item["contest_number"])
            if contest in descriptions:
                raise RuntimeError(f"duplicate description contest {contest}")
            if not str(item.get("canny", "")).strip():
                raise RuntimeError(f"contest {contest} has an empty canny description")
            descriptions[contest] = {"split": split, **item}
            ids.append(contest)
        split_ids[split] = sorted(ids)
    ids = sorted(descriptions)
    if len(ids) != 362 or id_digest(ids) != EXPECTED_ID_DIGEST:
        raise RuntimeError("description ID set does not match the pinned 362-contest release")
    missing = sorted(set(range(530, 896)) - set(ids))
    if missing != EXPECTED_MISSING_IN_NUMERIC_RANGE:
        raise RuntimeError(f"unexpected missing IDs in 530..895: {missing}")
    return descriptions, split_ids, file_hashes


def load_parent_hia_rows() -> dict[int, list[dict[str, Any]]]:
    if not PARENT_DIR.is_dir():
        raise FileNotFoundError(
            f"{PARENT_DIR} is missing; run scripts/create_homer_population_v2.py first"
        )
    by_contest: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(PARENT_DIR.glob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("dataset") != "humor_in_ai":
                continue
            contest = int(row["contest_number"])
            by_contest.setdefault(contest, []).append(dict(row))
    return by_contest


def ranking_inventory(contest: int) -> tuple[Path, int, str]:
    path = ROOT / "data/external/benchmarks/humor_in_ai/ranking/source" / f"{contest}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "caption" not in (reader.fieldnames or []):
            raise RuntimeError(f"{path} has no caption column")
        count = sum(1 for row in reader if str(row.get("caption", "")).strip())
    if count < 1:
        raise RuntimeError(f"{path} contains no non-empty captions")
    return path, count, sha256_file(path)


def make_population_row(
    contest: int,
    description: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(source_rows) != 3:
        raise RuntimeError(f"contest {contest} must have exactly three source caption rows")
    source_rows = sorted(source_rows, key=lambda row: str(row["row_id"]))
    first = dict(source_rows[0])
    for row in source_rows:
        if row.get("standard_description") != description["canny"]:
            raise RuntimeError(f"contest {contest} canny description mismatch")
        if row.get("image") != first.get("image"):
            raise RuntimeError(f"contest {contest} has inconsistent image paths")
    image = Path(str(first["image"])).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    image_root = (ROOT / "data/external/benchmarks/humor_in_ai/cartoons/source").resolve()
    if image.parent != image_root or image.stem != str(contest):
        raise RuntimeError(f"contest {contest} image is not the expected HIA source file: {image}")
    ranking, ranking_rows, ranking_hash = ranking_inventory(contest)

    row = {
        key: value
        for key, value in first.items()
        if key not in {"caption", "caption_rank", "caption_score", "source_split"}
    }
    row.update(
        {
            "row_id": f"humor_in_ai:{contest}:population",
            "cluster_id": f"nycc_{contest}",
            "dataset": "humor_in_ai",
            "contest_number": contest,
            "source_split": "homer_public_release_362",
            "homer_description_split": description["split"],
            "homer_description_source": rel(DESCRIPTION_ROOT / f"{description['split']}.jsonl"),
            "homer_description": {
                "canny": description["canny"],
                "uncanny": description.get("uncanny", ""),
                "location": description.get("location", ""),
                "entities": description.get("entities", []),
            },
            "standard_description": description["canny"],
            "image": rel(image),
            "image_sha256": sha256_file(image),
            "ranking_path": rel(ranking),
            "ranking_sha256": ranking_hash,
            "ranking_caption_rows": ranking_rows,
            "source_caption_count": len(source_rows),
            "source_caption_row_ids": [str(item["row_id"]) for item in source_rows],
            "population_key": ["humor_in_ai", contest, sha256_file(image)],
            "population_version": "homer_pretrained_7b_public_release_362",
        }
    )
    return row


def make_source_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    image = Path(str(result["image"])).resolve()
    result["image"] = rel(image)
    result["image_sha256"] = sha256_file(image)
    result["population_version"] = "homer_pretrained_7b_public_release_362"
    return result


def build() -> dict[str, Any]:
    descriptions, split_ids, description_hashes = load_description_splits()
    parent_rows = load_parent_hia_rows()
    if sorted(parent_rows) != sorted(descriptions):
        missing = sorted(set(descriptions) - set(parent_rows))
        extra = sorted(set(parent_rows) - set(descriptions))
        raise RuntimeError(f"parent/source join mismatch: missing={missing}, extra={extra}")

    population_rows = [
        make_population_row(contest, descriptions[contest], parent_rows[contest])
        for contest in sorted(descriptions)
    ]
    source_rows = [
        make_source_row(row)
        for contest in sorted(parent_rows)
        for row in sorted(parent_rows[contest], key=lambda item: str(item["row_id"]))
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "source_rows.jsonl", source_rows)
    write_jsonl(OUTPUT_DIR / "population_rows.jsonl", population_rows)
    write_jsonl(OUTPUT_DIR / "trace_inputs.jsonl", [
        {
            "cluster_id": row["cluster_id"],
            "contest_number": row["contest_number"],
            "dataset": row["dataset"],
            "split": row["homer_description_split"],
            "image": row["image"],
            "image_sha256": row["image_sha256"],
            "standard_description": row["standard_description"],
            "description_fields": row["homer_description"],
            "population_key": row["population_key"],
            "population_version": row["population_version"],
        }
        for row in population_rows
    ])
    for split in ("train", "validation", "test"):
        write_jsonl(
            OUTPUT_DIR / f"{split}.jsonl",
            [row for row in population_rows if row["homer_description_split"] == split],
        )
    # Existing route tooling historically names the HOMER test split this way;
    # the alias is byte-for-byte deterministic and is explicitly recorded.
    write_jsonl(
        OUTPUT_DIR / "official_hia_unseen_test.jsonl",
        [row for row in population_rows if row["homer_description_split"] == "test"],
    )

    ids = sorted(descriptions)
    allowlist = {
        "schema_version": 1,
        "dataset": "humor_in_ai",
        "source_repository": HF_DATASET,
        "source_revision": HF_REVISION,
        "source_revision_role": "pinned_public_dataset_commit",
        "description_source": rel(DESCRIPTION_ROOT),
        "description_files": {
            split: {
                "path": rel(DESCRIPTION_ROOT / f"{split}.jsonl"),
                "sha256": description_hashes[split],
                "rows": len(split_ids[split]),
            }
            for split in ("train", "validation", "test")
        },
        "contest_ids": ids,
        "contest_count": len(ids),
        "contest_ids_sha256": id_digest(ids),
        "numeric_range": [530, 895],
        "missing_ids_in_numeric_range": EXPECTED_MISSING_IN_NUMERIC_RANGE,
        "selection_policy": "exact_union_of_public_gpt4o_description_splits_at_pinned_revision",
        "canonical_365_allowlist": {
            "verified": False,
            "reason": "No machine-readable 365-ID allow-list is published in the pinned sources.",
        },
    }
    write_json(ALLOWLIST_PATH, allowlist)

    files: list[dict[str, Any]] = []
    for path in sorted(OUTPUT_DIR.glob("*.jsonl")):
        files.append(
            {
                "path": rel(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(read_jsonl(path)),
            }
        )
    image_repairs = []
    population_by_contest = {int(row["contest_number"]): row for row in population_rows}
    for contest in ids:
        parent_hashes = {str(row.get("image_sha256", "")) for row in parent_rows[contest]}
        current_hash = population_by_contest[contest]["image_sha256"]
        if parent_hashes != {current_hash}:
            image_repairs.append(
                {
                    "contest_number": contest,
                    "parent_hashes": sorted(parent_hashes),
                    "current_hash": current_hash,
                }
            )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "data_version": "homer_pretrained_7b_public_release_362",
        "population_status": "verified_public_release_362",
        "source": {
            "public_dataset": HF_DATASET,
            "public_dataset_revision": HF_REVISION,
            "homer_repository": "https://github.com/Shang-hub/HOMER-Official-Implementation",
            "homer_commit": HOMER_COMMIT,
            "homer_paper": "https://arxiv.org/html/2602.06423",
            "parent_data_version": "homer_pretrained_7b_v2",
            "parent_manifest": "manifests/homer_population_v2.json",
        },
        "population_key": ["dataset", "contest_number", "image_sha256"],
        "population_rows": rel(OUTPUT_DIR / "population_rows.jsonl"),
        "trace_input_manifest": rel(OUTPUT_DIR / "trace_inputs.jsonl"),
        "trace_index": None,
        "allowlist": rel(ALLOWLIST_PATH),
        "allowlist_sha256": sha256_file(ALLOWLIST_PATH),
        "allowlist_contest_count": len(ids),
        "allowlist_contest_ids_sha256": id_digest(ids),
        "split_counts": {split: len(split_ids[split]) for split in ("train", "validation", "test")},
        "source_caption_rows": len(source_rows),
        "source_caption_rows_policy": "all three parent rows per allow-listed contest are retained in source_rows.jsonl",
        "image_byte_policy": "current_local_source_bytes; no upstream image bytes were edited",
        "image_repairs_relative_to_parent": image_repairs,
        "known_upstream_data_issues": {
            "reference": "https://github.com/nextml/caption-contest-data-api/issues/38",
            "policy": "published bytes are retained and any known issue is provenance, not silently repaired content",
        },
        "formal_route_gate": {
            "verified_public_release_allowlist": True,
            "verified_public_release_count": 362,
            "verified_365_contest_allowlist": False,
            "canonical_365_status": "unresolved",
            "population_rows": rel(OUTPUT_DIR / "population_rows.jsonl"),
            "trace_index": None,
            "trace_status": "pending_current_pretrained_planner_traces",
            "reason": "The public machine-readable release is 362 contests; the paper-level 365 claim remains a separate unresolved population.",
        },
        "files": files,
        "rows_total": len(population_rows),
        "manifest_provenance": {
            "description_ids_digest": EXPECTED_ID_DIGEST,
            "description_file_hashes": description_hashes,
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    )
    write_json(MANIFEST_PATH, manifest)
    write_json(OUTPUT_DIR / "manifest.json", manifest)
    return manifest


def main() -> None:
    global OUTPUT_DIR, MANIFEST_PATH, ALLOWLIST_PATH
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--allowlist", type=Path, default=ALLOWLIST_PATH)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    MANIFEST_PATH = args.manifest.resolve()
    ALLOWLIST_PATH = args.allowlist.resolve()
    result = build()
    print(json.dumps({
        "status": "pass",
        "data_version": result["data_version"],
        "population_status": result["population_status"],
        "contests": result["allowlist_contest_count"],
        "split_counts": result["split_counts"],
        "source_caption_rows": result["source_caption_rows"],
        "manifest": rel(MANIFEST_PATH),
        "allowlist": rel(ALLOWLIST_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
