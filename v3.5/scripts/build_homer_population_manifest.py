#!/usr/bin/env python3
"""Build a source-aware inventory of the public HOMER/HIA evaluation population.

This is an inventory, not a training-data sampler.  It intentionally keeps every
ranking CSV and image visible, records the released evaluator population, and
reports the paper's 365-contest claim separately from the files found locally.
That separation prevents a representative-row bridge pilot from being mistaken
for the HOMER benchmark population.
"""
from __future__ import annotations

from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = "d1334f295cc1a8f8f6dc67ba7e846c5939dddcec"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_digest(ids: list[int]) -> str:
    payload = "\n".join(str(value) for value in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def csv_inventory(path: Path) -> dict[str, Any]:
    rows = 0
    fields: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            if "caption" not in row:
                raise ValueError(f"{path} has no caption column")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "caption_rows": rows,
        "fields": fields,
    }


def image_for(root: Path, contest: int) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = root / f"{contest}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def jsonl_contests(paths: list[Path]) -> dict[str, Any]:
    ids: list[int] = []
    records_by_file: dict[str, int] = {}
    for path in paths:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                contest = int(item["contest_number"])
                ids.append(contest)
                count += 1
        records_by_file[str(path.relative_to(ROOT))] = count
    unique = sorted(set(ids))
    return {
        "records": len(ids),
        "unique_contests": len(unique),
        "contest_ids": unique,
        "contest_ids_sha256": ids_digest(unique),
        "records_by_file": records_by_file,
    }


def evaluator_inventory(path: Path, *, expected_key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    ids = sorted({int(item["contest_number"]) for item in payload})
    missing_key = sum(expected_key not in item for item in payload)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "records": len(payload),
        "unique_contests": len(ids),
        "contest_ids_sha256": ids_digest(ids),
        "missing_expected_group_key": missing_key,
    }


def hia_inventory() -> dict[str, Any]:
    ranking_root = ROOT / "data/external/benchmarks/humor_in_ai/ranking/source"
    image_root = ROOT / "data/external/benchmarks/humor_in_ai/cartoons/source"
    files = sorted(
        (path for path in ranking_root.glob("*.csv") if re.fullmatch(r"\d+", path.stem)),
        key=lambda path: int(path.stem),
    )
    if not files:
        raise FileNotFoundError(ranking_root)
    contests = [int(path.stem) for path in files]
    per_contest: list[dict[str, Any]] = []
    missing_images: list[int] = []
    for path, contest in zip(files, contests):
        image = image_for(image_root, contest)
        entry = {"contest_number": contest, "ranking": csv_inventory(path)}
        if image is None:
            missing_images.append(contest)
        else:
            entry["image"] = {
                "path": str(image.relative_to(ROOT)),
                "sha256": sha256(image),
                "bytes": image.stat().st_size,
            }
        per_contest.append(entry)
    in_paper_range = [contest for contest in contests if 530 <= contest <= 895]
    outside_paper_range = [contest for contest in contests if not 530 <= contest <= 895]
    return {
        "paper_declared_contests": 365,
        "paper_declared_range": [530, 895],
        "local_ranking_contests": len(contests),
        "local_ranking_contest_ids": contests,
        "local_ranking_contest_ids_sha256": ids_digest(contests),
        "local_ranking_contests_in_paper_numeric_range": len(in_paper_range),
        "local_ranking_contests_outside_paper_numeric_range": outside_paper_range,
        "local_ranking_caption_rows": sum(item["ranking"]["caption_rows"] for item in per_contest),
        "missing_images": missing_images,
        "per_contest": per_contest,
    }


def electronic_inventory() -> dict[str, Any]:
    image_root = ROOT / "data/external/benchmarks/electronic_sheep/images/all_contest_images"
    annotation = ROOT / "data/external/benchmarks/electronic_sheep/annotations/all_newyorker_contest_annotations.json"
    images = sorted(
        (path for path in image_root.iterdir() if path.is_file() and path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    annotation_payload = json.loads(annotation.read_text(encoding="utf-8"))
    image_ids = sorted({int(path.stem) for path in images})
    annotation_ids = sorted(int(key) for key in annotation_payload)
    return {
        "paper_declared_contests": 679,
        "local_image_files": len(images),
        "local_image_contest_ids_sha256": ids_digest(image_ids),
        "local_annotation_records": len(annotation_payload),
        "local_annotation_contest_ids_sha256": ids_digest(annotation_ids),
        "standard_description_records": 679,
        "note": "Image and annotation inventories are reported separately from the 679 released standard descriptions.",
    }


def build() -> dict[str, Any]:
    official_root = ROOT / "data/external/homer_official" / OFFICIAL_COMMIT
    descriptions = sorted(
        (official_root / "data/datasets/humorbench/gpt4o_description").glob("*.jsonl")
    )
    if not descriptions:
        raise FileNotFoundError("HOMER standard descriptions are missing")
    hia_descriptions = jsonl_contests(descriptions)
    hia_eval = evaluator_inventory(
        official_root / "evaluation/humorousAI/sample5gt.json", expected_key="1-9"
    )
    es_eval = evaluator_inventory(
        official_root / "evaluation/electronic_sheep/sampled3_evaluate_data.json", expected_key="group A"
    )
    return {
        "schema_version": 1,
        "source": {
            "repository": "https://github.com/Shang-hub/HOMER-Official-Implementation",
            "commit": OFFICIAL_COMMIT,
            "paper": "https://arxiv.org/html/2602.06423",
            "dataset_paper": "https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf",
        },
        "population_status": "inventory_complete_benchmark_allowlist_still_required",
        "humor_in_ai": hia_inventory(),
        "humor_in_ai_standard_descriptions": hia_descriptions,
        "electronic_sheep": electronic_inventory(),
        "official_evaluation_assets": {
            "humor_in_ai": hia_eval,
            "electronic_sheep": es_eval,
            "evaluator_models": {
                "group_overall": "GPT-4-Turbo (HIA benchmark protocol)",
                "group_best_pick": "GPT-4o-vision (HIA benchmark protocol)",
                "homer_primary": "GPT-5 (HOMER protocol)",
            },
        },
        "legacy_bridge_manifest_warning": {
            "path": "data/processed/latent_bridge_v35/manifest.json",
            "policy": "historical_representative_rows_max_3_per_source; not a HOMER population",
            "must_not_be_used_for": ["public_code_exact", "pretrained_7b_bridge", "paper_population_claim"],
        },
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/homer_population_v1.json")
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "hia_ranking_contests": payload["humor_in_ai"]["local_ranking_contests"],
        "hia_caption_rows": payload["humor_in_ai"]["local_ranking_caption_rows"],
        "hia_standard_descriptions": payload["humor_in_ai_standard_descriptions"]["records"],
        "hia_evaluator_records": payload["official_evaluation_assets"]["humor_in_ai"]["records"],
        "es_evaluator_records": payload["official_evaluation_assets"]["electronic_sheep"]["records"],
        "status": payload["population_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
