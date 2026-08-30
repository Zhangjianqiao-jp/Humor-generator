"""Build image-clustered bridge data from pinned public benchmark artifacts."""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetBuildResult:
    manifest: dict[str, Any]
    split_rows: dict[str, list[dict[str, Any]]]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ranked_captions(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row.get("rank") or 10**9))
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        caption = " ".join(str(row.get("caption") or "").split())
        if not caption or caption.casefold() in seen:
            continue
        seen.add(caption.casefold())
        output.append({
            "caption": caption,
            "caption_rank": int(row.get("rank") or len(output)),
            "caption_score": float(row.get("mean") or 0.0),
        })
        if len(output) == limit:
            break
    return output


def _finalists(annotation: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, raw in enumerate(annotation.get("official_newyorker_finalists") or []):
        caption = " ".join(str(raw).strip(" \t\r\n\"“”").split())
        if not caption or caption.casefold() in seen:
            continue
        seen.add(caption.casefold())
        output.append({"caption": caption, "caption_rank": rank, "caption_score": None})
        if len(output) == limit:
            break
    return output


def _description_text(record: dict[str, Any]) -> str:
    # Preserve HOMER's released standard description verbatim.  Other fields
    # remain available as metadata but are not silently merged into the text.
    value = " ".join(str(record.get("canny") or "").split())
    if not value:
        raise ValueError(f"empty standard description: {record}")
    return value


def _row_variants(
    *,
    dataset: str,
    contest: int,
    image: Path,
    description: dict[str, Any],
    captions: Iterable[dict[str, Any]],
    source_split: str,
) -> list[dict[str, Any]]:
    if not image.is_file():
        raise FileNotFoundError(image)
    image_hash = sha256(image)
    result = []
    for caption in captions:
        result.append({
            "row_id": f"{dataset}:{contest}:{caption['caption_rank']}",
            "cluster_id": f"nycc_{contest}",
            "contest_number": contest,
            "dataset": dataset,
            "source_split": source_split,
            "image": str(image),
            "image_sha256": image_hash,
            "standard_description": _description_text(description),
            "description_fields": {
                key: description[key]
                for key in ("uncanny", "location", "entities")
                if key in description
            },
            **caption,
        })
    return result


def build_clustered_bridge_dataset(
    *,
    official_description_root: Path,
    humor_in_ai_root: Path,
    electronic_sheep_root: Path,
    adapter_seen_manifest: Path,
    output: Path,
    seed: int = 20260830,
    captions_per_source: int = 3,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> DatasetBuildResult:
    if captions_per_source < 1:
        raise ValueError("captions_per_source must be positive")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("invalid validation/test fractions")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    input_files: dict[str, str] = {}
    hic_descriptions = official_description_root / "humorbench/gpt4o_description"
    for source_split in ("train", "validation", "test"):
        description_file = hic_descriptions / f"{source_split}.jsonl"
        input_files[str(description_file)] = sha256(description_file)
        for description in _jsonl(description_file):
            contest = int(description["contest_number"])
            image = humor_in_ai_root / f"cartoons/source/{contest}.jpg"
            ranking = humor_in_ai_root / f"ranking/source/{contest}.csv"
            input_files[str(ranking)] = sha256(ranking)
            variants = _row_variants(
                dataset="humor_in_ai",
                contest=contest,
                image=image,
                description=description,
                captions=_ranked_captions(ranking, captions_per_source),
                source_split=source_split,
            )
            grouped[variants[0]["cluster_id"]].extend(variants)

    sheep_description_file = official_description_root / "electronic_sheep/description/description.jsonl"
    sheep_annotations_file = electronic_sheep_root / "annotations/all_newyorker_contest_annotations.json"
    input_files[str(sheep_description_file)] = sha256(sheep_description_file)
    input_files[str(sheep_annotations_file)] = sha256(sheep_annotations_file)
    annotations = json.loads(sheep_annotations_file.read_text())
    for description in _jsonl(sheep_description_file):
        contest = int(description["contest_number"])
        finalists = _finalists(annotations[str(contest)], captions_per_source)
        if not finalists:
            # Electronic Sheep releases descriptions for some contests without
            # official New Yorker finalists.  They are valid description data,
            # but cannot supervise a caption bridge and must not be fabricated.
            skipped.append({
                "dataset": "electronic_sheep",
                "contest_number": contest,
                "reason": "no_official_newyorker_finalists",
            })
            continue
        variants = _row_variants(
            dataset="electronic_sheep",
            contest=contest,
            image=electronic_sheep_root / f"images/all_contest_images/{contest}.jpeg",
            description=description,
            captions=finalists,
            source_split="released_full",
        )
        grouped[variants[0]["cluster_id"]].extend(variants)

    seen_manifest = json.loads(adapter_seen_manifest.read_text())
    seen = set(seen_manifest["exclude_from_latent_validation_and_test"])
    clusters = sorted(grouped, key=lambda value: int(value.rsplit("_", 1)[1]))
    unseen = [cluster for cluster in clusters if cluster not in seen]
    rng = random.Random(seed)
    rng.shuffle(unseen)
    n_validation = round(len(clusters) * validation_fraction)
    n_test = round(len(clusters) * test_fraction)
    if len(unseen) < n_validation + n_test:
        raise RuntimeError("not enough adapter-unseen clusters for validation and test")
    split_clusters = {
        "validation": set(unseen[:n_validation]),
        "test": set(unseen[n_validation : n_validation + n_test]),
    }
    split_clusters["train"] = set(clusters) - split_clusters["validation"] - split_clusters["test"]
    if seen & (split_clusters["validation"] | split_clusters["test"]):
        raise AssertionError("adapter-seen cluster leaked into validation/test")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    output.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        rows = [row for cluster in sorted(split_clusters[split]) for row in grouped[cluster]]
        rows.sort(key=lambda row: row["row_id"])
        split_rows[split] = rows
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        output_hashes[path.name] = sha256(path)

    overlaps = {
        "train_validation": len(split_clusters["train"] & split_clusters["validation"]),
        "train_test": len(split_clusters["train"] & split_clusters["test"]),
        "validation_test": len(split_clusters["validation"] & split_clusters["test"]),
    }
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "captions_per_source": captions_per_source,
        "description_policy": "verbatim canny field from HOMER released standard-description JSONL",
        "split_unit": "NYCC contest cluster_id",
        "adapter_seen_policy": "all frozen Planner/Generator SFT train+validation clusters forced into latent train",
        "source_rows": sum(len(values) for values in grouped.values()),
        "unique_clusters": len(clusters),
        "cross_source_duplicate_clusters": sum(
            len({row["dataset"] for row in values}) > 1 for values in grouped.values()
        ),
        "skipped_records": skipped,
        "skipped_record_count": len(skipped),
        "split_rows": {name: len(rows) for name, rows in split_rows.items()},
        "split_clusters": {name: len(values) for name, values in split_clusters.items()},
        "adapter_seen_clusters_in_validation": len(seen & split_clusters["validation"]),
        "adapter_seen_clusters_in_test": len(seen & split_clusters["test"]),
        "cluster_overlap": overlaps,
        "input_sha256": dict(sorted(input_files.items())),
        "adapter_seen_manifest": str(adapter_seen_manifest),
        "adapter_seen_manifest_sha256": sha256(adapter_seen_manifest),
        "output_sha256": output_hashes,
        "references": [
            "HOMER, ICLR 2026, arXiv:2602.06423",
            "Humor in AI, NeurIPS 2024, arXiv:2406.10522",
            "Electronic Sheep, ACL 2023, ACL:2023.acl-long.41",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return DatasetBuildResult(manifest, split_rows)
