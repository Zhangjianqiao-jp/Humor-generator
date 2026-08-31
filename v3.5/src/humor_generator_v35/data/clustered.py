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


INVALID_CAPTION_SENTINELS = {"nan", "none", "null", "unk", "unknown"}


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
    raw_finalists = annotation.get("official_newyorker_finalists")
    # Electronic Sheep uses the scalar string "UNKNOWN" when finalists are
    # unavailable.  Iterating that value used to create the bogus captions
    # "U", "N", and "K".  Only an actual list is a released finalist set.
    if not isinstance(raw_finalists, list):
        return []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, raw in enumerate(raw_finalists):
        caption = " ".join(str(raw).strip(" \t\r\n\"“”").split())
        normalized = caption.casefold()
        if not caption or normalized in INVALID_CAPTION_SENTINELS or normalized in seen:
            continue
        seen.add(normalized)
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
    internal_test_fraction: float = 0.15,
) -> DatasetBuildResult:
    if captions_per_source < 1:
        raise ValueError("captions_per_source must be positive")
    if (
        validation_fraction <= 0
        or internal_test_fraction <= 0
        or validation_fraction + internal_test_fraction >= 1
    ):
        raise ValueError("invalid validation/internal-test fractions")

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

    # The public 47-image Humor in AI test is never resampled into bridge
    # training or checkpoint selection. This preserves an external benchmark
    # in addition to the internal method-development test.
    official_hia_all = {
        cluster
        for cluster, rows in grouped.items()
        if any(row["dataset"] == "humor_in_ai" and row["source_split"] == "test" for row in rows)
    }
    official_hia_seen = seen & official_hia_all
    official_hia_unseen = official_hia_all - seen
    if not official_hia_seen or not official_hia_unseen:
        raise RuntimeError("expected both seen and unseen partitions of the official HIA test")

    eligible = [cluster for cluster in clusters if cluster not in seen | official_hia_all]
    strata: dict[str, list[str]] = defaultdict(list)
    for cluster in eligible:
        signature = "+".join(sorted({row["dataset"] for row in grouped[cluster]}))
        strata[signature].append(cluster)
    validation: set[str] = set()
    internal_test: set[str] = set()
    for offset, signature in enumerate(sorted(strata)):
        values = sorted(strata[signature], key=lambda value: int(value.rsplit("_", 1)[1]))
        random.Random(seed + offset).shuffle(values)
        n_validation = round(len(values) * validation_fraction)
        n_internal_test = round(len(values) * internal_test_fraction)
        validation.update(values[:n_validation])
        internal_test.update(values[n_validation : n_validation + n_internal_test])
    split_clusters = {
        "validation": validation,
        "internal_test": internal_test,
        "official_hia_unseen_test": official_hia_unseen,
        "official_hia_seen_diagnostic": official_hia_seen,
    }
    split_clusters["train"] = set(clusters) - validation - internal_test - official_hia_all
    uncontaminated_evaluation = validation | internal_test | official_hia_unseen
    if seen & uncontaminated_evaluation:
        raise AssertionError("adapter-seen cluster leaked into an evaluation split")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    output.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for split in (
        "train", "validation", "internal_test",
        "official_hia_unseen_test", "official_hia_seen_diagnostic",
    ):
        rows = [row for cluster in sorted(split_clusters[split]) for row in grouped[cluster]]
        if split.startswith("official_hia_"):
            rows = [row for row in rows if row["dataset"] == "humor_in_ai"]
        rows.sort(key=lambda row: row["row_id"])
        split_rows[split] = rows
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        output_hashes[path.name] = sha256(path)

    # Planner traces depend on image/description inputs, not caption labels.
    # Pin those inputs separately so caption cleaning does not invalidate
    # expensive traces when their actual inputs are byte-identical.
    trace_inputs: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        for row in split_rows[split]:
            trace_inputs.setdefault(row["cluster_id"], {
                "cluster_id": row["cluster_id"],
                "split": split,
                "image_sha256": row["image_sha256"],
                "standard_description": row["standard_description"],
            })
    trace_input_path = output / "trace_inputs.jsonl"
    trace_input_path.write_text("".join(
        json.dumps(trace_inputs[cluster], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for cluster in sorted(trace_inputs)
    ))
    output_hashes[trace_input_path.name] = sha256(trace_input_path)

    overlaps = {
        "train_validation": len(split_clusters["train"] & split_clusters["validation"]),
        "train_internal_test": len(split_clusters["train"] & split_clusters["internal_test"]),
        "train_official_hia_unseen_test": len(split_clusters["train"] & split_clusters["official_hia_unseen_test"]),
        "train_official_hia_seen_diagnostic": len(split_clusters["train"] & split_clusters["official_hia_seen_diagnostic"]),
        "validation_internal_test": len(split_clusters["validation"] & split_clusters["internal_test"]),
        "validation_official_hia_unseen_test": len(split_clusters["validation"] & split_clusters["official_hia_unseen_test"]),
        "validation_official_hia_seen_diagnostic": len(split_clusters["validation"] & split_clusters["official_hia_seen_diagnostic"]),
        "internal_official_hia_unseen_test": len(split_clusters["internal_test"] & split_clusters["official_hia_unseen_test"]),
        "internal_official_hia_seen_diagnostic": len(split_clusters["internal_test"] & split_clusters["official_hia_seen_diagnostic"]),
        "official_unseen_seen_diagnostic": len(split_clusters["official_hia_unseen_test"] & split_clusters["official_hia_seen_diagnostic"]),
    }
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "captions_per_source": captions_per_source,
        "description_policy": "verbatim canny field from HOMER released standard-description JSONL",
        "split_unit": "NYCC contest cluster_id",
        "adapter_seen_policy": (
            "adapter-seen clusters forced into bridge train except the 23 official HIA test "
            "clusters, which are excluded from bridge fitting and labelled diagnostic-only"
        ),
        "official_hia_test_policy": (
            "all 47 official HIA test clusters excluded from bridge training; "
            "24 adapter-unseen clusters are confirmatory and 23 adapter-seen clusters diagnostic only"
        ),
        "split_policy": "source-stratified image-cluster split after reserving official HIA test",
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
        "adapter_seen_clusters_in_internal_test": len(seen & split_clusters["internal_test"]),
        "adapter_seen_clusters_in_official_hia_unseen_test": len(seen & split_clusters["official_hia_unseen_test"]),
        "adapter_seen_clusters_in_official_hia_seen_diagnostic": len(seen & split_clusters["official_hia_seen_diagnostic"]),
        "official_hia_all_clusters_in_train": len(official_hia_all & split_clusters["train"]),
        "official_hia_all_clusters_in_validation": len(official_hia_all & split_clusters["validation"]),
        "cluster_overlap": overlaps,
        "input_sha256": dict(sorted(input_files.items())),
        "adapter_seen_manifest": str(adapter_seen_manifest),
        "adapter_seen_manifest_sha256": sha256(adapter_seen_manifest),
        "output_sha256": output_hashes,
        "trace_input_manifest": trace_input_path.name,
        "trace_input_manifest_sha256": output_hashes[trace_input_path.name],
        "references": [
            "HOMER, ICLR 2026, arXiv:2602.06423",
            "Humor in AI, NeurIPS 2024, arXiv:2406.10522",
            "Electronic Sheep, ACL 2023, ACL:2023.acl-long.41",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return DatasetBuildResult(manifest, split_rows)
