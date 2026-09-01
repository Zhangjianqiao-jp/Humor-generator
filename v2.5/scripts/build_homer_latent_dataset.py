#!/usr/bin/env python3
"""Build the HOMER two-benchmark bridge corpus with image-clustered splits.

HOMER reports 365 Humor-in-AI and 679 Electronic-Sheep examples.  The two
sources overlap in contest IDs, so this builder reports both dataset rows and
independent image clusters and never lets a cluster cross train/val/test.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import csv
import json
from pathlib import Path
import random
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def extract_image_path(row: dict[str, Any]) -> str | None:
    return row.get("image") or row.get("image_path")


def extract_caption(row: dict[str, Any]) -> str | None:
    if row.get("caption"):
        return str(row["caption"])
    messages = row.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return None


def image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hic_rows(rankings: Path, images: Path, limit: int) -> list[dict[str, Any]]:
    output = []
    for ranking in sorted(rankings.glob("*.csv"), key=lambda path: int(path.stem)):
        image = images / f"{ranking.stem}.jpg"
        with ranking.open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        if not image.exists() or not records:
            continue
        records.sort(key=lambda row: float(row.get("mean") or 0), reverse=True)
        output.append({
            "dataset": "humor_in_ai", "source_id": ranking.stem, "image": str(image),
            "caption": str(records[0]["caption"]).strip(), "image_sha256": image_digest(image),
            "cluster_id": f"nycc_{ranking.stem}",
        })
        if len(output) == limit:
            break
    return output


def _electronic_sheep_rows(images: Path, annotations: Path, limit: int) -> list[dict[str, Any]]:
    raw = json.loads(annotations.read_text(encoding="utf-8"))
    output = []
    for contest in sorted(raw, key=lambda value: int(value)):
        image = images / f"{contest}.jpeg"
        finalists = raw[contest].get("official_newyorker_finalists") or []
        if image.exists() and finalists:
            output.append({
                "dataset": "electronic_sheep", "source_id": contest, "image": str(image),
                "caption": str(finalists[0]).strip(), "image_sha256": image_digest(image),
                "cluster_id": f"nycc_{contest}",
            })
        if len(output) == limit:
            break
    return output


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--hic-rankings", type=Path, default=Path("data/raw/newyorker_caption_ranking/ranking/source"))
    parser.add_argument("--hic-images", type=Path, default=Path("data/raw/newyorker_caption_ranking/cartoons/source"))
    parser.add_argument("--sheep-images", type=Path, default=Path("data/raw/electronic_sheep/images/all_contest_images"))
    parser.add_argument("--sheep-annotations", type=Path, default=Path("data/raw/electronic_sheep/annotations/all_newyorker_contest_annotations.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/homer_latent"))
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    rows = _hic_rows(args.hic_rankings, args.hic_images, 365) + _electronic_sheep_rows(args.sheep_images, args.sheep_annotations, 679)
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[row["cluster_id"]].append(row)
    keys = sorted(clusters)
    random.Random(args.seed).shuffle(keys)
    n_train, n_val = int(0.8 * len(keys)), int(0.1 * len(keys))
    split_keys = {"train": keys[:n_train], "validation": keys[n_train:n_train+n_val], "test": keys[n_train+n_val:]}
    args.output.mkdir(parents=True, exist_ok=True)
    for split, subset in split_keys.items():
        values = [row for key in subset for row in clusters[key]]
        write_jsonl(args.output / f"{split}.jsonl", values)
    manifest = {
        "paper_target_rows": 1044,
        "actual_rows": len(rows),
        "independent_image_clusters": len(clusters),
        "cross_source_duplicate_rows": len(rows) - len(clusters),
        "split_isolation_key": "NYCC contest ID (stronger than byte hash across JPEG encodings)",
        "split_cluster_counts": {key: len(value) for key, value in split_keys.items()},
        "warning": "HOMER source counts are not 1,044 independent cartoons; report rows and clusters separately.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
