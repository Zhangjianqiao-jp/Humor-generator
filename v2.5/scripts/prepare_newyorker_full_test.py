#!/usr/bin/env python
"""Prepare explicit full-pair and unique-image New Yorker held-out test files."""

from __future__ import annotations

import hashlib
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.sft_dataset import extract_caption, extract_image_path
from src.utils.io import read_jsonl, write_jsonl


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prepare(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        image = extract_image_path(row)
        caption = extract_caption(row)
        image_id = str(row.get("image_id") or Path(str(image or "")).stem).strip()
        if not image or not caption or not image_id:
            raise ValueError(f"Test row {index} is missing image, image_id, or caption.")
        pair = {
            "image": str(image),
            "image_id": image_id,
            "gold_caption": str(caption).strip(),
            "pair_id": f"{image_id}::{index:06d}",
            "meta": row.get("meta") or {},
        }
        pairs.append(pair)
        group = grouped.setdefault(
            image_id,
            {
                "image": str(image),
                "image_id": image_id,
                "gold_caption": str(caption).strip(),
                "gold_captions": [],
                "reference_count": 0,
            },
        )
        if str(caption).strip() not in group["gold_captions"]:
            group["gold_captions"].append(str(caption).strip())
        group["reference_count"] += 1
    unique = list(grouped.values())
    return pairs, unique


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs, unique = prepare(read_jsonl(args.input_jsonl))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.output_dir / "test_pairs_full.jsonl"
    unique_path = args.output_dir / "test_images_unique.jsonl"
    write_jsonl(pair_path, pairs)
    write_jsonl(unique_path, unique)
    manifest = {
        "source": str(args.input_jsonl),
        "full_pairs": len(pairs),
        "unique_images": len(unique),
        "duplicate_image_caption_pairs": len(pairs) - len(unique),
        "files": {
            pair_path.name: {"rows": len(pairs), "sha256": digest(pair_path)},
            unique_path.name: {"rows": len(unique), "sha256": digest(unique_path)},
        },
        "evaluation_note": "Run inference once for every unique test image without exposing references; use all full-pair captions as the multi-reference evaluation pool.",
    }
    manifest_path = args.output_dir / "test_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
