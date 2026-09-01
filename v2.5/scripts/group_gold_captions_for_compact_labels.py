#!/usr/bin/env python
"""Group every selected gold caption by image for consensus-label generation."""

from __future__ import annotations

import hashlib
import json
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.sft_dataset import extract_caption, extract_image_path
from src.utils.io import read_jsonl, write_jsonl


def load_descriptions(raw_dir: Path) -> dict[int, dict[str, Any]]:
    descriptions: dict[int, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        path = raw_dir / "gpt4o_description" / f"{split}.jsonl"
        for row in read_jsonl(path):
            descriptions[int(row["contest_number"])] = row
    return descriptions


def group_rows(rows: list[dict[str, Any]], descriptions: dict[int, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for index, row in enumerate(rows):
        image = extract_image_path(row)
        image_id = str(row.get("image_id") or Path(str(image or "")).stem).strip()
        caption = str(extract_caption(row) or "").strip()
        raw_contest = (row.get("meta") or {}).get("contest_number") or image_id.removeprefix("nycc_")
        contest = int(raw_contest) if str(raw_contest).isdigit() else None
        if not image or not image_id or not caption:
            raise ValueError(f"Row {index} is missing image, image_id, or caption.")
        group = groups.setdefault(
            image_id,
            {
                "image": str(image),
                "image_id": image_id,
                "contest_number": contest,
                "captions": [],
                "source_rows": 0,
            },
        )
        if str(image) != group["image"]:
            raise ValueError(f"Image ID {image_id} maps to multiple image paths.")
        group["source_rows"] += 1
        if caption not in group["captions"]:
            group["captions"].append(caption)

    outputs: list[dict[str, Any]] = []
    for group in groups.values():
        captions = group["captions"]
        ranked = "\n".join(f"{index}. {caption}" for index, caption in enumerate(captions, start=1))
        description = (descriptions or {}).get(group["contest_number"], {})
        context = (
            "Auxiliary literal description:\n"
            f"{str(description.get('canny') or '').strip()}\n\n"
            "Auxiliary unusual visual fact:\n"
            f"{str(description.get('uncanny') or '').strip()}\n\n"
            "Auxiliary visible entity list:\n"
            f"{json.dumps(description.get('entities') or [], ensure_ascii=False)}\n\n"
            "Ranked high-rated captions:\n"
            f"{ranked}"
        )
        outputs.append(
            {
                "image": group["image"],
                "image_id": group["image_id"],
                "gold_caption": context,
                "gold_captions": captions,
                "auxiliary_visual_description": description,
                "caption_count": len(captions),
                "source_rows": group["source_rows"],
                "caption_set_sha256": hashlib.sha256(ranked.encode("utf-8")).hexdigest(),
            }
        )
    return outputs


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/newyorker_caption_ranking"))
    args = parser.parse_args()
    outputs = group_rows(read_jsonl(args.input_jsonl), descriptions=load_descriptions(args.raw_dir))
    write_jsonl(args.output_jsonl, outputs)
    print(
        json.dumps(
            {
                "source_rows": sum(row["source_rows"] for row in outputs),
                "unique_images": len(outputs),
                "unique_captions": sum(row["caption_count"] for row in outputs),
                "output": str(args.output_jsonl),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
