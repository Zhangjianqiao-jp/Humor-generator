#!/usr/bin/env python3
"""Export ranked captions and blank humor-point records for 7B train images."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = Path(
    "data/annotations/nycc_planner_v2/packets/train_image_only/manifest.jsonl"
)
DEFAULT_CAPTIONS = Path("data/processed/newyorker_top3pct_sft/sft_train.jsonl")
DEFAULT_OUTPUT = Path(
    "data/annotations/nycc_planner_v2/packets/train_caption_evidence"
)

MECHANISMS = [
    "role_reversal",
    "scale_violation",
    "object_substitution",
    "context_collision",
    "literalized_idiom",
    "anachronism",
    "status_reversal",
    "expectation_break",
    "knowledge_reference",
    "unclear",
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def assistant_text(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return content.strip()
        for item in content:
            if item.get("type") == "text":
                return str(item.get("text") or "").strip()
    return ""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export(
    image_manifest_path: Path,
    caption_path: Path,
    output_dir: Path,
    review_count: int,
) -> dict[str, Any]:
    image_manifest = read_jsonl(image_manifest_path)
    image_by_id = OrderedDict((str(row["image_id"]), row) for row in image_manifest)
    if len(image_by_id) != len(image_manifest):
        raise ValueError("Image manifest contains duplicate image_id values")

    grouped: dict[str, list[dict[str, Any]]] = {image_id: [] for image_id in image_by_id}
    for row in read_jsonl(caption_path):
        image_id = str(row.get("image_id") or "")
        if image_id not in grouped:
            continue
        caption = assistant_text(row)
        if not caption:
            raise ValueError(f"Empty caption for {image_id}")
        meta = row.get("meta") or {}
        grouped[image_id].append(
            {
                "rank": int(meta["rank"]),
                "caption": caption,
                "score": float(meta["score"]),
                "votes": int(meta["votes"]),
                "funny_votes": int(meta["funny_votes"]),
            }
        )

    missing = [image_id for image_id, captions in grouped.items() if not captions]
    if missing:
        raise ValueError(f"Training images without gold captions: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    text_dir = output_dir / "top10_by_image"
    text_dir.mkdir(parents=True, exist_ok=True)

    full_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []

    for image_id, image_record in image_by_id.items():
        captions = sorted(grouped[image_id], key=lambda item: (item["rank"], item["caption"]))
        top_captions = captions[:review_count]
        index = int(image_record["index"])
        evidence_name = f"{index:03d}_{image_id}.txt"

        full_rows.append(
            {
                "index": index,
                "image_id": image_id,
                "image": f"../train_image_only/{image_record['filename']}",
                "caption_count": len(captions),
                "captions": captions,
            }
        )
        review_rows.append(
            {
                "index": index,
                "image_id": image_id,
                "image": f"../train_image_only/{image_record['filename']}",
                "caption_count_full_pool": len(captions),
                "captions": top_captions,
            }
        )

        with (text_dir / evidence_name).open("w", encoding="utf-8") as handle:
            handle.write(f"image_id: {image_id}\n")
            handle.write(f"image: ../../train_image_only/{image_record['filename']}\n")
            handle.write(f"full_top_3pct_caption_count: {len(captions)}\n\n")
            for position, item in enumerate(top_captions, start=1):
                handle.write(f"{position}. [source rank {item['rank']}] {item['caption']}\n")

        annotation_rows.append(
            {
                "index": index,
                "image_id": image_id,
                "image": f"../train_image_only/{image_record['filename']}",
                "caption_evidence": f"top10_by_image/{evidence_name}",
                "annotation_version": "compact-humor-point-v2",
                "humorous_point": {
                    "literal_facts": [],
                    "normal_expectation": "",
                    "incongruity": "",
                    "resolution": "",
                    "mechanism": "",
                    "speaker_options": [],
                    "caption_angles": [],
                    "external_knowledge": [],
                },
                "visual_ambiguities": [],
                "confidence": "",
                "annotator_notes": "",
                "status": "unlabeled",
            }
        )

    write_jsonl(output_dir / "gold_captions_all.jsonl", full_rows)
    write_jsonl(output_dir / "gold_captions_top10.jsonl", review_rows)
    write_jsonl(output_dir / "humorous_point_template.jsonl", annotation_rows)

    manifest = {
        "image_count": len(image_by_id),
        "gold_caption_count": sum(len(captions) for captions in grouped.values()),
        "review_captions_per_image": review_count,
        "selection": "per-cartoon top 3% by source rank; review file keeps the first 10",
        "annotation_order": [
            "Fill literal_facts, speaker_options, visual_ambiguities, and confidence from the image only.",
            "Then open caption_evidence and fill the remaining humor reasoning fields.",
        ],
        "allowed_mechanisms": MECHANISMS,
        "license": "CC-BY-NC-4.0; academic research only under the local source dataset terms",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-count", type=int, default=10)
    args = parser.parse_args()
    if args.review_count < 1:
        raise ValueError("--review-count must be positive")
    manifest = export(
        resolve(args.image_manifest),
        resolve(args.captions),
        resolve(args.output_dir),
        args.review_count,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
