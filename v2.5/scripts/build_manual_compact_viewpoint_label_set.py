#!/usr/bin/env python3
"""Build a concise, reviewed compact-viewpoint label set and SFT teacher file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finalize_compact_viewpoint_labels import load_overrides
from scripts.verify_sft_generations import parse_compact_viewpoint
from src.utils.io import read_jsonl, write_jsonl


MEDIUM_CONFIDENCE_IDS = {
    "nycc_532",
    "nycc_539",
    "nycc_542",
    "nycc_546",
    "nycc_547",
    "nycc_555",
    "nycc_556",
    "nycc_569",
    "nycc_572",
    "nycc_577",
    "nycc_581",
    "nycc_584",
    "nycc_588",
    "nycc_595",
    "nycc_599",
    "nycc_642",
    "nycc_647",
    "nycc_650",
}
LOW_CONFIDENCE_IDS = {"nycc_656"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def confidence(image_id: str) -> str:
    if image_id in LOW_CONFIDENCE_IDS:
        return "low"
    if image_id in MEDIUM_CONFIDENCE_IDS:
        return "medium"
    return "high"


def build(
    base_path: Path,
    overrides_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_rows = read_jsonl(base_path)
    overrides = load_overrides(overrides_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(base_rows):
        image_id = str(source.get("image_id") or "").strip()
        image = str(source.get("image") or "").strip()
        if not image_id or not image or image_id in seen:
            raise ValueError(f"Missing or duplicate identity at base row {index}: {image_id!r}")
        seen.add(image_id)
        candidates = source.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"{image_id}: expected exactly one base candidate")

        base_label = parse_compact_viewpoint(str(candidates[0]))
        changed = image_id in overrides
        label = overrides[image_id]["label"] if changed else base_label
        label = parse_compact_viewpoint(json.dumps(label, ensure_ascii=False))
        review_notes = ""
        if image_id == "nycc_656":
            review_notes = (
                "The image depicts a guitar-playing therapy patient, but all ranked captions "
                "describe an unrelated political/planetary scene. The label is image-grounded; "
                "this sample should be manually checked or excluded before final SFT."
            )

        review = {
            "status": "reviewed",
            "reviewer": "OpenAI Codex manual visual-and-caption-consensus review",
            "review_date": "2026-08-20",
            "changed_from_previous_label": changed,
            "confidence": confidence(image_id),
            "caption_evidence_used": True,
            "notes": review_notes,
        }
        labels.append(
            {
                "image": image,
                "image_id": image_id,
                "label": label,
                "review": review,
            }
        )
        teacher_rows.append(
            {
                "image": image,
                "image_id": image_id,
                "source_image_id": source.get("source_image_id") or image_id,
                "caption_count": int(source.get("caption_count") or 0),
                "caption_set_sha256": source.get("caption_set_sha256"),
                "candidates": [
                    json.dumps(label, ensure_ascii=False, separators=(",", ":"))
                ],
                "manual_review": review,
                "previous_label": (
                    json.dumps(base_label, ensure_ascii=False, separators=(",", ":"))
                    if changed
                    else None
                ),
                "manual_correction_reason": (
                    overrides[image_id]["reason"] if changed else None
                ),
            }
        )

    unused_overrides = sorted(set(overrides) - seen)
    if unused_overrides:
        raise ValueError(f"Overrides absent from base data: {unused_overrides}")
    if len(labels) != 79:
        raise ValueError(f"Expected 79 reviewed training labels, got {len(labels)}")

    labels_path = output_dir / "train_labels_manual_v2.jsonl"
    teacher_path = output_dir / "train_teacher_manual_v2.jsonl"
    write_jsonl(labels_path, labels)
    write_jsonl(teacher_path, teacher_rows)
    report = {
        "base": str(base_path),
        "overrides": str(overrides_path),
        "rows": len(labels),
        "reviewed_rows": sum(row["review"]["status"] == "reviewed" for row in labels),
        "corrected_rows": sum(row["review"]["changed_from_previous_label"] for row in labels),
        "accepted_after_review_rows": sum(
            not row["review"]["changed_from_previous_label"] for row in labels
        ),
        "confidence_counts": {
            level: sum(row["review"]["confidence"] == level for row in labels)
            for level in ("high", "medium", "low")
        },
        "low_confidence_ids": sorted(LOW_CONFIDENCE_IDS),
        "caption_image_conflict_ids": ["nycc_656"],
        "outputs": {
            labels_path.name: {"sha256": digest(labels_path)},
            teacher_path.name: {"sha256": digest(teacher_path)},
        },
        "provenance_note": (
            "These are AI-assisted manual visual reviews by Codex, not independent human "
            "annotations. A human should adjudicate medium/low-confidence records before training."
        ),
    }
    report_path = output_dir / "train_manual_v2_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("outputs/newyorker_compact_viewpoint_teacher_7b/train_final.jsonl"),
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path("data/overrides/compact_viewpoint_train_manual_v2_overrides.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/nycc_planner_v2/adjudicated"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.base, args.overrides, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
