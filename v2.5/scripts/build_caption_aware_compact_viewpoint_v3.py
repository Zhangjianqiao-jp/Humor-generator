#!/usr/bin/env python3
"""Merge reviewed visual labels with caption-derived humor bridges for planner SFT."""

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

from scripts.verify_sft_generations import parse_compact_viewpoint
from src.utils.io import read_jsonl, write_jsonl


CONFLICT_ID = "nycc_656"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build(base_path: Path, targets_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(base_path)
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    if not isinstance(targets, dict):
        raise ValueError("Caption-aware targets must be a JSON object keyed by image_id.")

    output_dir.mkdir(parents=True, exist_ok=True)
    labels: list[dict[str, Any]] = []
    teacher_clean: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, source in enumerate(rows):
        image_id = str(source.get("image_id") or "").strip()
        if not image_id or image_id in seen:
            raise ValueError(f"Missing or duplicate image_id at row {index}: {image_id!r}")
        seen.add(image_id)
        if image_id not in targets:
            raise ValueError(f"Missing caption-aware target for {image_id}")

        candidates = source.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"{image_id}: expected exactly one visual label")
        label = parse_compact_viewpoint(str(candidates[0]))
        target = targets[image_id]
        status = "quarantine" if target is None else "accepted"
        if target is not None:
            if not isinstance(target, str) or len(target.split()) < 8:
                raise ValueError(f"{image_id}: caption-aware target is too short or invalid")
            label["target"] = target
            # The bridge explicitly invokes caption-supported semantic frames.
            label["external_knowledge"] = True
            label = parse_compact_viewpoint(json.dumps(label, ensure_ascii=False))

        review = {
            "status": status,
            "reviewer": "OpenAI Codex caption-consensus bridge annotation",
            "review_date": "2026-08-20",
            "caption_evidence_used": True,
            "confidence": "low" if status == "quarantine" else "human_review_required",
            "notes": (
                "Upstream mapping conflict: ranking/655.csv and ranking/656.csv contain the same "
                "5,278 captions and match cartoons/source/655.jpg, which belongs to the test split; "
                "cartoons/source/656.jpg instead matches ranking/657.csv. Excluded to prevent "
                "image-caption corruption and train-test leakage."
                if status == "quarantine"
                else "AI-assisted bridge inferred from the image and its high-scoring caption family; not independent human annotation."
            ),
        }
        label_row = {
            "image": source["image"],
            "image_id": image_id,
            "label": label if status == "accepted" else None,
            "previous_visual_label": label if status == "quarantine" else None,
            "review": review,
        }
        labels.append(label_row)

        teacher_row = {
            "image": source["image"],
            "image_id": image_id,
            "source_image_id": source.get("source_image_id") or image_id,
            "caption_count": int(source.get("caption_count") or 0),
            "caption_set_sha256": source.get("caption_set_sha256"),
            "candidates": (
                [json.dumps(label, ensure_ascii=False, separators=(",", ":"))]
                if status == "accepted"
                else []
            ),
            "manual_review": review,
            "label_provenance": "caption_consensus_bridge_v3",
        }
        if status == "accepted":
            teacher_clean.append(teacher_row)
        else:
            quarantine.append({**label_row, "teacher_source": teacher_row})

    extra = sorted(set(targets) - seen)
    if extra:
        raise ValueError(f"Targets absent from the label set: {extra}")
    if len(labels) != 79 or len(teacher_clean) != 78 or len(quarantine) != 1:
        raise ValueError(
            f"Expected 79 total, 78 clean, 1 quarantine; got "
            f"{len(labels)}, {len(teacher_clean)}, {len(quarantine)}"
        )

    labels_path = output_dir / "train_labels_caption_aware_v3.jsonl"
    teacher_path = output_dir / "train_teacher_caption_aware_v3_clean.jsonl"
    quarantine_path = output_dir / "train_caption_aware_v3_quarantine.jsonl"
    write_jsonl(labels_path, labels)
    write_jsonl(teacher_path, teacher_clean)
    write_jsonl(quarantine_path, quarantine)

    report = {
        "definition": (
            "A caption-aware bridge links visible incongruity to the shared semantic frames "
            "found across high-scoring captions; it is neither a literal scene description "
            "nor a finished caption."
        ),
        "source_visual_labels": str(base_path),
        "source_caption_aware_targets": str(targets_path),
        "total_images": len(labels),
        "clean_training_rows": len(teacher_clean),
        "quarantined_rows": len(quarantine),
        "quarantined_ids": [CONFLICT_ID],
        "source_mapping_audit": {
            "ranking_655_unique_captions": 5278,
            "ranking_656_unique_captions": 5278,
            "ranking_655_656_shared_captions": 5278,
            "ranking_655_656_caption_jaccard": 1.0,
            "ranking_656_657_shared_captions": 2,
            "correct_globe_image": "data/raw/newyorker_caption_ranking/cartoons/source/655.jpg",
            "correct_globe_image_split": "test",
        },
        "human_review_required": True,
        "outputs": {
            labels_path.name: digest(labels_path),
            teacher_path.name: digest(teacher_path),
            quarantine_path.name: digest(quarantine_path),
        },
    }
    report_path = output_dir / "train_caption_aware_v3_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/annotations/nycc_planner_v2/adjudicated/train_teacher_manual_v2.jsonl"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/overrides/compact_viewpoint_train_caption_aware_v3_targets.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations/nycc_planner_v3/adjudicated"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.targets, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
