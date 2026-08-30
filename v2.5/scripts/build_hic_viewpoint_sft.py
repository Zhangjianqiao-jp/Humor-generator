#!/usr/bin/env python
"""Convert audited 7B HIC-viewpoint generations into multimodal SFT JSONL."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_sft_generations import parse_compact_viewpoint
from src.utils.io import read_jsonl, write_jsonl


def build_rows(
    rows: list[dict[str, Any]],
    planner_prompt: str,
    prompt_version: str = "image-to-compact-viewpoint-v1",
    task_name: str = "image_to_compact_viewpoint",
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for index, row in enumerate(rows):
        image = str(row.get("image") or "").strip()
        image_id = str(row.get("image_id") or Path(image).stem).strip()
        candidates = row.get("candidates")
        if not image or not image_id:
            raise ValueError(f"Teacher row {index} is missing image or image_id.")
        if image_id in seen_images:
            raise ValueError(f"Duplicate image_id in teacher rows: {image_id}")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"Teacher row {index} must contain exactly one candidate.")
        target = str(candidates[0]).strip()
        annotation = parse_compact_viewpoint(target)
        # Re-serialize to a stable compact representation without changing values.
        target = json.dumps(annotation, ensure_ascii=False, separators=(",", ":"))
        manual_review = row.get("manual_review")
        is_manually_reviewed = isinstance(manual_review, dict)
        seen_images.add(image_id)
        outputs.append(
            {
                "image": image,
                "image_id": image_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": planner_prompt},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": target}]},
                ],
                "meta": {
                    "task": task_name,
                    "prompt_version": prompt_version,
                    "teacher_caption_count": int(row.get("caption_count") or 0),
                    "teacher_caption_set_sha256": row.get("caption_set_sha256"),
                    "teacher": (
                        "OpenAI Codex manual visual-and-caption-consensus review"
                        if is_manually_reviewed
                        else "Qwen2.5-VL-7B-Instruct base self-distillation"
                    ),
                    "label_provenance": row.get("label_provenance") or (
                        "manual_ai_review_v2"
                        if is_manually_reviewed
                        else "base_self_distillation"
                    ),
                    "manual_review": manual_review if is_manually_reviewed else None,
                },
            }
        )
    return outputs


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-generations", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--planner-prompt-file", type=Path, required=True)
    parser.add_argument(
        "--prompt-version", default="image-to-compact-viewpoint-v1"
    )
    parser.add_argument("--task-name", default="image_to_compact_viewpoint")
    args = parser.parse_args()
    planner_prompt = args.planner_prompt_file.read_text(encoding="utf-8").strip()
    if not planner_prompt:
        raise ValueError(f"Planner prompt is empty: {args.planner_prompt_file}")
    outputs = build_rows(
        read_jsonl(args.teacher_generations),
        planner_prompt=planner_prompt,
        prompt_version=args.prompt_version,
        task_name=args.task_name,
    )
    write_jsonl(args.output_jsonl, outputs)
    print(f"[viewpoint-sft] saved {len(outputs)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
