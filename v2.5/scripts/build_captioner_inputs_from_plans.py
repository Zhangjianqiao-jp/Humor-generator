#!/usr/bin/env python
"""Bridge planner generations into image+plan prompts for the captioner."""

from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl
from scripts.verify_sft_generations import parse_compact_viewpoint, parse_hic_viewpoint


CAPTIONER_INSTRUCTION = "Generate one short, natural, image-specific humorous caption. Do not explain."
HIC_BASE_PROMPT = "Generate one short, natural, image-specific humorous caption for this image. Do not explain."


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _strip_source_caption(text: str, gold_caption: str) -> str:
    text = _clean_text(text)
    caption = _clean_text(gold_caption)
    if caption:
        text = re.sub(re.escape(caption), "the target joke", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe\s+gold\s+caption\b", "the target joke", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgold\s+caption\b", "target joke", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe\s+caption\b", "the target joke", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcaption\s+['\"][^'\"]{1,160}['\"]", "target joke", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def build_hic_compact_json_prompt(annotation: dict[str, Any], gold_caption: str) -> tuple[str, str]:
    anchors: list[dict[str, str]] = []
    for value in annotation.get("visual_anchors") or []:
        if not isinstance(value, dict):
            continue
        anchors.append(
            {
                "label": _clean_text(value.get("label")),
                "evidence": _clean_text(value.get("evidence")),
                "role": _strip_source_caption(str(value.get("role") or ""), gold_caption),
            }
        )
        if len(anchors) >= 4:
            break
    views = [_clean_text(value) for value in annotation.get("required_viewpoints") or [] if _clean_text(value)]
    primary_view = _clean_text(annotation.get("primary_viewpoint")) or (views[0] if views else "full_image")
    if primary_view not in views:
        views.insert(0, primary_view)
    payload = {
        "scene": _clean_text(annotation.get("literal_image_description")),
        "type": _clean_text(annotation.get("humor_type")) or "unclear_or_weak",
        "target": _strip_source_caption(str(annotation.get("humor_point") or ""), gold_caption),
        "primary_view": primary_view,
        "views": views[:4],
        "anchors": anchors,
        "external_knowledge": bool(annotation.get("needs_external_knowledge")),
    }
    compact_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    lines = [
        "Auxiliary visual joke annotations are provided below as compact JSON.",
        "Trust the image first. Ignore annotations that conflict with it.",
        "Use the compact JSON as joke clues, not wording to copy.",
        "Write a caption, not an image description.",
        "Prefer a punchline or meme-style line over a full sentence explanation.",
        "Maximum 12 words.",
        "Output exactly one caption.",
        "Do not repeat the JSON or explain it.",
        "Do not use because, since, which is, creating, visual effect, image, photo, scene, joke, humor, or funny.",
        "Do not name the humor type, viewpoint, annotation labels, or JSON fields.",
        "Do not use abstract analysis words such as contrast, mismatch, reversal, unexpected, target, anchor, viewpoint, label, role, or scale.",
        "",
        f"<joke_annotations>{compact_json}</joke_annotations>",
        "",
        HIC_BASE_PROMPT,
    ]
    return "\n".join(lines).strip(), compact_json


def build_compact_json_caption_prompt(payload: dict[str, Any]) -> tuple[str, str]:
    compact_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    lines = [
        "Auxiliary visual joke annotations are provided below as compact JSON.",
        "Trust the image first. Ignore annotations that conflict with it.",
        "Use the compact JSON as joke clues, not wording to copy.",
        "Write a caption, not an image description.",
        "Prefer a punchline or meme-style line over a full sentence explanation.",
        "Maximum 12 words.",
        "Output exactly one caption.",
        "Do not repeat the JSON or explain it.",
        "Do not use because, since, which is, creating, visual effect, image, photo, scene, joke, humor, or funny.",
        "Do not name the humor type, viewpoint, annotation labels, or JSON fields.",
        "Do not use abstract analysis words such as contrast, mismatch, reversal, unexpected, target, anchor, viewpoint, label, role, or scale.",
        "",
        f"<joke_annotations>{compact_json}</joke_annotations>",
        "",
        HIC_BASE_PROMPT,
    ]
    return "\n".join(lines).strip(), compact_json


def build_captioner_rows(planner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for index, row in enumerate(planner_rows):
        image = str(row.get("image") or "").strip()
        image_id = str(row.get("image_id") or Path(image).stem).strip()
        candidates = row.get("candidates")
        if not image:
            raise ValueError(f"Planner row {index} has no image path.")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Planner row {index} has no generated plan candidate.")
        plan = str(candidates[0]).strip()
        if not plan:
            raise ValueError(f"Planner row {index} has an empty generated plan candidate.")
        if image_id in seen_images:
            raise ValueError(f"Planner generations contain duplicate image_id: {image_id}")
        seen_images.add(image_id)
        outputs.append(
            {
                "image": image,
                "image_id": image_id,
                "prompt": f"{CAPTIONER_INSTRUCTION}\n\nHumor plan:\n{plan}",
                "planner_candidate": plan,
                "planner_gold": str(row.get("gold_caption") or "").strip(),
                "gold_caption": str(row.get("gold_caption") or "").strip(),
            }
        )
    return outputs


def build_hic_compact_json_rows(planner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for index, row in enumerate(planner_rows):
        image = str(row.get("image") or "").strip()
        image_id = str(row.get("image_id") or Path(image).stem).strip()
        candidates = row.get("candidates")
        gold_caption = str(row.get("gold_caption") or "").strip()
        if not image:
            raise ValueError(f"Planner row {index} has no image path.")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Planner row {index} has no generated annotation candidate.")
        if not gold_caption:
            raise ValueError(f"Planner row {index} has no gold caption for compact rendering.")
        try:
            annotation = parse_hic_viewpoint(str(candidates[0]).strip())
        except ValueError as exc:
            raise ValueError(f"Planner row {index} annotation is invalid: {exc}") from exc
        if image_id in seen_images:
            raise ValueError(f"Planner generations contain duplicate image_id: {image_id}")
        seen_images.add(image_id)
        prompt, compact_json = build_hic_compact_json_prompt(annotation, gold_caption)
        outputs.append(
            {
                "image": image,
                "image_id": image_id,
                "prompt": prompt,
                "planner_candidate": str(candidates[0]).strip(),
                "compact_json": compact_json,
                "planner_gold": gold_caption,
                "gold_caption": gold_caption,
                "source_image_id": row.get("source_image_id") or image_id.split("::", 1)[0],
            }
        )
    return outputs


def build_compact_viewpoint_rows(planner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for index, row in enumerate(planner_rows):
        image = str(row.get("image") or "").strip()
        image_id = str(row.get("image_id") or Path(image).stem).strip()
        candidates = row.get("candidates")
        if not image:
            raise ValueError(f"Planner row {index} has no image path.")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError(f"Planner row {index} must have exactly one compact annotation.")
        if image_id in seen_images:
            raise ValueError(f"Planner generations contain duplicate image_id: {image_id}")
        try:
            payload = parse_compact_viewpoint(str(candidates[0]).strip())
        except ValueError as exc:
            raise ValueError(f"Planner row {index} compact annotation is invalid: {exc}") from exc
        seen_images.add(image_id)
        prompt, compact_json = build_compact_json_caption_prompt(payload)
        outputs.append(
            {
                "image": image,
                "image_id": image_id,
                "prompt": prompt,
                "planner_candidate": str(candidates[0]).strip(),
                "compact_json": compact_json,
                "gold_caption": str(row.get("gold_caption") or "").strip(),
                "gold_captions": row.get("gold_captions"),
                "source_image_id": row.get("source_image_id") or image_id,
            }
        )
    return outputs


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--planner-generations", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("three-line", "hic-compact-json", "compact-viewpoint"),
        default="three-line",
    )
    args = parser.parse_args()

    planner_rows = read_jsonl(args.planner_generations)
    if args.format == "compact-viewpoint":
        captioner_rows = build_compact_viewpoint_rows(planner_rows)
    elif args.format == "hic-compact-json":
        captioner_rows = build_hic_compact_json_rows(planner_rows)
    else:
        captioner_rows = build_captioner_rows(planner_rows)
    write_jsonl(args.output_jsonl, captioner_rows)
    print(f"[bridge] saved {len(captioner_rows)} image+generated-plan rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
