#!/usr/bin/env python
"""Verify objective structural invariants of planner/captioner generation JSONL."""

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

from src.utils.io import read_jsonl

PLANNER_PREFIXES = ("ANCHOR:", "CONTRAST:", "ANGLE:")
PLANNER_PLACEHOLDERS = (
    "key visible object or action",
    "violated expectation",
    "concise punchline direction",
)
HIC_VIEWPOINT_KEYS = {
    "literal_image_description",
    "gold_joke_explanation",
    "humor_type",
    "humor_point",
    "visual_anchors",
    "required_viewpoints",
    "primary_viewpoint",
    "needs_external_knowledge",
    "confidence",
    "uncertainty",
}
COMPACT_VIEWPOINT_KEYS = (
    "scene",
    "type",
    "target",
    "primary_view",
    "views",
    "anchors",
    "external_knowledge",
)
ALLOWED_VIEWPOINTS = {
    "full_image",
    "object_crop",
    "relation_crop",
    "scale_reference_crop",
    "face_expression_crop",
    "text_region_crop",
    "foreground_background_view",
    "context_scene_view",
    "pose_action_view",
}


def parse_hic_viewpoint(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"HIC viewpoint output is not valid JSON: {exc}") from exc
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as nested_exc:
            raise ValueError(f"HIC viewpoint output is not valid JSON: {nested_exc}") from nested_exc
    if not isinstance(value, dict):
        raise ValueError("HIC viewpoint output must be a JSON object.")
    if set(value) != HIC_VIEWPOINT_KEYS:
        raise ValueError(
            "HIC viewpoint output has the wrong top-level schema: "
            f"expected {sorted(HIC_VIEWPOINT_KEYS)}, got {sorted(value)}"
        )
    if not isinstance(value["visual_anchors"], list):
        raise ValueError("visual_anchors must be a JSON array.")
    if not isinstance(value["required_viewpoints"], list):
        raise ValueError("required_viewpoints must be a JSON array.")
    if not isinstance(value["needs_external_knowledge"], bool):
        raise ValueError("needs_external_knowledge must be a JSON boolean.")
    return value


def parse_compact_viewpoint(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"Compact viewpoint output is not valid JSON: {exc}") from exc
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as nested_exc:
            raise ValueError(f"Compact viewpoint output is not valid JSON: {nested_exc}") from nested_exc
    if not isinstance(value, dict) or tuple(value) != COMPACT_VIEWPOINT_KEYS:
        actual = tuple(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"Wrong compact viewpoint schema: expected {COMPACT_VIEWPOINT_KEYS}, got {actual}")
    for key in ("scene", "type", "target", "primary_view"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"Compact viewpoint field {key} must be a non-empty string.")
    views = value["views"]
    if not isinstance(views, list) or not 1 <= len(views) <= 4:
        raise ValueError("Compact viewpoint views must contain 1 to 4 values.")
    if any(view not in ALLOWED_VIEWPOINTS for view in views):
        raise ValueError(f"Compact viewpoint contains an unsupported view: {views}")
    if value["primary_view"] not in views:
        raise ValueError("Compact viewpoint primary_view must occur in views.")
    anchors = value["anchors"]
    if not isinstance(anchors, list) or not 1 <= len(anchors) <= 4:
        raise ValueError("Compact viewpoint anchors must contain 1 to 4 values.")
    for anchor in anchors:
        if not isinstance(anchor, dict) or tuple(anchor) != ("label", "evidence", "role"):
            raise ValueError("Each compact viewpoint anchor must contain label, evidence, role in order.")
        if any(not isinstance(anchor[key], str) or not anchor[key].strip() for key in anchor):
            raise ValueError("Compact viewpoint anchor values must be non-empty strings.")
    if not isinstance(value["external_knowledge"], bool):
        raise ValueError("Compact viewpoint external_knowledge must be a JSON boolean.")
    return value
def texts_from_row(row: dict[str, Any]) -> list[str]:
    candidates = row.get("candidates")
    if isinstance(candidates, list):
        return [str(value).strip() for value in candidates]
    if "generated_caption" in row:
        return [str(row["generated_caption"]).strip()]
    raise ValueError("Generation row has neither candidates nor generated_caption.")


def verify_generation_rows(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    if kind not in {"planner", "hic-viewpoint", "compact-viewpoint", "captioner"}:
        raise ValueError(f"Unsupported generation kind: {kind}")
    if not rows:
        raise ValueError("Generation file contains zero rows.")

    seen_images: set[str] = set()
    candidate_count = 0
    for row_index, row in enumerate(rows):
        image_id = str(row.get("image_id") or "").strip()
        if not image_id:
            raise ValueError(f"Row {row_index} has no image_id.")
        if image_id in seen_images:
            raise ValueError(f"Duplicate image_id in evaluation rows: {image_id}")
        seen_images.add(image_id)
        prompt = str(row.get("prompt") or "").strip()
        for candidate_index, text in enumerate(texts_from_row(row)):
            candidate_count += 1
            if not text:
                raise ValueError(f"Row {row_index} candidate {candidate_index} is empty.")
            if prompt and prompt in text:
                raise ValueError(f"Row {row_index} candidate {candidate_index} leaks the prompt.")
            if kind == "planner":
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if len(lines) != 3 or tuple(line.startswith(prefix) for line, prefix in zip(lines, PLANNER_PREFIXES)) != (True, True, True):
                    raise ValueError(
                        f"Row {row_index} candidate {candidate_index} violates the three-line planner schema: {text!r}"
                    )
                values = [line[len(prefix) :].strip() for line, prefix in zip(lines, PLANNER_PREFIXES)]
                if any(not value for value in values):
                    raise ValueError(f"Row {row_index} candidate {candidate_index} has an empty planner field.")
                if any(value.casefold() == placeholder for value, placeholder in zip(values, PLANNER_PLACEHOLDERS)):
                    raise ValueError(
                        f"Row {row_index} candidate {candidate_index} echoes a planner placeholder: {text!r}"
                    )
            elif kind == "hic-viewpoint":
                parse_hic_viewpoint(text)
            elif kind == "compact-viewpoint":
                parse_compact_viewpoint(text)
            elif "\n" in text or "\r" in text:
                raise ValueError(f"Row {row_index} candidate {candidate_index} is not a single-line caption.")

    return {
        "kind": kind,
        "rows": len(rows),
        "unique_images": len(seen_images),
        "candidates": candidate_count,
        "schema_valid": True,
        "prompt_leakage": 0,
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("planner", "hic-viewpoint", "compact-viewpoint", "captioner"),
        required=True,
    )
    args = parser.parse_args()
    report = verify_generation_rows(read_jsonl(args.input_jsonl), args.kind)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
