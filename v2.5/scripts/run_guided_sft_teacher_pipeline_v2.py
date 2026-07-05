#!/usr/bin/env python
"""Boolean-audit entry point for the guided teacher-data pipeline.

Qwen2.5-VL copied numeric placeholder scores from the first audit schema during
the end-to-end smoke test. Boolean decisions avoid that schema-copy failure
while retaining the same strict all-dimensions-must-pass gate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_guided_sft_teacher_pipeline as pipeline
from scripts.run_guided_sft_pipeline import normalize_space


BOOLEAN_AUDIT_TEMPLATE = """You are selecting a high-quality supervised target for humorous image-caption training.

Judge the candidates against the attached image. Do not rewrite them.
Reject unsupported visual facts, guessed identity/profession/relationship/emotion/intention,
generic dialogue, literal non-jokes, broken language, explanations, and multi-part text.

Select the best candidate. Then make each quality decision independently.

Candidates:
{candidates}

Return only valid JSON:
{{
  "best_index": 1,
  "visually_grounded": true,
  "natural": true,
  "humorous": true,
  "format_ok": true,
  "overall_good": true,
  "reason": "brief reason"
}}

Indices are 1-{n}. Use false whenever uncertain. Do not copy the example
booleans automatically; inspect the image and selected caption first."""


def as_bool(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def boolean_audit_result(parsed: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    try:
        best_index = int(parsed.get("best_index", 0))
    except (TypeError, ValueError):
        best_index = 0
    decisions = {
        "visually_grounded": as_bool(parsed.get("visually_grounded")),
        "natural": as_bool(parsed.get("natural")),
        "humorous": as_bool(parsed.get("humorous")),
        "format_ok": as_bool(parsed.get("format_ok")),
        "overall_good": as_bool(parsed.get("overall_good")),
    }
    valid_index = 1 <= best_index <= len(candidates)
    caption = candidates[best_index - 1].strip() if valid_index else ""
    words = re.findall(r"\b[\w'-]+\b", caption)
    mechanical_ok = (
        bool(caption)
        and 3 <= len(words) <= 20
        and len(caption) <= 140
        and "\n" not in caption
        and ";" not in caption
    )
    passed = valid_index and mechanical_ok and all(decisions.values())
    # Preserve the v1 metadata shape consumed by materialization and summaries.
    scores = {
        "visual_grounding": 4 if decisions["visually_grounded"] else 1,
        "naturalness": 4 if decisions["natural"] else 1,
        "humor": 4 if decisions["humorous"] else 1,
        "format": 4 if decisions["format_ok"] else 1,
        "overall": 4 if decisions["overall_good"] else 1,
    }
    return {
        "best_index": best_index,
        "caption": caption,
        "scores": scores,
        "decisions": decisions,
        "mechanical_ok": mechanical_ok,
        "passed": passed,
        "reason": normalize_space(parsed.get("reason"))[:200],
    }


pipeline.TARGET_AUDIT_TEMPLATE = BOOLEAN_AUDIT_TEMPLATE
pipeline.audit_result = boolean_audit_result


if __name__ == "__main__":
    pipeline.run_all()
