#!/usr/bin/env python
"""Production entry point: two-pass cues, boolean audit, tolerant empty cues."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_guided_sft_pipeline as core
import scripts.run_guided_sft_teacher_pipeline_v3 as v3


_strict_extract_json = core.extract_json


def tolerant_extract_json(text: str) -> dict:
    try:
        return _strict_extract_json(text)
    except (ValueError, json.JSONDecodeError):
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        if not cleaned:
            return {"humor_cue": ""}
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            raise
        if isinstance(value, str):
            return {"humor_cue": value}
        raise ValueError("Expected a JSON object")


core.extract_json = tolerant_extract_json


if __name__ == "__main__":
    v3.pipeline.run_all()
