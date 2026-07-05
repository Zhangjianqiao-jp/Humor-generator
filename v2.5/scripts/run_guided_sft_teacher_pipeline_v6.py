#!/usr/bin/env python
"""Production v6: concrete cues with stricter speculative/ordinary filtering."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_guided_sft_teacher_pipeline_v5 as v5

core = v5.core
pipeline = v5.pipeline

_v5_clean_cue = core.clean_cue


def final_clean_cue(value: Any) -> tuple[str, str | None]:
    raw = " ".join(str(value or "").split())
    if re.search(r"\b(?:suggests?|suggesting)\b", raw, flags=re.IGNORECASE):
        return "", "speculation"
    cue, rejected = _v5_clean_cue(value)
    if not cue:
        return cue, rejected
    # Remove the model's subjective evaluation while preserving the visible
    # relation that precedes it.
    cue = re.split(
        r",?\s+which (?:is|are) (?:an? )?(?:unusual|unexpected|odd|strange)\b",
        cue,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip().rstrip(".,")
    if re.search(
        r"\b(?:cigarette|cigar|pipe)\b.*\b(?:mouth|lips)\b",
        cue,
        flags=re.IGNORECASE,
    ):
        return "", "ordinary object placement"
    words = re.findall(r"\b[\w'-]+\b", cue)
    if len(words) < 6:
        return "", "cue too short after factual cleanup"
    return cue, None


core.clean_cue = final_clean_cue


if __name__ == "__main__":
    pipeline.run_all()
