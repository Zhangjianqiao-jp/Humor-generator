#!/usr/bin/env python
"""Production v5: concrete two-pass cues with low-information filtering."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_guided_sft_teacher_pipeline_v4 as v4

core = v4.core
v3 = v4.v3
pipeline = v3.pipeline


CONCRETE_CUE_PROMPT = """Inspect the attached image for at most one clearly visible incongruity.

Conservative literal description:
{description}

Allowed types are: unusual physical detail, obvious size difference, contrasting visible
actions or poses, composition/overlap relationship, or visible subject/object placement mismatch.

Return only valid JSON:
{{
  "humor_cue": "one concrete neutral visual observation, or an empty string"
}}

The cue must explicitly name the visible subjects or objects and state how they differ,
overlap, contrast, or are positioned. Never answer with a category label such as
"a size difference", "an unusual physical detail", "action contrast", or
"a composition relationship".

Strict rules:
- This is a factual visual observation, not a joke or final caption.
- Mention only clearly visible objects, positions, sizes, actions, poses, or composition.
- Do not guess identity, age, profession, relationship, emotion, intention, cause,
  motivation, thought, or hidden story.
- Do not use metaphor, fictional characters, pop culture, or "looks like/as if".
- Do not infer stealing, guarding, escaping, pretending, planning, helping, competing,
  driving, speaking, or trying to do something.
- If no concrete statement can be written safely, return an empty string.

Valid: "The miniature meal is much smaller than the hand holding it."
Valid: "A dog is positioned behind a car steering wheel."
Valid: "One person is standing while the surrounding people are crouching."
Invalid: "A large and obvious size difference."
Invalid: "The dog is driving to work."

Return JSON only."""


LOW_INFORMATION_PATTERNS = (
    r"^(?:a |an |the )?(?:clearly )?(?:unusual )?physical detail$",
    r"^(?:a |an |the )?(?:large |clear |obvious |large and obvious )?size difference$",
    r"^(?:a |an |the )?(?:visible )?action contrast$",
    r"^(?:a |an |the )?(?:visible )?(?:composition|compositional) relationship$",
    r"^(?:a |an |the )?(?:visible )?(?:object|subject)[/-]?(?:role|placement) mismatch$",
)

_safe_clean_cue = core.clean_cue


def concrete_clean_cue(value: Any) -> tuple[str, str | None]:
    cue, rejected = _safe_clean_cue(value)
    if not cue:
        return cue, rejected
    normalized = re.sub(r"\s+", " ", cue.strip().lower()).rstrip(".")
    if any(re.fullmatch(pattern, normalized) for pattern in LOW_INFORMATION_PATTERNS):
        return "", "low-information category label"
    # A concrete observation should contain enough material to identify both
    # the visible entity and its relation/detail.
    words = re.findall(r"\b[\w'-]+\b", normalized)
    if len(words) < 6:
        return "", "cue too short to be concrete"
    return cue, rejected


v3.CUE_RECHECK_PROMPT = CONCRETE_CUE_PROMPT
core.clean_cue = concrete_clean_cue


if __name__ == "__main__":
    pipeline.run_all()
