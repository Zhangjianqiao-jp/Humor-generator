from __future__ import annotations

import re

GENERIC_PATTERNS = ["when you realize", "this is fine", "pov"]
UNSAFE_WORDS = ["kill", "hate", "slur"]


class HeuristicJudge:
    def __init__(self, generic_patterns: list[str] | None = None, unsafe_words: list[str] | None = None):
        self.generic_patterns = generic_patterns or GENERIC_PATTERNS
        self.unsafe_words = unsafe_words or UNSAFE_WORDS

    def score_caption(self, caption: str, image_description: str | None = None) -> dict[str, float]:
        c = caption.strip()
        lower = c.lower()
        ir = 5.0 if not image_description else 6.0
        hu = 6.0 if re.search(r"\b(when|plot twist|somehow|nobody|chaos)\b", lower) else 4.5
        sp = 6.0
        if any(p in lower for p in self.generic_patterns):
            sp = 3.0
        ra = 7.0
        if len(c) < 5 or len(c) > 220:
            ra = 2.0
        cr = 6.0 if len(set(lower.split())) > 6 else 4.0
        sa = 8.0
        if any(w in lower for w in self.unsafe_words):
            sa = 2.0
        return {"IR": ir, "HU": hu, "SP": sp, "RA": ra, "CR": cr, "SA": sa}


class ExternalJudge:
    def score_caption(self, caption: str, image_description: str | None = None) -> dict[str, float]:
        del caption, image_description
        raise NotImplementedError("External judge API not implemented in V1.")
