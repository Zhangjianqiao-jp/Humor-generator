"""Strict parsing extension; the pinned official prompt module is immutable."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class CaptionParseResult:
    caption: str
    explanation: str
    status: str
    version: str = "strict_caption_v1"


def parse_caption_strict(text: str) -> CaptionParseResult:
    """Separate malformed output from captions; retain the official parser above.

    Known refusal detection is deliberately narrow, not a semantic classifier.
    Callers retain raw output and keep failed items in the evaluation denominator.
    """
    refusal = re.compile(r"^(?:i[’']m sorry,? but |i am sorry,? but )?i (?:cannot|can't) answer this question\.?$", re.I)
    if refusal.fullmatch(text.strip()):
        return CaptionParseResult("", "", "known_refusal")
    markers = list(re.finditer(r"(?:^|\n|(?=##))\s*(?:#{1,3}\s*)?(Caption|Explanation)\s*:\s*", text, re.I))
    if [m.group(1).lower() for m in markers] != ["caption", "explanation"]:
        return CaptionParseResult("", "", "invalid_markers")
    if text[:markers[0].start()].strip():
        return CaptionParseResult("", "", "unexpected_preamble")
    caption = text[markers[0].end():markers[1].start()].strip()
    explanation = text[markers[1].end():].strip()
    if refusal.fullmatch(caption):
        return CaptionParseResult("", explanation, "known_refusal")
    status = "ok" if caption and explanation else "empty_section"
    return CaptionParseResult(caption, explanation, status)
