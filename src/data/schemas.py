from dataclasses import dataclass
from typing import Any


@dataclass
class SFTExample:
    image: str
    image_id: str
    messages: list[dict[str, Any]]
    meta: dict[str, Any]


@dataclass
class CandidateExample:
    image: str
    image_id: str
    candidates: list[str]
    meta: dict[str, Any]


@dataclass
class RankedCaption:
    caption: str
    scores: dict[str, float]
    total_score: float
