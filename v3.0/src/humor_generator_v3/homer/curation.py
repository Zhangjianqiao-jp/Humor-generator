"""Auditable joke-corpus curation disclosed in HOMER Appendix B.1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .retrieval import words


@dataclass(frozen=True)
class JokeRecord:
    text: str
    source: str
    rating: float | None = None


@dataclass(frozen=True)
class CurationReport:
    input_rows: int
    rating_removed: int
    empty_removed: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    output_rows: int
    overlap_definition: str


def english_word_overlap(left: str, right: str) -> float:
    """Overlap coefficient used for the paper's underspecified '80% shared words'."""
    a, b = set(words(left)), set(words(right))
    denominator = min(len(a), len(b))
    return len(a & b) / denominator if denominator else 0.0


def curate_jokes(
    records: Iterable[JokeRecord],
    *,
    rating_threshold: float = 3.0,
    near_duplicate_threshold: float = 0.80,
) -> tuple[list[JokeRecord], CurationReport]:
    rows = list(records)
    rating_removed = empty_removed = exact_removed = near_removed = 0
    candidates: list[JokeRecord] = []
    exact: dict[str, int] = {}
    for row in rows:
        text = " ".join(row.text.split()).strip()
        if not text:
            empty_removed += 1
            continue
        if row.rating is not None and row.rating < rating_threshold:
            rating_removed += 1
            continue
        key = text.casefold()
        normalized = JokeRecord(text, row.source, row.rating)
        if key in exact:
            exact_removed += 1
            old = candidates[exact[key]]
            if len(text) > len(old.text):
                candidates[exact[key]] = normalized
            continue
        exact[key] = len(candidates)
        candidates.append(normalized)

    # Exact but quadratic reference implementation.  Formal 335k construction
    # must use a verified indexed implementation and match this on a test shard.
    kept: list[JokeRecord] = []
    for candidate in sorted(candidates, key=lambda row: (-len(row.text), row.text.casefold())):
        if any(english_word_overlap(candidate.text, previous.text) > near_duplicate_threshold for previous in kept):
            near_removed += 1
            continue
        kept.append(candidate)
    kept.sort(key=lambda row: (row.source, row.text.casefold()))
    return kept, CurationReport(
        input_rows=len(rows), rating_removed=rating_removed, empty_removed=empty_removed,
        exact_duplicates_removed=exact_removed, near_duplicates_removed=near_removed,
        output_rows=len(kept), overlap_definition="|unique-word intersection| / min(unique-word counts)",
    )
