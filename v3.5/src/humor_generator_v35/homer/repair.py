"""Validator-feedback repair for otherwise valid HOMER generations.

The recovery turn is deliberately appended to the original HOMER conversation:
it may repair serialization or make an already stated opposition explicit, but
must not introduce, remove, paraphrase, or reorder semantic content.
"""
from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from .contracts import parse_associations, parse_conflicts
from .prompts import text_part


REPAIR_POLICY_VERSION = "validator-feedback-format-only-v1"

REPAIR_SYSTEM = (
    "Repair only the schema or explicit delimiter of your immediately preceding answer. "
    "Preserve every stated semantic item, entity, association step, and conflict side verbatim. "
    "Do not add, remove, merge, reorder, paraphrase, or infer content. Output only the repaired "
    "answer, with no explanation."
)

CHANNEL_REQUIREMENTS = {
    "conflict": (
        'Return a JSON list of objects with exactly the keys "left" and "right". '
        "Each object must encode one opposition already present in the preceding answer."
    ),
    "local": (
        "Return one JSON object mapping each previously stated root entity to exactly three "
        "previously stated association-step strings."
    ),
    "global": (
        "Return one JSON object mapping each previously stated root entity to exactly three "
        "previously stated association-step strings."
    ),
}

_NUMBERED_ITEM = re.compile(
    r"(?:^|\s)(?:\d+\s*[.)]|[-*])\s*(.*?)(?=(?:\s+\d+\s*[.)]|\s+[-*])\s|$)",
    re.S,
)
_SCHEMA_KEYS = {"entity", "associations", "imaginations", "left", "right", "conflicts"}


def _normalized(value: str) -> str:
    return " ".join(value.split()).strip(" \t\r\n,;[]{}\"").casefold()


def _association_semantics(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, str):
        result.append(_normalized(value))
    elif isinstance(value, list):
        for item in value:
            result.extend(_association_semantics(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() not in _SCHEMA_KEYS:
                result.append(_normalized(key))
            result.extend(_association_semantics(item))
    return [item for item in result if item]


def assert_lossless_repair(invalid_output: str, repaired_output: str, *, channel: str) -> None:
    """Reject repairs that change content rather than serialization.

    Association outputs must preserve the exact multiset of semantic strings.
    Conflict repair may only insert an explicit opposition delimiter between
    two verbatim substrings of each original numbered item.
    """
    if channel in {"local", "global"}:
        try:
            before = json.loads(invalid_output.strip().removeprefix("```json").removesuffix("```").strip())
            after = json.loads(repaired_output.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError as exc:
            raise ValueError("lossless association repair requires JSON on both sides") from exc
        if Counter(_association_semantics(before)) != Counter(_association_semantics(after)):
            raise ValueError("repair changed association semantic strings")
        parse_associations(repaired_output, view=channel)
        return
    if channel != "conflict":
        raise ValueError(f"unsupported repair channel: {channel}")
    raw_items = [match.group(1).strip(" .") for match in _NUMBERED_ITEM.finditer(invalid_output.strip())]
    if not raw_items:
        raw_items = [invalid_output.strip()]
    pairs = parse_conflicts(repaired_output)
    if len(raw_items) != len(pairs):
        raise ValueError("repair changed the number of conflict items")
    unused = raw_items.copy()
    for pair in pairs:
        left, right = _normalized(pair.left), _normalized(pair.right)
        match_index = next(
            (
                index for index, item in enumerate(unused)
                if left in _normalized(item) and right in _normalized(item)
            ),
            None,
        )
        if match_index is None:
            raise ValueError("repair added or paraphrased a conflict side")
        unused.pop(match_index)


def validator_feedback_messages(
    original_messages: list[dict[str, Any]],
    *,
    invalid_output: str,
    validation_error: str,
    channel: str,
) -> list[dict[str, Any]]:
    """Append one constrained repair turn to the unchanged HOMER prompt."""
    if channel not in CHANNEL_REQUIREMENTS:
        raise ValueError(f"unsupported repair channel: {channel}")
    feedback = (
        f"{REPAIR_SYSTEM}\n\nValidator error: {validation_error}\n"
        f"Required schema: {CHANNEL_REQUIREMENTS[channel]}"
    )
    return [
        *original_messages,
        {"role": "assistant", "content": [text_part(invalid_output)]},
        {"role": "user", "content": [text_part(feedback)]},
    ]
