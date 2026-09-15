"""Validator-feedback repair for otherwise valid HOMER generations.

The recovery turn is deliberately appended to the original HOMER conversation:
it may repair serialization or make an already stated opposition explicit, but
must not introduce, remove, paraphrase, or reorder semantic content.
"""
from __future__ import annotations

import ast
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
    "summary": (
        "Return one JSON object. Preserve every root key and every value string from the "
        "preceding answer verbatim; only normalize its serialization so each value is a "
        "non-empty list of strings. Do not add, remove, merge, reorder, paraphrase, or infer."
    ),
    "entities": (
        "Return exactly [entity1, entity2], choosing only names from the supplied entity list. "
        "A name may be normalized to the unique supplied key it explicitly refers to; do not "
        "invent, add, remove, or paraphrase an entity."
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


def _jsonish(value: str) -> Any:
    """Parse JSON, or a Python-literal serialization, without executing code."""
    candidate = value.strip()
    # Qwen occasionally emits the JSON opening fence but truncates the final
    # closing fence at the generation boundary.  Treat that delimiter-only
    # defect as recoverable: the JSON payload itself remains the semantic
    # source and the validator-feedback turn may add the missing delimiter.
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.I | re.S)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        opening = re.match(r"^```(?:json)?(?:\s+|\s*$)", candidate, flags=re.I | re.S)
        if opening:
            candidate = candidate[opening.end():].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError, TypeError) as exc:
            raise ValueError("output is not losslessly parseable JSON") from exc


def _summary_semantics(value: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Canonicalize summary content while preserving keys, values, and order."""
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list) and value and all(isinstance(item, dict) and len(item) == 1 for item in value):
        items = []
        for item in value:
            items.extend(item.items())
    else:
        raise ValueError("summary semantic source must be an object or one-key object list")
    result: list[tuple[str, tuple[str, ...]]] = []
    for key, raw_values in items:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("summary root key is not a string")
        values = [raw_values] if isinstance(raw_values, str) else raw_values
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("summary values are not a non-empty string sequence")
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError("summary contains a non-string value")
        result.append((_normalized(key), tuple(_normalized(item) for item in values)))
    return tuple(result)


def assert_lossless_summary_repair(invalid_output: str, repaired_output: str) -> None:
    """Allow only a serialization repair of a summary response.

    The repaired response must be a strict JSON object whose per-root value is
    a list of strings.  Comparing the canonical ordered key/value sequence to
    the original prevents a repair model from inventing or paraphrasing
    imagination content.
    """
    before = _summary_semantics(_jsonish(invalid_output))
    after_value = _jsonish(repaired_output)
    if not isinstance(after_value, dict) or not after_value:
        raise ValueError("repaired summary must be a non-empty JSON object")
    after = _summary_semantics(after_value)
    if before != after:
        raise ValueError("repair changed summary semantic strings")
    # A valid repair must be JSON (not merely a Python literal) and list-valued.
    try:
        parsed = json.loads(repaired_output.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError as exc:
        raise ValueError("repaired summary must be strict JSON") from exc
    if not isinstance(parsed, dict) or any(
        not isinstance(key, str) or not isinstance(values, list) or not values
        or any(not isinstance(item, str) or not item.strip() for item in values)
        for key, values in parsed.items()
    ):
        raise ValueError("repaired summary values must be non-empty lists of strings")


def _entity_candidates(value: str) -> list[str]:
    matches = re.findall(r"\[([^\]]*)\]", value, flags=re.S)
    if not matches:
        raise ValueError("entity response has no bracketed list")
    values = [item.strip().strip("\\\"'") for item in matches[-1].split(",") if item.strip().strip("\\\"'")]
    return values


def _entity_tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w]+", value.casefold()))


def assert_reference_only_entity_repair(
    invalid_output: str,
    repaired_output: str,
    *,
    available: list[str],
) -> None:
    """Validate entity repair as unique reference normalization only.

    This permits e.g. ``Dragon`` → ``Dragon holding flowers`` when that key is
    the unique available reference, but rejects invented entities, paraphrases,
    duplicate selections, and ambiguous mappings.
    """
    before = _entity_candidates(invalid_output)
    after = _entity_candidates(repaired_output)
    if len(before) != 2 or len(after) != 2:
        raise ValueError("entity repair must preserve exactly two candidates")
    folded = {str(item).casefold(): str(item) for item in available}
    mapped: list[str] = []
    for candidate in before:
        exact = folded.get(candidate.casefold())
        if exact is not None:
            matches = [exact]
        else:
            tokens = _entity_tokens(candidate)
            matches = [
                item for item in available
                if tokens and (
                    tokens.issubset(_entity_tokens(item))
                    or _entity_tokens(item).issubset(tokens)
                )
            ]
        if len(matches) != 1:
            raise ValueError(f"entity reference is not uniquely repairable: {candidate!r}")
        mapped.append(matches[0])
    if len({item.casefold() for item in mapped}) != 2:
        raise ValueError("entity repair cannot make duplicate references distinct")
    if [item.casefold() for item in after] != [item.casefold() for item in mapped]:
        raise ValueError("repair changed or reordered entity references")
    if [item for item in after if item.casefold() not in folded] or len(set(after)) != 2:
        raise ValueError("repaired entities must be distinct supplied keys")


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
