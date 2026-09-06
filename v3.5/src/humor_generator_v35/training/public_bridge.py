"""HOMER public-code context and prompt contract for the latent bridge.

The historical bridge trainer intentionally remains available for reproducing
old v3.5 artifacts.  This module is the separate current route: it consumes a
source-aware, adapter-free Planner trace plus the cached HOMER
summary/retrieval/selection context and uses the pinned public caption prompt
for both the text teacher and the latent-plan student.

The bridge is an alternative communication channel for the *same* selected
HOMER plan.  It must not become a generic caption SFT task with a free-floating
latent prefix.  The context cache therefore records the selected conflict,
selected entities and sampled DFS paths, while the trace cache retains the raw
Planner responses used to produce that context.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ..homer.official_prompts import caption_messages
PUBLIC_BRIDGE_PROMPT_TRACK = "public_code_exact_caption_latent_plan_replacement_v1"
REQUIRED_CONTEXT_FIELDS = {
    "schema_version",
    "data_version",
    "cluster_id",
    "contest_number",
    "image",
    "image_sha256",
    "standard_description",
    "summary_imagination",
    "summary_raw",
    "retrieved_imagination",
    "selected_conflict",
    "selection_raw",
    "selected_entities",
    "selected_paths",
    "free_association_text",
    "trace_path",
    "trace_sha256",
    "planner_outputs_sha256",
    "call_log",
    "selection_policy",
    "context_prompt_track",
    "provenance",
}


@dataclass(frozen=True)
class PublicHomerContext:
    """The immutable post-trace context consumed by a bridge training row."""

    schema_version: int
    data_version: str
    cluster_id: str
    contest_number: int
    image: str
    image_sha256: str
    standard_description: str
    summary_imagination: dict[str, Any]
    summary_raw: str
    retrieved_imagination: dict[str, Any]
    selected_conflict: str
    selection_raw: dict[str, str]
    selected_entities: tuple[str, ...]
    selected_paths: dict[str, tuple[str, ...]]
    free_association_text: str
    trace_path: str
    trace_sha256: str
    planner_outputs_sha256: str
    call_log: tuple[dict[str, Any], ...]
    selection_policy: str
    context_prompt_track: str
    provenance: dict[str, Any]


def _as_context(value: Mapping[str, Any]) -> PublicHomerContext:
    missing = sorted(REQUIRED_CONTEXT_FIELDS - set(value))
    if missing:
        raise ValueError(f"HOMER context is missing fields: {missing}")
    entities = value["selected_entities"]
    if not isinstance(entities, list) or len(entities) != 2:
        raise ValueError("HOMER context must select exactly two entities")
    if any(not isinstance(item, str) or not item.strip() for item in entities):
        raise ValueError("selected_entities must contain non-empty strings")
    paths_value = value["selected_paths"]
    if not isinstance(paths_value, dict) or set(paths_value) != set(entities):
        raise ValueError("selected_paths must cover exactly the selected entities")
    paths: dict[str, tuple[str, ...]] = {}
    for entity in entities:
        raw = paths_value[entity]
        if not isinstance(raw, list) or len(raw) < 2:
            raise ValueError(f"selected path for {entity!r} is empty or too short")
        if any(not isinstance(item, str) or not item.strip() for item in raw):
            raise ValueError(f"selected path for {entity!r} contains a non-string node")
        if raw[0].casefold() != entity.casefold():
            raise ValueError(f"selected path for {entity!r} does not start at its root")
        paths[entity] = tuple(raw)
    if not isinstance(value["summary_imagination"], dict) or not value["summary_imagination"]:
        raise ValueError("summary_imagination must be a non-empty object")
    if not isinstance(value["summary_raw"], str) or not value["summary_raw"].strip():
        raise ValueError("summary_raw must preserve the non-empty Planner response")
    if not isinstance(value["retrieved_imagination"], dict) or not value["retrieved_imagination"]:
        raise ValueError("retrieved_imagination must be a non-empty object")
    selection_raw = value["selection_raw"]
    if (
        not isinstance(selection_raw, dict)
        or set(selection_raw) != {"conflict", "entities"}
        or any(not isinstance(item, str) or not item.strip() for item in selection_raw.values())
    ):
        raise ValueError("selection_raw must preserve conflict and entity selection responses")
    for name in ("trace_path", "trace_sha256", "planner_outputs_sha256", "selection_policy"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError(f"HOMER context field {name} is empty")
    if not re.fullmatch(r"[0-9a-f]{64}", str(value["trace_sha256"])) or not re.fullmatch(
        r"[0-9a-f]{64}", str(value["planner_outputs_sha256"])
    ):
        raise ValueError("HOMER context trace/output hashes must be SHA-256 values")
    if (
        not isinstance(value["call_log"], list)
        or not value["call_log"]
        or any(not isinstance(item, dict) for item in value["call_log"])
    ):
        raise ValueError("HOMER context call_log is missing")
    if not isinstance(value["provenance"], dict) or not value["provenance"]:
        raise ValueError("HOMER context provenance is missing")
    return PublicHomerContext(
        schema_version=int(value["schema_version"]),
        data_version=str(value["data_version"]),
        cluster_id=str(value["cluster_id"]),
        contest_number=int(value["contest_number"]),
        image=str(value["image"]),
        image_sha256=str(value["image_sha256"]),
        standard_description=str(value["standard_description"]),
        summary_imagination=dict(value["summary_imagination"]),
        summary_raw=str(value["summary_raw"]),
        retrieved_imagination=dict(value["retrieved_imagination"]),
        selected_conflict=str(value["selected_conflict"]),
        selection_raw={str(key): str(item) for key, item in selection_raw.items()},
        selected_entities=tuple(str(item) for item in entities),
        selected_paths=paths,
        free_association_text=str(value["free_association_text"]),
        trace_path=str(value["trace_path"]),
        trace_sha256=str(value["trace_sha256"]),
        planner_outputs_sha256=str(value["planner_outputs_sha256"]),
        call_log=tuple(dict(item) for item in value["call_log"]),
        selection_policy=str(value["selection_policy"]),
        context_prompt_track=str(value["context_prompt_track"]),
        provenance=dict(value["provenance"]),
    )


def load_context_index(path: Path) -> dict[str, PublicHomerContext]:
    """Load and validate the immutable post-trace context index."""
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, PublicHomerContext] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            context = _as_context(value)
            assert_context_path_consistent(context)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid HOMER context at {path}:{line_number}: {exc}") from exc
        if context.cluster_id in result:
            raise ValueError(f"duplicate HOMER context cluster: {context.cluster_id}")
        result[context.cluster_id] = context
    if not result:
        raise ValueError(f"empty HOMER context index: {path}")
    return result


def context_as_dict(context: PublicHomerContext) -> dict[str, Any]:
    """Return a JSON-safe representation for provenance and tests."""
    return {
        "schema_version": context.schema_version,
        "data_version": context.data_version,
        "cluster_id": context.cluster_id,
        "contest_number": context.contest_number,
        "image": context.image,
        "image_sha256": context.image_sha256,
        "standard_description": context.standard_description,
        "summary_imagination": context.summary_imagination,
        "summary_raw": context.summary_raw,
        "retrieved_imagination": context.retrieved_imagination,
        "selected_conflict": context.selected_conflict,
        "selection_raw": context.selection_raw,
        "selected_entities": list(context.selected_entities),
        "selected_paths": {key: list(value) for key, value in context.selected_paths.items()},
        "free_association_text": context.free_association_text,
        "trace_path": context.trace_path,
        "trace_sha256": context.trace_sha256,
        "planner_outputs_sha256": context.planner_outputs_sha256,
        "call_log": list(context.call_log),
        "selection_policy": context.selection_policy,
        "context_prompt_track": context.context_prompt_track,
        "provenance": context.provenance,
    }


def public_text_caption_messages(context: PublicHomerContext) -> list[dict[str, Any]]:
    """Official HOMER caption request carrying the selected text plan."""
    return caption_messages(
        context.standard_description,
        context.selected_conflict,
        context.free_association_text,
    )


def public_latent_caption_messages(context: PublicHomerContext) -> list[dict[str, Any]]:
    """Official caption request with the plan replaced by bridge memory.

    The exact public system prompt and two user-block serialization are kept.
    The conflict/path fields are empty only in the student request; the bridge
    receives the same cached Planner trace and selected context provenance.
    This prevents a second textual copy of the plan from leaking into the
    latent condition.
    """
    return caption_messages(context.standard_description, "", "")


def validate_row_context(row: Mapping[str, Any], context: PublicHomerContext) -> None:
    """Reject image/description mismatches before a receiver forward."""
    for field in ("cluster_id", "contest_number", "image", "image_sha256", "standard_description"):
        row_value = str(row.get(field, ""))
        context_value = str(getattr(context, field))
        if row_value != context_value:
            raise ValueError(
                f"HOMER context mismatch for {field}: row={row_value!r}, context={context_value!r}"
            )


def target_caption_text(row: Mapping[str, Any]) -> str:
    """Return the unmodified human caption used for teacher forcing.

    Source ranking rows contain captions but no gold explanations.  We do not
    fabricate an explanation.  The official caption prompt still requires the
    ``##Caption``/``##Explanation`` contract at inference; generated blobs are
    parsed with ``parse_caption_blob``.  The target policy is recorded as
    ``source_caption_only`` in the current bridge manifest.
    """
    caption = str(row.get("caption", "")).strip()
    if not caption:
        raise ValueError(f"bridge row {row.get('row_id')} has an empty caption")
    return caption


def selected_path_text(context: PublicHomerContext) -> str:
    """Rebuild the exact free-association block used by the public generator."""
    return "\n".join(
        f"The free-association chain of {entity} is {' -> '.join(context.selected_paths[entity])}"
        for entity in context.selected_entities
    )


def assert_context_path_consistent(context: PublicHomerContext) -> None:
    if context.free_association_text != selected_path_text(context):
        raise ValueError(f"free_association_text mismatch for {context.cluster_id}")
    if context.context_prompt_track != PUBLIC_BRIDGE_PROMPT_TRACK:
        raise ValueError(
            f"unexpected HOMER bridge context prompt track: {context.context_prompt_track}"
        )
