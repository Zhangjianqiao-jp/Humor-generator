"""HOMER public-code call graph with a frozen local model interface.

This module intentionally implements the *stages* of the pinned public
repository rather than silently reusing the historical cached-trace path:

``description -> conflict -> global/local imagination -> summary -> retrieval
-> conflict/entity selection -> caption``.

The raw public repository executes eight HTTP requests when all stages are
online (1 description + 1 conflict + 2 imagination + 1 summary + 2 selection
+ 1 caption).  HOMER's paper appendix reports a 2+3+2 budget of seven.  Both
facts are returned in provenance; callers must not collapse them into a false
"exact" claim.  Replacing the official gpt-4o calls with a local frozen
Qwen2.5-VL-7B is the project's adapted pretrained-7B route.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
from typing import Any, Callable, Mapping, Protocol

from .official_prompts import (
    caption_messages,
    conflict_messages,
    description_messages,
    extract_marked,
    global_imagination_messages,
    local_imagination_messages,
    parse_caption_blob,
    select_conflict_messages,
    select_entity_messages,
    summary_messages,
)


class PublicCodeBackend(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> str: ...


class PublicCodeProtocolError(ValueError):
    """A stage output cannot satisfy the public-code contract."""


SummaryRetriever = Callable[[dict[str, Any], str, str], Mapping[str, Any]]


@dataclass(frozen=True)
class PublicCodeRun:
    image: str
    seed: int
    description: str
    conflict_scripts: str
    global_imagination: str
    local_imagination: str
    summary_imagination: str
    retrieved_imagination: dict[str, Any]
    selected_conflict: str
    selected_entities: tuple[str, ...]
    selected_paths: dict[str, tuple[str, ...]]
    caption: str
    explanation: str
    raw_caption: str
    call_log: tuple[dict[str, Any], ...]
    omega: str | None

    @property
    def request_count(self) -> int:
        return len(self.call_log)


def _strip_json_fence(text: str) -> str:
    candidate = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.I | re.S)
    return match.group(1).strip() if match else candidate


def _json_object(text: str, *, stage: str) -> dict[str, Any]:
    try:
        value = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise PublicCodeProtocolError(f"{stage} must be a JSON object") from exc
    if not isinstance(value, dict) or not value:
        raise PublicCodeProtocolError(f"{stage} must be a non-empty JSON object")
    return value


def _entity_names(text: str, *, available: list[str]) -> tuple[str, ...]:
    """Parse the exact list format requested by generator.py."""
    candidate = extract_marked(text, "##Conflict Scripts")
    del candidate  # selection parsing below intentionally uses the entity blob
    matches = re.findall(r"\[([^\]]*)\]", text, flags=re.S)
    if not matches:
        raise PublicCodeProtocolError("entity selection must contain [entity1, entity2]")
    values = tuple(
        item.strip().strip("\\\"'")
        for item in matches[-1].split(",")
        if item.strip().strip("\\\"'")
    )
    if len(values) != 2:
        raise PublicCodeProtocolError(f"entity selection must contain exactly two entities; got {values!r}")
    available_folded = {name.casefold(): name for name in available}
    normalized: list[str] = []
    for value in values:
        if value.casefold() not in available_folded:
            raise PublicCodeProtocolError(
                f"selected entity {value!r} is not present in summary keys {available!r}"
            )
        normalized.append(available_folded[value.casefold()])
    if normalized[0].casefold() == normalized[1].casefold():
        raise PublicCodeProtocolError("selected entities must be distinct")
    return tuple(normalized)


def _paths_for_entity(
    entity: str, value: Any, *, successors: int
) -> tuple[tuple[str, ...], ...]:
    """Convert public summary values to the graph paths used by generator.py."""
    if not isinstance(value, list) or not value:
        raise PublicCodeProtocolError(f"summary entity {entity!r} has no association paths")
    # The public prompt returns [successor1, successor2, successor3].  The
    # released retrieval code may instead return an edge list; preserve both
    # representations without inventing edges.
    if all(isinstance(item, str) for item in value):
        if len(value) != successors:
            raise PublicCodeProtocolError(
                f"summary entity {entity!r} must have exactly {successors} successors; got {len(value)}"
            )
        return ((entity, *tuple(value)),)
    if not all(isinstance(item, list) and len(item) == 2 for item in value):
        raise PublicCodeProtocolError(f"summary entity {entity!r} has invalid edge records")
    graph: dict[str, list[str]] = {}
    for edge in value:
        left, right = (str(edge[0]), str(edge[1]))
        graph.setdefault(left, []).append(right)
    if entity not in graph:
        # Case-preserving lookup is useful for JSON keys returned with a
        # different capitalization, but never creates a new semantic node.
        key = next((item for item in graph if item.casefold() == entity.casefold()), None)
        if key is None:
            raise PublicCodeProtocolError(f"summary has no graph root {entity!r}")
        entity = key
    paths: list[tuple[str, ...]] = []

    def visit(node: str, path: tuple[str, ...]) -> None:
        children = graph.get(node, [])
        if not children:
            paths.append(path)
            return
        for child in children:
            if child in path:
                continue
            visit(child, (*path, child))

    visit(entity, (entity,))
    if not paths:
        raise PublicCodeProtocolError(f"summary entity {entity!r} has no acyclic path")
    return tuple(dict.fromkeys(paths))


class HomerPublicCodePipeline:
    """Run the public HOMER stage order with a deterministic seed ledger."""

    def __init__(
        self,
        backend: PublicCodeBackend,
        *,
        retriever: SummaryRetriever | None = None,
        provider: str = "qwen",
        require_retrieval: bool = True,
        successors: int = 3,
    ) -> None:
        if provider not in {"qwen", "openai"}:
            raise ValueError("provider must be qwen or openai")
        if successors < 1:
            raise ValueError("successors must be positive")
        if require_retrieval and retriever is None:
            raise ValueError("the HOMER public protocol requires a retrieval callback")
        self.backend = backend
        self.retriever = retriever
        self.provider = provider
        self.require_retrieval = require_retrieval
        self.successors = successors

    def _call(
        self,
        stage: str,
        messages: list[dict[str, Any]],
        *,
        seed: int,
        call_log: list[dict[str, Any]],
    ) -> str:
        result = self.backend.generate(
            messages,
            temperature=1.0,
            max_new_tokens=1000,
            seed=seed,
        )
        if not isinstance(result, str) or not result.strip():
            raise PublicCodeProtocolError(f"{stage} returned an empty response")
        call_log.append({
            "stage": stage,
            "seed": seed,
            "temperature": 1.0,
            "max_new_tokens": 1000,
        })
        return result.strip()

    def run(
        self,
        *,
        image: str,
        seed: int,
        standard_description: str | None = None,
        omega: str | None = None,
    ) -> PublicCodeRun:
        call_log: list[dict[str, Any]] = []
        # Use independent deterministic sub-seeds while retaining the Python
        # random-selection semantics of the public generator implementation.
        stage_seed = seed * 1000
        if standard_description is None:
            description_blob = self._call(
                "extractor.description", description_messages(image, provider=self.provider),
                seed=stage_seed + 1, call_log=call_log,
            )
            description = extract_marked(description_blob, "##Vivid Description")
        else:
            description = standard_description.strip()
            if not description:
                raise PublicCodeProtocolError("standard description must not be empty")

        conflict_blob = self._call(
            "extractor.conflict", conflict_messages(description),
            seed=stage_seed + 2, call_log=call_log,
        )
        global_blob = self._call(
            "imaginator.global",
            global_imagination_messages(image, conflict_blob, provider=self.provider),
            seed=stage_seed + 3, call_log=call_log,
        )
        local_blob = self._call(
            "imaginator.local",
            local_imagination_messages(description, conflict_blob),
            seed=stage_seed + 4, call_log=call_log,
        )
        summary_blob = self._call(
            "imaginator.summary",
            summary_messages(description, conflict_blob, global_blob, local_blob),
            seed=stage_seed + 5, call_log=call_log,
        )
        summary = _json_object(summary_blob, stage="summary imagination")
        if self.retriever is None:
            if self.require_retrieval:
                raise PublicCodeProtocolError("retrieval callback is required")
            retrieved = summary
        else:
            retrieved = dict(self.retriever(summary, description, conflict_blob))
            if not retrieved:
                raise PublicCodeProtocolError("retrieval returned an empty imagination graph")

        conflict_selection_blob = self._call(
            "generator.select_conflict",
            select_conflict_messages(description, conflict_blob),
            seed=stage_seed + 6, call_log=call_log,
        )
        conflict_selection = extract_marked(conflict_selection_blob, "##Conflict Scripts")
        if not conflict_selection:
            raise PublicCodeProtocolError("selected conflict script is empty")
        entity_selection_blob = self._call(
            "generator.select_entities",
            select_entity_messages(conflict_selection, list(retrieved)),
            seed=stage_seed + 7, call_log=call_log,
        )
        selected_entities = _entity_names(entity_selection_blob, available=list(retrieved))
        rng = random.Random(seed)
        selected_paths: dict[str, tuple[str, ...]] = {}
        free_assoc_lines: list[str] = []
        for entity in selected_entities:
            paths = _paths_for_entity(entity, retrieved[entity], successors=self.successors)
            path = rng.choice(paths)
            selected_paths[entity] = path
            free_assoc_lines.append(
                f"The free-association chain of {entity} is {' -> '.join(path)}"
            )
        caption_blob = self._call(
            "generator.caption",
            caption_messages(description, conflict_selection, "\n".join(free_assoc_lines), omega=omega),
            seed=stage_seed + 8, call_log=call_log,
        )
        caption, explanation = parse_caption_blob(caption_blob)
        if not caption:
            raise PublicCodeProtocolError("caption section is empty")
        return PublicCodeRun(
            image=image,
            seed=seed,
            description=description,
            conflict_scripts=conflict_blob,
            global_imagination=global_blob,
            local_imagination=local_blob,
            summary_imagination=summary_blob,
            retrieved_imagination=retrieved,
            selected_conflict=conflict_selection,
            selected_entities=selected_entities,
            selected_paths=selected_paths,
            caption=caption,
            explanation=explanation,
            raw_caption=caption_blob,
            call_log=tuple(call_log),
            omega=omega,
        )
