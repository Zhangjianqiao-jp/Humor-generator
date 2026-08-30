"""Strict internal contracts for the public HOMER protocol.

The paper's conflict prompt returns numbered prose while its imaginator prompt
requests JSON.  We preserve those external formats and validate both into a
single typed representation instead of silently accepting malformed output.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence


class SchemaError(ValueError):
    """A model output violates a preregistered HOMER contract."""


@dataclass(frozen=True)
class ValidationLimits:
    min_description_chars: int = 20
    max_description_chars: int = 2400
    min_conflict_pairs: int = 2
    max_conflict_pairs: int = 12
    max_conflict_side_chars: int = 180
    chain_length: int = 3
    max_entities: int = 32
    max_entity_chars: int = 120


@dataclass(frozen=True)
class ConflictPair:
    left: str
    right: str

    def render(self) -> str:
        return f"{self.left} vs. {self.right}"


@dataclass(frozen=True)
class AssociationChain:
    root: str
    steps: tuple[str, str, str]
    view: str

    @property
    def path(self) -> tuple[str, ...]:
        return (self.root, *self.steps)


@dataclass(frozen=True)
class HumorLeaf:
    entity: str
    total: float
    relevance: float
    frequency: float
    diversity: float


@dataclass(frozen=True)
class NodeExpansion:
    node_index: int
    node: str
    retrieved_jokes: tuple[str, ...]
    leaves: tuple[HumorLeaf, ...]


@dataclass(frozen=True)
class ImaginationTree:
    backbone: AssociationChain
    expansions: tuple[NodeExpansion, ...]

    def paths(self) -> tuple[tuple[str, ...], ...]:
        """Enumerate ancestor-to-leaf paths exactly as HOMER's DFS stage."""
        backbone = self.backbone.path
        result: list[tuple[str, ...]] = []
        expansion_by_index = {item.node_index: item for item in self.expansions}
        for index, _node in enumerate(backbone):
            expansion = expansion_by_index.get(index)
            if expansion is not None:
                result.extend(backbone[: index + 1] + (leaf.entity,) for leaf in expansion.leaves)
        last = expansion_by_index.get(len(backbone) - 1)
        if last is None or not last.leaves:
            result.append(backbone)
        # Retrieval tokens can repeat backbone nodes; keep paths stable and unique.
        return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class HomerPlan:
    description: str
    conflicts: tuple[ConflictPair, ...]
    local_chains: tuple[AssociationChain, ...]
    global_chains: tuple[AssociationChain, ...]
    imagination_trees: tuple[ImaginationTree, ...] = ()

    def conflicts_text(self) -> str:
        return " ".join(f"{index}. {pair.render()}" for index, pair in enumerate(self.conflicts, 1))


_NUMBERED = re.compile(r"(?:^|\s)(?:\d+\s*[.)]|[-*])\s*(.*?)(?=(?:\s+\d+\s*[.)]|\s+[-*])\s|$)", re.S)
_OPPOSITION = re.compile(r"\s+(?:vs\.?|versus|/|↔|<->)\s+", re.I)


def _clean(value: Any, *, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{label} must be a string")
    result = " ".join(value.split()).strip(" \t\r\n,;[]{}\"")
    if not result:
        raise SchemaError(f"{label} must not be empty")
    if len(result) > maximum:
        raise SchemaError(f"{label} exceeds {maximum} characters")
    return result


def validate_description(text: str, limits: ValidationLimits = ValidationLimits()) -> str:
    result = _clean(text, label="description", maximum=limits.max_description_chars)
    if len(result) < limits.min_description_chars:
        raise SchemaError(f"description is shorter than {limits.min_description_chars} characters")
    if re.search(r"\bcaption\s*:", result, flags=re.I):
        raise SchemaError("description must not contain a generated caption")
    return result


def _pair_from_item(item: Any, limits: ValidationLimits) -> ConflictPair:
    if isinstance(item, Mapping):
        left, right = item.get("left"), item.get("right")
    elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
        left, right = item
    elif isinstance(item, str):
        parts = _OPPOSITION.split(item.strip(), maxsplit=1)
        if len(parts) != 2:
            raise SchemaError(f"conflict item has no explicit opposition marker: {item!r}")
        left, right = parts
    else:
        raise SchemaError(f"unsupported conflict item: {item!r}")
    left = _clean(left, label="conflict.left", maximum=limits.max_conflict_side_chars)
    right = _clean(right, label="conflict.right", maximum=limits.max_conflict_side_chars)
    if left.casefold() == right.casefold():
        raise SchemaError("two sides of a conflict must differ")
    return ConflictPair(left, right)


def parse_conflicts(text: str, limits: ValidationLimits = ValidationLimits()) -> tuple[ConflictPair, ...]:
    """Parse JSON or the numbered prose requested by HOMER Appendix prompt 1."""
    raw: Any
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.I | re.S)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError:
        items = [match.group(1).strip(" .") for match in _NUMBERED.finditer(text.strip())]
        raw = items or [text]
    if isinstance(raw, Mapping):
        raw = raw.get("conflicts", raw.get("script_oppositions"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SchemaError("conflicts must be a list or numbered list")
    pairs = tuple(_pair_from_item(item, limits) for item in raw)
    if not limits.min_conflict_pairs <= len(pairs) <= limits.max_conflict_pairs:
        raise SchemaError(
            f"expected {limits.min_conflict_pairs}..{limits.max_conflict_pairs} conflict pairs; got {len(pairs)}"
        )
    if len({(pair.left.casefold(), pair.right.casefold()) for pair in pairs}) != len(pairs):
        raise SchemaError("duplicate conflict pairs are not allowed")
    return pairs


def parse_associations(
    text: str,
    *,
    view: str,
    limits: ValidationLimits = ValidationLimits(),
) -> tuple[AssociationChain, ...]:
    if view not in {"local", "global"}:
        raise SchemaError("association view must be local or global")
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.I | re.S)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SchemaError("imaginator output must be valid JSON") from exc
    normalized: list[tuple[Any, Any]] = []
    if isinstance(raw, Mapping) and raw:
        normalized = list(raw.items())
    elif isinstance(raw, list) and raw:
        # Two lossless wrappers are accepted around the paper's entity->chain
        # mapping: a list of one-key mappings, or strictly typed records. No
        # field aliases or arbitrary prose structures are admitted.
        if all(isinstance(item, Mapping) and len(item) == 1 for item in raw):
            for item in raw:
                normalized.extend(item.items())
        else:
            for item in raw:
                if not isinstance(item, Mapping) or set(item) != {"entity", "associations"}:
                    raise SchemaError(
                        "record-style imagination requires exactly entity and associations"
                    )
                associations = item["associations"]
                if (
                    isinstance(associations, list)
                    and len(associations) == limits.chain_length
                    and all(
                        isinstance(edge, list) and len(edge) == 2
                        and all(isinstance(value, str) for value in edge)
                        for edge in associations
                    )
                ):
                    # Qwen may spell one three-step chain as three contiguous
                    # edges: [root,a], [a,b], [b,c]. This is mathematically the
                    # same chain, not a relaxed or inferred label.
                    root = _clean(item["entity"], label="association.root", maximum=limits.max_entity_chars)
                    edges = associations
                    if (
                        _clean(edges[0][0], label="association.edge", maximum=limits.max_entity_chars).casefold()
                        != root.casefold()
                        or any(
                            _clean(edges[index][1], label="association.edge", maximum=limits.max_entity_chars).casefold()
                            != _clean(edges[index + 1][0], label="association.edge", maximum=limits.max_entity_chars).casefold()
                            for index in range(len(edges) - 1)
                        )
                    ):
                        raise SchemaError(f"association {root!r} edge chain is not contiguous")
                    normalized.append((root, [edge[1] for edge in edges]))
                elif (
                    isinstance(associations, list)
                    and associations
                    and all(isinstance(value, list) for value in associations)
                ):
                    normalized.extend((item["entity"], value) for value in associations)
                else:
                    normalized.append((item["entity"], associations))
    else:
        raise SchemaError("imaginator JSON must be a non-empty object or typed record list")
    if len(normalized) > limits.max_entities:
        raise SchemaError(f"imaginator returned more than {limits.max_entities} chains")
    chains: list[AssociationChain] = []
    for root, values in normalized:
        root_clean = _clean(root, label="association.root", maximum=limits.max_entity_chars)
        if not isinstance(values, list) or len(values) != limits.chain_length:
            raise SchemaError(
                f"association {root_clean!r} must contain exactly {limits.chain_length} chained entities"
            )
        steps = tuple(_clean(item, label="association.step", maximum=limits.max_entity_chars) for item in values)
        if len({value.casefold() for value in (root_clean, *steps)}) != 4:
            raise SchemaError(f"association {root_clean!r} repeats an entity")
        chains.append(AssociationChain(root_clean, steps, view))
    return tuple(chains)


def validate_plan(
    description: str,
    conflicts_text: str,
    local_text: str,
    global_text: str,
    limits: ValidationLimits = ValidationLimits(),
) -> HomerPlan:
    return HomerPlan(
        description=validate_description(description, limits),
        conflicts=parse_conflicts(conflicts_text, limits),
        local_chains=parse_associations(local_text, view="local", limits=limits),
        global_chains=parse_associations(global_text, view="global", limits=limits),
    )
