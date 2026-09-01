"""Immutable Planner trace records for formal bridge experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..homer.contracts import AssociationChain, ConflictPair, HomerPlan
from ..latent.state_capture import AlignedMessageStates


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_to_record(plan: HomerPlan) -> dict[str, Any]:
    return {
        "description": plan.description,
        "conflicts": [asdict(value) for value in plan.conflicts],
        "local_chains": [asdict(value) for value in plan.local_chains],
        "global_chains": [asdict(value) for value in plan.global_chains],
    }


def plan_from_record(value: dict[str, Any]) -> HomerPlan:
    return HomerPlan(
        description=value["description"],
        conflicts=tuple(ConflictPair(**item) for item in value["conflicts"]),
        local_chains=tuple(
            AssociationChain(item["root"], tuple(item["steps"]), item["view"])
            for item in value["local_chains"]
        ),
        global_chains=tuple(
            AssociationChain(item["root"], tuple(item["steps"]), item["view"])
            for item in value["global_chains"]
        ),
    )


def save_trace(path: Path, states: dict[str, AlignedMessageStates]) -> str:
    if set(states) != {"conflict", "local", "global"}:
        raise ValueError("trace requires conflict/local/global states")
    payload: dict[str, dict[str, Any]] = {}
    for name, aligned in states.items():
        if aligned.states.shape[:2] != aligned.token_ids.shape:
            raise ValueError(f"unaligned trace channel: {name}")
        if (
            not aligned.semantics.strip()
            or aligned.semantics in {"state_used_to_predict_corresponding_token", "unspecified"}
        ):
            raise ValueError(f"trace channel has no actual generated semantics: {name}")
        payload[name] = {
            "states": aligned.states.to(dtype=torch.float16, device="cpu").contiguous(),
            "token_ids": aligned.token_ids.to(dtype=torch.long, device="cpu").contiguous(),
            "semantics": aligned.semantics,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return file_sha256(path)


def load_trace(path: Path, *, expected_sha256: str | None = None) -> dict[str, AlignedMessageStates]:
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise RuntimeError(f"trace hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    result = {
        name: AlignedMessageStates(item["token_ids"], item["states"], item["semantics"])
        for name, item in payload.items()
    }
    if set(result) != {"conflict", "local", "global"}:
        raise RuntimeError(f"trace has invalid channel set: {path}")
    if any(
        not item.semantics.strip()
        or item.semantics in {"state_used_to_predict_corresponding_token", "unspecified"}
        for item in result.values()
    ):
        raise RuntimeError(f"trace contains placeholder rather than generated semantics: {path}")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
