#!/usr/bin/env python3
"""Validate the compact v3.5 project-memory handoff."""
from __future__ import annotations

import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "memory"


def load_yaml(name: str) -> dict:
    value = yaml.safe_load((MEMORY / name).read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"{name} must be a schema-version-1 mapping")
    return value


def require_paths(paths: list[str], *, label: str) -> None:
    missing = [value for value in paths if not (ROOT / value).exists()]
    if missing:
        raise ValueError(f"{label} references missing paths: {missing}")


def main() -> None:
    project = load_yaml("project_memory.yaml")
    working = load_yaml("working_state.yaml")
    retrieval = load_yaml("retrieval_index.yaml")
    if project.get("scope") != "v3.5_only":
        raise ValueError("project memory must retain the v3.5 isolation boundary")
    if working.get("volatile") is not True or working.get("must_revalidate") is not True:
        raise ValueError("working state must be explicitly volatile and revalidated")
    if not re.fullmatch(r"[0-9a-f]{40}", working.get("source_commit", "")):
        raise ValueError("working state requires a full source commit")
    phases = project["experiment_plan"]["phases"]
    phase_ids = [item["id"] for item in phases]
    if len(phase_ids) != len(set(phase_ids)):
        raise ValueError("experiment phase IDs must be unique")
    if project["trace_contract"]["required_clusters"] != 666:
        raise ValueError("formal trace gate must remain 666 clusters")
    if project["evaluation"]["pilot"]["role"] != "screening_only":
        raise ValueError("Group-of-3 must not become a confirmatory endpoint")
    if project["evaluation"]["confirmatory"]["protocol"] != "mirrored_blind_group_of_10":
        raise ValueError("confirmatory evaluation must remain mirrored Group-of-10")
    require_paths(project["canonical_docs"], label="canonical_docs")
    for topic, record in retrieval["topics"].items():
        require_paths(record["files"], label=f"retrieval topic {topic}")

    episodes = [
        json.loads(line)
        for line in (MEMORY / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ids = [item["id"] for item in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("episode IDs must be unique")
    for episode in episodes:
        require_paths(episode["evidence"], label=f"episode {episode['id']}")
        if not episode.get("lesson"):
            raise ValueError(f"episode {episode['id']} has no reusable lesson")
    print(json.dumps({
        "status": "pass",
        "phases": len(phases),
        "retrieval_topics": len(retrieval["topics"]),
        "episodes": len(episodes),
        "working_state_volatile": True,
    }, indent=2))


if __name__ == "__main__":
    main()
