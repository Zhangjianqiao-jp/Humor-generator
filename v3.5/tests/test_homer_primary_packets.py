from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_homer_primary_judgments import aggregate
from scripts.build_homer_primary_packets import extract_official_system_prompt


def test_official_homer_prompt_is_extracted_from_pinned_source() -> None:
    prompt = extract_official_system_prompt()
    assert "You are an impartial humor judge" in prompt
    assert "Output format:" in prompt
    assert "A or B" in prompt


def test_primary_aggregate_rejects_partial_reference_group(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.jsonl"
    judgments = tmp_path / "judgments.jsonl"
    rows = []
    decisions = []
    for candidate_index in range(5):
        packet_id = f"p{candidate_index}"
        rows.append({
            "packet_id": packet_id,
            "condition": "latent_bridge",
            "image_id": "im1",
            "contest_number": 1,
            "image_path": "x.jpg",
            "image_sha256": "a" * 64,
            "split": "test",
            "trial": 0,
            "candidate_index": candidate_index,
            "generation_seed": 100 + candidate_index,
            "reference_group": "1-9",
            "reference_index": 0,
        })
        decisions.append({"packet_id": packet_id, "decision": "A"})
    mapping.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    judgments.write_text("".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8")
    with pytest.raises(ValueError, match="reference indices"):
        aggregate(mapping, judgments, tmp_path / "records.jsonl")
