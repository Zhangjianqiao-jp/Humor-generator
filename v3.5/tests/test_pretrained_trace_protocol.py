from __future__ import annotations

from pathlib import Path

from scripts.cache_pretrained_homer_traces import channel_prompts, load_inputs


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data/processed/homer_pretrained_7b_public_release_362"


def test_public_release_trace_inputs_are_complete_and_source_aware() -> None:
    rows = load_inputs(DATASET)
    assert len(rows) == 362
    assert [int(row["contest_number"]) for row in rows] == sorted(
        int(row["contest_number"]) for row in rows
    )
    assert all(row["population_version"] == "homer_pretrained_7b_public_release_362" for row in rows)


def test_pretrained_trace_prompts_use_the_three_official_channel_builders() -> None:
    row = load_inputs(DATASET)[0]
    prompts = channel_prompts(row, "1. ordinary work vs. absurd bureaucracy")
    assert set(prompts) == {"conflict", "global", "local"}
    assert any(
        block.get("type") == "image"
        for block in prompts["global"][1]["content"]
        if isinstance(block, dict)
    )
    assert all(message["role"] in {"system", "user"} for values in prompts.values() for message in values)


def test_route_config_points_to_adapter_free_trace_index() -> None:
    config_text = (ROOT / "configs/homer_public_code_pretrained_7b.yaml").read_text()
    assert "trace_index: data/cache/homer_pretrained_7b_planner_traces/index.jsonl" in config_text
