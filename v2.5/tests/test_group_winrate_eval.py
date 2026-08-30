import json
import sys

from scripts.build_group_winrate_eval import build_packet
from scripts.report_group_winrate_eval import main as report_main
from scripts.report_group_winrate_eval import wilson
from scripts.report_multiseed_group_eval import (
    clustered_bootstrap_ci,
    parse_policy_and_seed,
)


def _systems() -> dict:
    return {
        name: {
            f"i{i}": {
                "image": f"i{i}.jpg",
                "image_id": f"i{i}",
                "candidates": [f"{name}-{i}-{j}" for j in range(3)],
            }
            for i in range(4)
        }
        for name in ("correct", "direct", "swapped")
    }


def test_packet_is_blind_unique_and_position_balanced() -> None:
    public, key_doc, template = build_packet(
        _systems(), [("correct", "direct"), ("correct", "swapped")], seed=9
    )
    assert len(public) == 8
    assert len({row["pair_id"] for row in public}) == 8
    assert set(template["decisions"]) == {row["pair_id"] for row in public}
    for comparison in key_doc["comparisons"]:
        rows = [row for row in key_doc["key"] if row["comparison"] == comparison]
        assert sum(row["group_A_system"] == row["system_a"] for row in rows) == 2
        assert all("system" not in row for row in public)
    assert all(
        {"absolute_A", "absolute_B"} <= set(decision)
        for decision in template["decisions"].values()
    )


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson(18, 24)
    assert low < 0.75 < high


def test_report_accepts_protocol_ties(tmp_path, monkeypatch) -> None:
    public, key_doc, template = build_packet(
        _systems(), [("correct", "direct")], seed=11
    )
    for decision in template["decisions"].values():
        decision.update(
            {
                "overall": "Tie",
                "best_pick": "Tie",
                "best_A_index": 1,
                "best_B_index": 1,
                "absolute_A": "weak",
                "absolute_B": "weak",
            }
        )
    public_path = tmp_path / "public.jsonl"
    key_path = tmp_path / "key.json"
    decisions_path = tmp_path / "decisions.json"
    output_path = tmp_path / "report.json"
    public_path.write_text("".join(json.dumps(row) + "\n" for row in public))
    key_path.write_text(json.dumps(key_doc))
    decisions_path.write_text(json.dumps(template))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_group_winrate_eval.py",
            "--public",
            str(public_path),
            "--key",
            str(key_path),
            "--decisions",
            str(decisions_path),
            "--output",
            str(output_path),
        ],
    )
    report_main()
    comparison = json.loads(output_path.read_text())["comparisons"]["correct_vs_direct"]
    assert comparison["overall_ties"] == 4
    assert comparison["overall_tie_adjusted_rate"] == 0.5
    assert comparison["overall_decisive_win_rate"] is None


def test_multiseed_helpers_parse_names_and_cluster_by_image() -> None:
    assert parse_policy_and_seed("dpo_s20260827") == ("dpo", 20260827)
    low, high = clustered_bootstrap_ci([0.0, 1.0, 1.0], samples=1000, seed=7)
    assert low <= 2 / 3 <= high
