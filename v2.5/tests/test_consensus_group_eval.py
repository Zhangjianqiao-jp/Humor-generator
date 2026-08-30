from scripts.build_group_winrate_eval import build_packet
from scripts.report_consensus_group_eval import summarize_consensus


def test_consensus_summary_maps_blind_sides_and_preserves_unresolved() -> None:
    systems = {
        name: {
            f"i{i}": {
                "image": f"i{i}.jpg",
                "image_id": f"i{i}",
                "candidates": [f"{name}-{i}-{j}" for j in range(3)],
            }
            for i in range(4)
        }
        for name in ("new", "old")
    }
    _, key_doc, _ = build_packet(systems, [("new", "old")], seed=4)
    consensus = {}
    expected = ("new", "old", "Tie", "unresolved")
    for key, wanted in zip(key_doc["key"], expected, strict=True):
        if wanted in {"Tie", "unresolved"}:
            label = wanted
        else:
            label = "A" if key["group_A_system"] == wanted else "B"
        consensus[key["pair_id"]] = {
            "overall": label,
            "best_pick": label,
            "absolute_A": "weak",
            "absolute_B": "weak",
        }
    report = summarize_consensus(
        key_doc, {"raters": ["r1", "r2", "r3"], "trials": 4, "consensus": consensus}
    )
    overall = report["comparisons"]["new_vs_old"]["overall"]
    assert (overall["wins"], overall["losses"], overall["ties"], overall["unresolved"]) == (
        1,
        1,
        1,
        1,
    )
    assert overall["neutral_imputed_score"] == 0.5
    assert overall["resolved_only_score"] == 0.5
    assert overall["decisive_win_rate"] == 0.5
