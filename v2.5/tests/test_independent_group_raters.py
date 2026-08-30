from scripts.build_independent_group_rater_packets import build_rater
from scripts.report_independent_group_raters import (
    canonicalize,
    cohen_kappa,
    exact_two_sided_binomial,
    fleiss_kappa,
)


def test_reblinding_is_complete_balanced_and_reversible(tmp_path) -> None:
    image = tmp_path / "x.jpg"
    image.write_bytes(b"not-decoded-by-this-unit-test")
    rows = [
        {"pair_id": f"p{i}", "image_id": f"im{i}", "image": str(image),
         "group_A": ["a1", "a2", "a3"], "group_B": ["b1", "b2", "b3"]}
        for i in range(5)
    ]
    public, key = build_rater(rows, "r1", 7)
    assert len({row["blind_id"] for row in public}) == 5
    assert sum(row["swapped"] for row in key) == 2
    decisions = {}
    for row in key:
        decisions[row["blind_id"]] = {
            "overall": "B" if row["swapped"] else "A",
            "best_pick": "B" if row["swapped"] else "A",
            "best_A_index": 1, "best_B_index": 2,
            "absolute_A": "bad" if row["swapped"] else "good",
            "absolute_B": "good" if row["swapped"] else "bad",
        }
    restored = canonicalize({"rater_id": "r1", "decisions": decisions}, key)
    assert all(row == {"overall": "A", "best_pick": "A", "absolute_A": "good", "absolute_B": "bad"} for row in restored.values())


def test_cohen_kappa_endpoints() -> None:
    assert cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == 1.0
    assert cohen_kappa(["A", "A", "B", "B"], ["B", "B", "A", "A"]) == -1.0
    assert fleiss_kappa([["A", "A", "A"], ["B", "B", "B"]]) == 1.0
    assert exact_two_sided_binomial(5, 5) == 1.0
    assert exact_two_sided_binomial(10, 0) < 0.01
