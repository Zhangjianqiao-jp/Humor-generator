from scripts.preference_diagnostics.build_quality_stratified_newyorker_pairs import tier


def row(chosen_rank: int, rejected_rank: int, z: float) -> dict:
    return {"chosen_rank": chosen_rank, "rejected_rank": rejected_rank, "z_margin": z}


def test_quality_tiers_are_exclusive_and_require_high_quality_chosen() -> None:
    assert tier(row(9, 60, 5.0), 100) == "clear"
    assert tier(row(15, 45, 4.0), 100) == "medium"
    assert tier(row(20, 41, 3.1), 100) == "hard"
    assert tier(row(30, 80, 8.0), 100) is None
    assert tier(row(5, 90, 2.9), 100) is None
