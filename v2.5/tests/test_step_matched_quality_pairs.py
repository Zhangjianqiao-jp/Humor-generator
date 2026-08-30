from scripts.preference_diagnostics.select_step_matched_quality_pairs import (
    proportional_quotas,
    select_rows,
)


def _row(image: str, tier: str, index: int) -> dict:
    return {"image_id": image, "quality_tier": tier, "pair_id": f"{image}-{tier}-{index}"}


def test_proportional_quotas_sum_exactly() -> None:
    quotas = proportional_quotas({"clear": 6, "medium": 3, "hard": 1}, 7)
    assert quotas == {"clear": 4, "medium": 2, "hard": 1}
    assert sum(quotas.values()) == 7


def test_selection_is_tier_stratified_and_image_diverse() -> None:
    rows = []
    for tier in ("clear", "medium", "hard"):
        for image in ("a", "b", "c"):
            rows.extend(_row(image, tier, index) for index in range(3))
    selected, quotas = select_rows(rows, 9)
    assert quotas == {"clear": 3, "hard": 3, "medium": 3}
    assert len(selected) == 9
    assert {row["image_id"] for row in selected} == {"a", "b", "c"}
    assert len({row["pair_id"] for row in selected}) == 9
