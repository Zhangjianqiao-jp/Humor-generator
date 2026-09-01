from scripts.preference_diagnostics.expand_published_newyorker_dpo_pairs import expand_train


def row(contest: int, chosen: str, rejected: str, rank: int, z: float = 3.1) -> dict:
    return {
        "contest_number": contest,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_rank": rank,
        "z_margin": z,
    }


def test_expansion_is_nested_and_prefers_new_captions() -> None:
    base = [row(1, "base chosen", "base rejected", 10)]
    candidates = base + [
        row(1, "new a", "new b", 20),
        row(1, "base chosen", "new c", 1),
        row(1, "new d", "new e", 30),
    ]
    expanded = expand_train(candidates, base, 3, 0.35, 7)
    assert expanded[0] == base[0]
    assert {(item["chosen"], item["rejected"]) for item in expanded[1:]} == {
        ("new a", "new b"),
        ("new d", "new e"),
    }
    assert all(item["selection"]["labels_unchanged"] for item in expanded[1:])


def test_expansion_filters_low_confidence_and_length_mismatch() -> None:
    base = [row(1, "base", "pair", 1)]
    candidates = base + [
        row(1, "low confidence", "other caption", 2, z=2.9),
        row(1, "x", "a caption that is much too long", 3),
        row(1, "valid one", "valid two", 4),
    ]
    expanded = expand_train(candidates, base, 2, 0.35, 7)
    assert [(item["chosen"], item["rejected"]) for item in expanded] == [
        ("base", "pair"),
        ("valid one", "valid two"),
    ]
