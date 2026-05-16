from src.data.split import split_image_ids


def test_no_image_leakage() -> None:
    ids = [f"img_{i}" for i in range(100)]
    splits = split_image_ids(ids, seed=123)
    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["val"].isdisjoint(splits["test"])
