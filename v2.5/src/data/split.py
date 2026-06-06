from __future__ import annotations

import random
from collections.abc import Iterable


def split_image_ids(
    image_ids: Iterable[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, set[str]]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    unique_ids = sorted({str(image_id) for image_id in image_ids})
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    return {
        "train": set(unique_ids[:n_train]),
        "val": set(unique_ids[n_train : n_train + n_val]),
        "test": set(unique_ids[n_train + n_val :]),
    }
