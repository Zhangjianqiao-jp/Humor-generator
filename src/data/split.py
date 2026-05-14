from __future__ import annotations

import random
from collections.abc import Iterable


def split_image_ids(image_ids: Iterable[str], train_ratio: float = 0.9, val_ratio: float = 0.05, test_ratio: float = 0.05, seed: int = 42) -> dict[str, set[str]]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")
    unique_ids = sorted(set(str(x) for x in image_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = set(unique_ids[:n_train])
    val = set(unique_ids[n_train:n_train + n_val])
    test = set(unique_ids[n_train + n_val:])
    return {"train": train, "val": val, "test": test}
