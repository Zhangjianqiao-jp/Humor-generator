"""Evaluation definitions disclosed by HOMER."""
from __future__ import annotations

from math import comb
from typing import Iterable


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if n < 1 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("require n>=1, 0<=c<=n, and 1<=k<=n")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def mean_pass_at_k(items: Iterable[tuple[int, int]], k: int) -> float:
    values = [unbiased_pass_at_k(n, c, k) for n, c in items]
    if not values:
        raise ValueError("at least one image is required")
    return sum(values) / len(values)
