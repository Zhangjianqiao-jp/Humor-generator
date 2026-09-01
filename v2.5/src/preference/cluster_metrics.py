"""Image-clustered summaries for correlated caption preference pairs."""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_image_clusters(
    rows: Iterable[dict[str, Any]],
    metric_names: Iterable[str],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260828,
) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("Image-cluster summary requires at least one pair.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["image_id"])].append(row)
    image_ids = sorted(grouped)
    rng = random.Random(seed)
    metrics: dict[str, Any] = {}
    for metric in metric_names:
        pair_values = [float(row[metric]) for row in records]
        image_values = [mean(float(row[metric]) for row in grouped[image_id]) for image_id in image_ids]
        resamples = []
        if bootstrap_samples > 0:
            for _ in range(bootstrap_samples):
                resamples.append(mean(rng.choice(image_values) for _ in image_values))
        ci = (
            [percentile(resamples, 0.025), percentile(resamples, 0.975)]
            if resamples
            else [None, None]
        )
        metrics[metric] = {
            "pair_mean": mean(pair_values),
            "image_mean": mean(image_values),
            "image_median": median(image_values),
            "image_cluster_bootstrap_95ci": ci,
        }
    return {
        "pair_count": len(records),
        "image_count": len(image_ids),
        "bootstrap_unit": "image",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "metrics": metrics,
    }
