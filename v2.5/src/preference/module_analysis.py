from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


TRANSFORMER_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def lora_parameter_count(in_features: int, out_features: int, rank: int) -> int:
    if min(in_features, out_features, rank) <= 0:
        raise ValueError("LoRA dimensions and rank must be positive")
    return rank * (in_features + out_features)


def rank_statistics(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row["adaptation_utility"]), str(row["module_path"])))
    total = sum(max(0.0, float(row["adaptation_utility"])) for row in ranked)
    cumulative = 0.0
    output = []
    for rank, source in enumerate(ranked, start=1):
        row = dict(source)
        cumulative += max(0.0, float(row["adaptation_utility"]))
        row["rank"] = rank
        row["cumulative_utility_fraction"] = cumulative / total if total else 0.0
        output.append(row)
    return output


def cumulative_selection(rows: Iterable[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    ranked = rank_statistics(rows)
    selected = []
    for row in ranked:
        selected.append(row)
        if float(row["cumulative_utility_fraction"]) >= threshold:
            break
    return selected


def aggregate_module_groups(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {"pref_grad_sq": 0.0, "fisher_sum": 0.0, "utility": 0.0, "params": 0.0, "count": 0.0}
    )
    for row in rows:
        value = grouped[str(row["module"])]
        value["pref_grad_sq"] += float(row["raw_grad_norm"]) ** 2
        value["fisher_sum"] += float(row["mean_fisher"])
        value["utility"] += float(row["adaptation_utility"])
        value["params"] += int(row["parameter_count"])
        value["count"] += 1
    output = []
    for module, value in sorted(grouped.items()):
        output.append(
            {
                "module": module,
                "raw_grad_norm": math.sqrt(value["pref_grad_sq"]),
                "mean_fisher": value["fisher_sum"] / max(value["count"], 1),
                "adaptation_utility": value["utility"],
                "utility_per_parameter": value["utility"] / max(value["params"], 1),
                "parameter_count": int(value["params"]),
            }
        )
    return output


def matched_uniform_rank(
    module_shapes: Iterable[tuple[int, int]], target_budget: int, *, minimum_rank: int = 1
) -> tuple[int, int]:
    denominator = sum(int(in_features) + int(out_features) for in_features, out_features in module_shapes)
    if denominator <= 0 or target_budget <= 0:
        raise ValueError("module shapes and target budget must be positive")
    ideal = target_budget / denominator
    candidates = {max(minimum_rank, int(math.floor(ideal))), max(minimum_rank, int(math.ceil(ideal)))}
    rank = min(candidates, key=lambda value: (abs(value * denominator - target_budget), value))
    return rank, rank * denominator


def random_budget_selection(
    rows: list[dict[str, Any]], target_parameter_count: int, seed: int
) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    count = 0
    for row in shuffled:
        candidate = count + int(row["parameter_count"])
        if selected and abs(count - target_parameter_count) <= abs(candidate - target_parameter_count):
            break
        selected.append(row)
        count = candidate
    return selected
