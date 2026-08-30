from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


DEFAULT_DIMENSIONS = ("humor", "grounding", "originality", "specificity")


def score_caption(scores: dict[str, Any], weights: dict[str, float]) -> float:
    missing = [name for name, weight in weights.items() if weight and name not in scores]
    if missing:
        raise ValueError(f"caption score is missing weighted dimensions: {missing}")
    return sum(float(scores[name]) * float(weight) for name, weight in weights.items())


def summarize_hint_rows(rows: Iterable[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        captions = row.get("judged_candidates") or row.get("captions") or []
        if len(captions) < 2:
            raise ValueError("Hint utility requires at least two independently sampled captions per hint")
        dimension_values: dict[str, list[float]] = defaultdict(list)
        rewards = []
        caption_records = []
        for item in captions:
            scores = dict(item.get("scores") or item)
            reward = score_caption(scores, weights)
            rewards.append(reward)
            for dimension in DEFAULT_DIMENSIONS:
                if dimension in scores:
                    dimension_values[dimension].append(float(scores[dimension]))
            caption_records.append(
                {
                    "caption": str(item.get("candidate") or item.get("caption") or ""),
                    "scores": {name: float(scores[name]) for name in DEFAULT_DIMENSIONS if name in scores},
                    "reward": reward,
                }
            )
        output.append(
            {
                "image_id": str(row["source_image_id"] if row.get("source_image_id") else row["image_id"]),
                "image": row["image"],
                "hint_id": str(row["hint_id"]),
                "hint": str(row["hint"]),
                "hint_utility": sum(rewards) / len(rewards),
                "utility_dimensions": {
                    name: sum(values) / len(values) for name, values in dimension_values.items() if values
                },
                "generated_captions": caption_records,
                "caption_samples": len(caption_records),
            }
        )
    return output


def dominates(chosen: dict[str, Any], rejected: dict[str, Any], tolerance: float = 0.0) -> bool:
    chosen_dims = chosen.get("utility_dimensions") or {}
    rejected_dims = rejected.get("utility_dimensions") or {}
    shared = set(chosen_dims) & set(rejected_dims)
    return bool(shared) and all(float(chosen_dims[key]) + tolerance >= float(rejected_dims[key]) for key in shared)


def build_hint_pairs(
    summaries: Iterable[dict[str, Any]], *, min_margin: float, require_dominance: bool
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[str(row["image_id"])].append(row)
    output = []
    for image_id, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda row: (-float(row["hint_utility"]), str(row["hint_id"])))
        for chosen_index, chosen in enumerate(ranked):
            for rejected in ranked[chosen_index + 1 :]:
                margin = float(chosen["hint_utility"]) - float(rejected["hint_utility"])
                if margin < min_margin:
                    continue
                if require_dominance and not dominates(chosen, rejected):
                    continue
                output.append(
                    {
                        "image_id": image_id,
                        "image": chosen["image"],
                        "chosen_hint": chosen["hint"],
                        "rejected_hint": rejected["hint"],
                        "chosen_hint_id": chosen["hint_id"],
                        "rejected_hint_id": rejected["hint_id"],
                        "chosen_hint_utility": chosen["hint_utility"],
                        "rejected_hint_utility": rejected["hint_utility"],
                        "utility_margin": margin,
                        "chosen_utility_dimensions": chosen["utility_dimensions"],
                        "rejected_utility_dimensions": rejected["utility_dimensions"],
                        "chosen_generated_captions": chosen["generated_captions"],
                        "rejected_generated_captions": rejected["generated_captions"],
                        "preference_basis": "frozen_generator_mean_downstream_caption_utility",
                    }
                )
    return output
