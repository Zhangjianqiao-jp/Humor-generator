"""Anonymous Group-of-3 evaluation with image-clustered uncertainty.

The relative decision is made over two groups of three captions.  Absolute
labels are also collected per candidate so a relative win cannot be reported
as an intrinsically good caption and generation-seed variance is estimable.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import random
from typing import Any, Iterable


LABEL_VALUE = {"bad": 0.0, "weak": 0.5, "good": 1.0}


def _blind_id(receiver: str, cluster: str, challenger: str) -> str:
    raw = f"v3-group3\0{receiver}\0{cluster}\0{challenger}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def build_group3_packets(
    generations: Iterable[dict[str, Any]],
    *,
    reference: str = "text_homer",
    seed: int = 20260830,
    include_standard_description: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return public blind packets and a private condition mapping.

    Each receiver/condition/image must have exactly three distinct generation
    seeds. Every non-reference condition is compared against the same receiver's
    reference. A/B orientation is deterministic but hidden from raters.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    cluster_meta: dict[tuple[str, str], dict[str, str]] = {}
    for row in generations:
        key = (str(row["receiver"]), str(row["condition"]), str(row["cluster_id"]))
        grouped[key].append(dict(row))
        cluster_meta[(key[0], key[2])] = {
            "image": str(row["image"]),
            "standard_description": str(row.get("standard_description", "")),
        }

    packets: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    receivers = sorted({key[0] for key in grouped})
    for receiver in receivers:
        conditions = sorted({key[1] for key in grouped if key[0] == receiver})
        if reference not in conditions:
            raise ValueError(f"receiver {receiver!r} has no {reference!r} reference")
        challengers = [value for value in conditions if value != reference]
        reference_clusters = {key[2] for key in grouped if key[:2] == (receiver, reference)}
        for challenger in challengers:
            challenger_clusters = {key[2] for key in grouped if key[:2] == (receiver, challenger)}
            if challenger_clusters != reference_clusters:
                raise ValueError(
                    f"cluster mismatch for {receiver}/{challenger}: "
                    f"reference={len(reference_clusters)} challenger={len(challenger_clusters)}"
                )
            for cluster in sorted(reference_clusters):
                candidates: dict[str, list[dict[str, Any]]] = {}
                for condition in (reference, challenger):
                    values = sorted(
                        grouped[(receiver, condition, cluster)],
                        key=lambda item: int(item["generation_seed"]),
                    )
                    seeds = [int(item["generation_seed"]) for item in values]
                    if len(values) != 3 or len(set(seeds)) != 3:
                        raise ValueError(
                            f"Group-of-3 requires exactly three unique seeds: "
                            f"{receiver}/{condition}/{cluster} has {seeds}"
                        )
                    candidates[condition] = values
                blind_id = _blind_id(receiver, cluster, challenger)
                flip = random.Random(f"{seed}:{blind_id}").randrange(2) == 1
                a_condition, b_condition = (
                    (challenger, reference) if flip else (reference, challenger)
                )
                meta = cluster_meta[(receiver, cluster)]
                packet = {
                    "blind_id": blind_id,
                    "image": meta["image"],
                    "group_A": [item["caption"] for item in candidates[a_condition]],
                    "group_B": [item["caption"] for item in candidates[b_condition]],
                }
                if include_standard_description:
                    packet["standard_description"] = meta["standard_description"]
                packets.append(packet)
                mapping.append({
                    "blind_id": blind_id,
                    "receiver": receiver,
                    "cluster_id": cluster,
                    "condition_A": a_condition,
                    "condition_B": b_condition,
                    "seeds_A": [int(item["generation_seed"]) for item in candidates[a_condition]],
                    "seeds_B": [int(item["generation_seed"]) for item in candidates[b_condition]],
                })
    return packets, mapping


def _bootstrap_ci(values: list[float], *, seed: int, samples: int = 10_000) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def aggregate_group3(
    mapping: Iterable[dict[str, Any]],
    ratings: Iterable[dict[str, Any]],
    *,
    seed: int = 20260830,
) -> dict[str, Any]:
    """Aggregate independent raters with clusters—not captions—as sample units."""
    mapping_by_id = {item["blind_id"]: dict(item) for item in mapping}
    if not mapping_by_id:
        raise ValueError("empty blind mapping")
    observations: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    absolute: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seed_absolute: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    rater_coverage: dict[str, int] = {}
    invalid: list[str] = []

    for payload in ratings:
        rater = str(payload["rater_id"])
        decisions = payload.get("decisions", {})
        rater_coverage[rater] = len(decisions)
        for blind_id, decision in decisions.items():
            if blind_id not in mapping_by_id:
                invalid.append(f"{rater}:unknown:{blind_id}")
                continue
            item = mapping_by_id[blind_id]
            overall = decision.get("overall")
            if overall not in {"A", "B", "Tie"}:
                invalid.append(f"{rater}:overall:{blind_id}")
                continue
            challenger = (
                item["condition_A"]
                if item["condition_B"] == "text_homer"
                else item["condition_B"]
            )
            challenger_side = "A" if item["condition_A"] == challenger else "B"
            value = 0.5 if overall == "Tie" else float(overall == challenger_side)
            observations[(item["receiver"], challenger, item["cluster_id"])].append(value)

            for side in ("A", "B"):
                condition = item[f"condition_{side}"]
                group_label = decision.get(f"absolute_{side}")
                if group_label in LABEL_VALUE:
                    absolute[(item["receiver"], challenger, condition)].append(LABEL_VALUE[group_label])
                labels = decision.get(f"candidate_labels_{side}")
                if labels is None:
                    continue
                if not isinstance(labels, list) or len(labels) != 3 or any(
                    label not in LABEL_VALUE for label in labels
                ):
                    invalid.append(f"{rater}:candidate_labels_{side}:{blind_id}")
                    continue
                for generation_seed, label in zip(item[f"seeds_{side}"], labels):
                    seed_absolute[(item["receiver"], challenger, condition, int(generation_seed))].append(
                        LABEL_VALUE[label]
                    )

    results: list[dict[str, Any]] = []
    pairs = sorted({key[:2] for key in observations})
    for pair_index, (receiver, challenger) in enumerate(pairs):
        cluster_values = [
            sum(values) / len(values)
            for (rec, cond, _cluster), values in observations.items()
            if (rec, cond) == (receiver, challenger)
        ]
        if not cluster_values:
            continue
        condition_summaries = {}
        for condition in ("text_homer", challenger):
            group_values = absolute.get((receiver, challenger, condition), [])
            candidate_values = [
                value
                for (rec, chal, cond, _generation_seed), values in seed_absolute.items()
                if (rec, chal, cond) == (receiver, challenger, condition)
                for value in values
            ]
            per_seed = {
                str(generation_seed): sum(values) / len(values)
                for (rec, chal, cond, generation_seed), values in seed_absolute.items()
                if (rec, chal, cond) == (receiver, challenger, condition)
            }
            seed_variance = None
            if len(per_seed) >= 2:
                mean = sum(per_seed.values()) / len(per_seed)
                seed_variance = sum((value - mean) ** 2 for value in per_seed.values()) / (len(per_seed) - 1)
            condition_summaries[condition] = {
                "absolute_group_score_mean_bad0_weak0.5_good1": (
                    sum(group_values) / len(group_values) if group_values else None
                ),
                "candidate_good_rate": (
                    sum(value == 1.0 for value in candidate_values) / len(candidate_values)
                    if candidate_values else None
                ),
                "candidate_weak_rate": (
                    sum(value == 0.5 for value in candidate_values) / len(candidate_values)
                    if candidate_values else None
                ),
                "candidate_bad_rate": (
                    sum(value == 0.0 for value in candidate_values) / len(candidate_values)
                    if candidate_values else None
                ),
                "per_generation_seed_absolute_score": per_seed or None,
                "generation_seed_sample_variance": seed_variance,
                "seed_variance_available": seed_variance is not None,
            }
        results.append({
            "receiver": receiver,
            "challenger": challenger,
            "reference": "text_homer",
            "image_clusters": len(cluster_values),
            "rater_averaged_win_rate_ties_half": sum(cluster_values) / len(cluster_values),
            "image_cluster_bootstrap_95_ci": _bootstrap_ci(
                cluster_values, seed=seed + pair_index
            ),
            "absolute_quality": condition_summaries,
        })
    return {
        "schema_version": 1,
        "statistical_unit": "image_cluster",
        "tie_policy": "0.5 win",
        "absolute_scale": LABEL_VALUE,
        "rater_coverage": rater_coverage,
        "invalid_decisions": invalid,
        "comparisons": results,
    }
