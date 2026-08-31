"""Blinded Group-of-N evaluation with clustered, multi-rater inference."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random
from typing import Any, Iterable

LABEL_VALUE = {"bad": 0.0, "weak": 0.5, "good": 1.0}
DECISIONS = {"A", "B", "Tie"}


def _blind_id(
    receiver: str, cluster: str, reference: str, challenger: str,
    group_size: int, family: str, orientation: int,
) -> str:
    raw = (
        f"v35-group{group_size}\0{family}\0{receiver}\0{cluster}\0"
        f"{reference}\0{challenger}\0{orientation}"
    ).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def build_group3_packets(
    generations: Iterable[dict[str, Any]], *,
    comparisons: Iterable[tuple[str, str]] | None = None,
    reference: str = "text_homer", seed: int = 20260830,
    include_standard_description: bool = False, group_size: int = 3,
    comparison_family: str = "unspecified", mirror_sides: bool = False,
    calibration_examples: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build anonymous group packets; `group_size=10` matches Humor in AI."""
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    calibration_hash = None
    if calibration_examples is not None:
        if len(calibration_examples) != 5:
            raise ValueError("Humor-in-AI judge calibration requires exactly five examples")
        required = {"image", "caption_A", "caption_B", "answer"}
        for index, example in enumerate(calibration_examples):
            if set(example) != required or example["answer"] not in {"A", "B"}:
                raise ValueError(f"invalid calibration example {index}")
            if any(not str(example[name]).strip() for name in required):
                raise ValueError(f"empty calibration field in example {index}")
        calibration_hash = hashlib.sha256(
            json.dumps(calibration_examples, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    cluster_meta: dict[tuple[str, str], dict[str, str]] = {}
    for source in generations:
        row = dict(source)
        key = (str(row["receiver"]), str(row["condition"]), str(row["cluster_id"]))
        grouped[key].append(row)
        proposed_meta = {
            "image": str(row["image"]),
            "standard_description": str(row.get("standard_description", "")),
            "split": str(row.get("split", "unspecified")),
        }
        previous = cluster_meta.get((key[0], key[2]))
        if previous is not None and previous != proposed_meta:
            raise ValueError(f"inconsistent metadata across conditions for {key[0]}/{key[2]}")
        cluster_meta[(key[0], key[2])] = proposed_meta
    packets, mapping = [], []
    for receiver in sorted({key[0] for key in grouped}):
        conditions = sorted({key[1] for key in grouped if key[0] == receiver})
        pairs = list(comparisons or [(reference, value) for value in conditions if value != reference])
        for reference_condition, challenger in pairs:
            if reference_condition == challenger:
                raise ValueError("reference and challenger must differ")
            if reference_condition not in conditions or challenger not in conditions:
                continue
            ref_clusters = {key[2] for key in grouped if key[:2] == (receiver, reference_condition)}
            chal_clusters = {key[2] for key in grouped if key[:2] == (receiver, challenger)}
            if ref_clusters != chal_clusters:
                raise ValueError(
                    f"cluster mismatch for {receiver}/{reference_condition}/{challenger}: "
                    f"reference={len(ref_clusters)} challenger={len(chal_clusters)}"
                )
            for cluster in sorted(ref_clusters):
                candidates: dict[str, list[dict[str, Any]]] = {}
                for condition in (reference_condition, challenger):
                    values = sorted(grouped[(receiver, condition, cluster)], key=lambda x: int(x["generation_seed"]))
                    seeds = [int(item["generation_seed"]) for item in values]
                    if len(values) != group_size or len(set(seeds)) != group_size:
                        raise ValueError(
                            f"Group-of-{group_size} requires exactly {group_size} unique seeds: "
                            f"{receiver}/{condition}/{cluster} has {seeds}"
                        )
                    candidates[condition] = values
                pair_id = _blind_id(
                    receiver, cluster, reference_condition, challenger,
                    group_size, comparison_family, 0,
                )
                rng = random.Random(f"{seed}:{pair_id}")
                first = (
                    (challenger, reference_condition) if rng.randrange(2)
                    else (reference_condition, challenger)
                )
                orientations = [first, tuple(reversed(first))] if mirror_sides else [first]
                for orientation, (a_condition, b_condition) in enumerate(orientations):
                    blind_id = _blind_id(
                        receiver, cluster, reference_condition, challenger,
                        group_size, comparison_family, orientation,
                    )
                    ordered: dict[str, list[dict[str, Any]]] = {}
                    for side, condition in (("A", a_condition), ("B", b_condition)):
                        ordered[side] = candidates[condition][:]
                        random.Random(f"{seed}:{blind_id}:{side}").shuffle(ordered[side])
                    meta = cluster_meta[(receiver, cluster)]
                    packet: dict[str, Any] = {
                        "blind_id": blind_id, "image": meta["image"],
                        "group_A": [item["caption"] for item in ordered["A"]],
                        "group_B": [item["caption"] for item in ordered["B"]],
                    }
                    if calibration_examples is not None:
                        packet["calibration_examples"] = calibration_examples
                    if include_standard_description:
                        packet["standard_description"] = meta["standard_description"]
                    packets.append(packet)
                    mapping.append({
                        "blind_id": blind_id, "mirror_pair_id": pair_id,
                        "orientation": orientation, "comparison_family": comparison_family,
                        "receiver": receiver, "cluster_id": cluster, "split": meta["split"],
                        "reference": reference_condition, "challenger": challenger,
                        "condition_A": a_condition, "condition_B": b_condition,
                        "group_size": group_size,
                        "calibration_sha256": calibration_hash,
                        "seeds_A": [int(item["generation_seed"]) for item in ordered["A"]],
                        "seeds_B": [int(item["generation_seed"]) for item in ordered["B"]],
                    })
    if not packets:
        raise ValueError("no complete requested comparisons were found")
    return packets, mapping


def _hierarchical_ci(by_cluster: dict[str, list[float]], *, seed: int, samples: int = 10_000) -> list[float]:
    clusters = sorted(by_cluster)
    if not clusters or any(not by_cluster[value] for value in clusters):
        raise ValueError("cannot bootstrap empty cluster ratings")
    rng, means = random.Random(seed), []
    for _ in range(samples):
        sampled = [rng.choice(clusters) for _ in clusters]
        cluster_means = [
            sum(rng.choice(by_cluster[cluster]) for _ in by_cluster[cluster])
            / len(by_cluster[cluster])
            for cluster in sampled
        ]
        means.append(sum(cluster_means) / len(cluster_means))
    means.sort()
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def _sign_flip_pvalue(by_cluster: dict[str, list[float]], *, seed: int, samples: int = 20_000) -> float:
    effects = [sum(values) / len(values) - 0.5 for values in by_cluster.values()]
    observed = abs(sum(effects) / len(effects))
    rng, extreme = random.Random(seed), 0
    for _ in range(samples):
        value = abs(sum(effect * (-1 if rng.randrange(2) else 1) for effect in effects) / len(effects))
        extreme += value >= observed - 1e-12
    return (extreme + 1) / (samples + 1)


def _krippendorff_nominal(items: dict[str, list[str]]) -> float | None:
    usable = [values for values in items.values() if len(values) >= 2]
    if not usable:
        return None
    observed_pairs = disagreements = 0.0
    pooled = Counter(value for values in usable for value in values)
    for values in usable:
        observed_pairs += len(values) * (len(values) - 1)
        disagreements += sum(left != right for left in values for right in values)
    total = sum(pooled.values())
    expected = 1.0 - sum((count / total) ** 2 for count in pooled.values())
    observed = disagreements / observed_pairs
    return None if expected <= 0 else 1.0 - observed / expected


def _holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, running = [1.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def aggregate_group3(
    mapping: Iterable[dict[str, Any]], ratings: Iterable[dict[str, Any]], *,
    seed: int = 20260830, strict: bool = True,
) -> dict[str, Any]:
    """Aggregate overall and best-pick outcomes at image-cluster level."""
    mapping_list = [dict(item) for item in mapping]
    mapping_by_id = {item["blind_id"]: item for item in mapping_list}
    if not mapping_by_id:
        raise ValueError("empty blind mapping")
    if len(mapping_by_id) != len(mapping_list):
        raise ValueError("duplicate blind_id in mapping")
    split_by_cluster = {
        (
            str(item.get("comparison_family", "unspecified")), item["receiver"],
            item["reference"], item["challenger"], int(item.get("group_size", 3)),
            item["cluster_id"],
        ): str(item.get("split", "unspecified"))
        for item in mapping_list
    }
    observations: dict[tuple[str, str, str, str, int, str, str], list[float]] = defaultdict(list)
    categorical: dict[tuple[str, str, str, str, int, str, str], list[str]] = defaultdict(list)
    absolute: dict[tuple[str, str, str, str, int, str], list[float]] = defaultdict(list)
    seed_absolute: dict[tuple[str, str, str, str, int, str, int], list[float]] = defaultdict(list)
    seed_absolute_category: dict[tuple[str, str, str, str, int, str, int], list[str]] = defaultdict(list)
    raw_observations: dict[tuple[tuple[str, ...], str, str], list[float]] = defaultdict(list)
    raw_absolute: dict[tuple[tuple[str, ...], str, str], list[float]] = defaultdict(list)
    raw_seed_absolute: dict[tuple[tuple[Any, ...], str, str], list[float]] = defaultdict(list)
    rater_coverage, rater_metadata, invalid = {}, {}, []
    seen_raters: set[str] = set()
    required_metadata = {"provider", "model", "version_or_date", "temperature", "prompt_sha256"}
    for payload in ratings:
        rater = str(payload["rater_id"])
        if rater in seen_raters:
            invalid.append(f"{rater}:duplicate_rater_id")
        seen_raters.add(rater)
        decisions = payload.get("decisions", {})
        rater_coverage[rater] = len(decisions)
        rater_metadata[rater] = payload.get("judge_metadata")
        metadata = payload.get("judge_metadata")
        if not isinstance(metadata, dict) or not required_metadata.issubset(metadata):
            invalid.append(f"{rater}:missing_or_incomplete_judge_metadata")
        missing_decisions = sorted(set(mapping_by_id) - set(decisions))
        if missing_decisions:
            invalid.append(f"{rater}:missing_{len(missing_decisions)}_decisions")
        for blind_id, decision in decisions.items():
            if blind_id not in mapping_by_id:
                invalid.append(f"{rater}:unknown:{blind_id}")
                continue
            item = mapping_by_id[blind_id]
            reference, challenger = item["reference"], item["challenger"]
            family = str(item.get("comparison_family", "unspecified"))
            challenger_side = "A" if item["condition_A"] == challenger else "B"
            group_size = int(item.get("group_size", 3))
            for side in ("A", "B"):
                best_index = decision.get(f"best_{side}_index")
                if not isinstance(best_index, int) or not 1 <= best_index <= group_size:
                    invalid.append(f"{rater}:best_{side}_index:{blind_id}")
            for metric in ("overall", "best_pick"):
                choice = decision.get(metric)
                if choice not in DECISIONS:
                    invalid.append(f"{rater}:{metric}:{blind_id}")
                    continue
                value = 0.5 if choice == "Tie" else float(choice == challenger_side)
                key = (
                    family, item["receiver"], reference, challenger, group_size,
                    metric, item["cluster_id"],
                )
                mirror_id = str(item.get("mirror_pair_id", blind_id))
                raw_observations[(key, rater, mirror_id)].append(value)
            for side in ("A", "B"):
                condition = item[f"condition_{side}"]
                group_label = decision.get(f"absolute_{side}")
                if group_label in LABEL_VALUE:
                    key = (family, item["receiver"], reference, challenger, group_size, condition)
                    raw_absolute[(key, rater, str(item.get("mirror_pair_id", blind_id)))].append(
                        LABEL_VALUE[group_label]
                    )
                else:
                    invalid.append(f"{rater}:absolute_{side}:{blind_id}")
                labels = decision.get(f"candidate_labels_{side}")
                if not isinstance(labels, list) or len(labels) != group_size or any(label not in LABEL_VALUE for label in labels):
                    invalid.append(f"{rater}:candidate_labels_{side}:{blind_id}")
                    continue
                for generation_seed, label in zip(item[f"seeds_{side}"], labels):
                    key = (
                        family, item["receiver"], reference, challenger, group_size,
                        condition, int(generation_seed),
                    )
                    raw_seed_absolute[(key, rater, str(item.get("mirror_pair_id", blind_id)))].append(
                        LABEL_VALUE[label]
                    )

    if not seen_raters:
        raise ValueError("at least one complete rater is required")
    if strict and invalid:
        raise ValueError("invalid blind ratings: " + "; ".join(invalid[:20]))
    # Mirrored A/B packets are a position-bias diagnostic, not independent
    # samples. Collapse each rater × image × comparison mirror pair first.
    for (key, _rater, _mirror_id), values in raw_observations.items():
        value = sum(values) / len(values)
        observations[key].append(value)
        categorical[key].append(
            "challenger" if value > 0.5 else "reference" if value < 0.5 else "tie"
        )
    for (key, _rater, _mirror_id), values in raw_absolute.items():
        absolute[key].append(sum(values) / len(values))
    for (key, _rater, _mirror_id), values in raw_seed_absolute.items():
        seed_absolute[key].append(sum(values) / len(values))
        seed_absolute_category[key].append(
            "mirror_disagreement" if len(set(values)) > 1 else (
                "good" if values[0] == 1.0 else "weak" if values[0] == 0.5 else "bad"
            )
        )
    comparison_keys = sorted({key[:5] for key in observations})
    results = []
    for pair_index, (family, receiver, reference, challenger, group_size) in enumerate(comparison_keys):
        metrics = {}
        for metric in ("overall", "best_pick"):
            by_cluster = {
                cluster: values
                for (fam, rec, ref, chal, size, met, cluster), values in observations.items()
                if (fam, rec, ref, chal, size, met) == (
                    family, receiver, reference, challenger, group_size, metric
                )
            }
            if not by_cluster:
                continue
            score = sum(sum(values) / len(values) for values in by_cluster.values()) / len(by_cluster)
            by_split = {}
            split_names = sorted({
                split_by_cluster[(
                    family, receiver, reference, challenger, group_size, cluster
                )]
                for cluster in by_cluster
            })
            for split_index, split in enumerate(split_names):
                split_values = {
                    cluster: values for cluster, values in by_cluster.items()
                    if split_by_cluster[(
                        family, receiver, reference, challenger, group_size, cluster
                    )] == split
                }
                split_score = (
                    sum(sum(values) / len(values) for values in split_values.values())
                    / len(split_values)
                )
                by_split[split] = {
                    "image_clusters": len(split_values),
                    "win_rate_ties_half": split_score,
                    "cluster_rater_bootstrap_95_ci": _hierarchical_ci(
                        split_values,
                        seed=seed + pair_index * 101 + split_index * 7
                        + int(metric == "best_pick"),
                    ),
                }
            metrics[metric] = {
                "rater_averaged_win_rate_ties_half": score,
                "hierarchical_cluster_rater_bootstrap_95_ci": _hierarchical_ci(
                    by_cluster, seed=seed + pair_index * 17 + int(metric == "best_pick")
                ),
                "two_sided_cluster_sign_flip_p": _sign_flip_pvalue(
                    by_cluster, seed=seed + pair_index * 31 + int(metric == "best_pick")
                ),
                "krippendorff_alpha_nominal": _krippendorff_nominal({
                    cluster: categorical[(family, receiver, reference, challenger, group_size, metric, cluster)]
                    for cluster in by_cluster
                }),
                "by_split": by_split,
            }
        condition_summaries = {}
        for condition in (reference, challenger):
            group_values = absolute.get((family, receiver, reference, challenger, group_size, condition), [])
            candidate_values = [
                value for (fam, rec, ref, chal, size, cond, _), values in seed_absolute.items()
                if (fam, rec, ref, chal, size, cond) == (
                    family, receiver, reference, challenger, group_size, condition
                )
                for value in values
            ]
            candidate_categories = [
                value
                for (fam, rec, ref, chal, size, cond, _), values in seed_absolute_category.items()
                if (fam, rec, ref, chal, size, cond) == (
                    family, receiver, reference, challenger, group_size, condition
                )
                for value in values
            ]
            per_seed = {
                str(generation_seed): sum(values) / len(values)
                for (fam, rec, ref, chal, size, cond, generation_seed), values in seed_absolute.items()
                if (fam, rec, ref, chal, size, cond) == (
                    family, receiver, reference, challenger, group_size, condition
                )
            }
            variance = None
            if len(per_seed) >= 2:
                mean = sum(per_seed.values()) / len(per_seed)
                variance = sum((value - mean) ** 2 for value in per_seed.values()) / (len(per_seed) - 1)
            condition_summaries[condition] = {
                "absolute_group_score_mean_bad0_weak0.5_good1": sum(group_values) / len(group_values) if group_values else None,
                "candidate_good_rate": candidate_categories.count("good") / len(candidate_categories) if candidate_categories else None,
                "candidate_weak_rate": candidate_categories.count("weak") / len(candidate_categories) if candidate_categories else None,
                "candidate_bad_rate": candidate_categories.count("bad") / len(candidate_categories) if candidate_categories else None,
                "candidate_mirror_disagreement_rate": candidate_categories.count("mirror_disagreement") / len(candidate_categories) if candidate_categories else None,
                "per_generation_seed_absolute_score": per_seed or None,
                "generation_seed_sample_variance": variance,
            }
        results.append({
            "comparison_family": family, "receiver": receiver,
            "reference": reference, "challenger": challenger,
            "group_size": group_size,
            "image_clusters": len({
                key[6] for key in observations
                if key[:5] == (family, receiver, reference, challenger, group_size)
            }),
            "relative_metrics": metrics, "absolute_quality": condition_summaries,
        })
    families: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, result in enumerate(results):
        families[(result["comparison_family"], result["receiver"], result["group_size"])].append(index)
    for family, indices in families.items():
        adjusted = _holm_adjust([
            results[index]["relative_metrics"].get("overall", {}).get(
                "two_sided_cluster_sign_flip_p", 1.0
            )
            for index in indices
        ])
        for index, value in zip(indices, adjusted):
            results[index]["overall_holm_adjusted_p"] = value
            results[index]["holm_family"] = {
                "comparison_family": family[0], "receiver": family[1],
                "group_size": family[2], "comparisons": len(indices)
            }
    return {
        "schema_version": 3, "statistical_unit": "image_cluster",
        "uncertainty": "cluster resampling plus within-cluster rater resampling",
        "mirror_policy": "collapse A/B orientations within rater × image × comparison before inference",
        "tie_policy": "0.5 win",
        "multiple_comparison_control": "Holm adjustment over overall comparison p-values",
        "absolute_scale": LABEL_VALUE, "rater_coverage": rater_coverage,
        "rater_metadata": rater_metadata, "invalid_decisions": invalid,
        "comparisons": results,
    }
