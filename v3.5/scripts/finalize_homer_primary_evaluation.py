#!/usr/bin/env python3
"""Finalize the exact-HOMER primary comparison after external A/B judgments.

The command is intentionally fail-closed: the aggregator must receive the
complete public-packet judgment set before any Pass@K or latent-vs-text number
is written.  It uses HOMER's unbiased estimator
``1 - C(n-c, k) / C(n, k)`` and a paired image-level bootstrap for the
condition difference.  It never calls an evaluator and never invents missing
decisions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any

try:  # Works both as ``python scripts/...`` and as a pytest package import.
    from .aggregate_homer_primary_judgments import aggregate
except ImportError:  # pragma: no cover - exercised by the CLI entry point.
    from aggregate_homer_primary_judgments import aggregate


SYSTEMS = ("latent_bridge", "text_homer_context_replay")
GROUPS = ("1-9", "200-209", "1000-1009")
KS = (1, 3, 5)


def pass_at_k(n: int, c: int, k: int) -> float:
    if not (1 <= k <= n) or not (0 <= c <= n):
        raise ValueError(f"invalid Pass@K inputs n={n}, c={c}, k={k}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    values = sorted(values)
    return values[int(round((len(values) - 1) * q))]


def bootstrap_mean(values: list[float], *, seed: int, replicates: int) -> tuple[float, float]:
    if not values or replicates < 1:
        raise ValueError("bootstrap requires non-empty values and positive replicates")
    rng = random.Random(seed)
    draws = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(replicates)]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def paired_bootstrap(values: list[float], *, seed: int, replicates: int) -> tuple[float, float]:
    return bootstrap_mean(values, seed=seed, replicates=replicates)


def validate_records(records: list[dict[str, Any]]) -> tuple[int, list[str], list[int], list[str]]:
    if not records:
        raise ValueError("aggregated HOMER records are empty")
    systems = sorted({row.get("system_id") for row in records})
    if systems != sorted(SYSTEMS):
        raise ValueError(f"expected systems {sorted(SYSTEMS)}, got {systems}")
    groups = sorted({row.get("official_reference_group") for row in records})
    if groups != sorted(GROUPS):
        raise ValueError(f"expected official reference groups {sorted(GROUPS)}, got {groups}")
    images = sorted({str(row.get("image_id")) for row in records})
    trials = sorted({int(row.get("trial")) for row in records})
    expected = {
        (system, image, trial, group, f"{group}/gt_caption{reference_index + 1}")
        for system in SYSTEMS
        for image in images
        for trial in trials
        for group in GROUPS
        for reference_index in range(5)
    }
    seen: set[tuple[str, str, int, str, str]] = set()
    for row in records:
        key = (
            str(row["system_id"]), str(row["image_id"]), int(row["trial"]),
            str(row["official_reference_group"]), str(row["reference_group"]),
        )
        if key in seen:
            raise ValueError(f"duplicate aggregated record: {key}")
        seen.add(key)
        if row.get("candidate_count") != 5:
            raise ValueError(f"HOMER primary requires candidate_count=5: {key}")
        if not 0 <= int(row.get("winning_caption_count", -1)) <= 5:
            raise ValueError(f"invalid winning_caption_count: {key}")
    if seen != expected:
        raise ValueError(f"incomplete HOMER grid: got={len(seen)} expected={len(expected)}")
    return 5, images, trials, list(GROUPS)


def compute(records: list[dict[str, Any]], *, bootstrap_seed: int, bootstrap_replicates: int) -> dict[str, Any]:
    n, images, trials, groups = validate_records(records)
    by_key = {
        (str(row["system_id"]), str(row["image_id"]), int(row["trial"]), str(row["official_reference_group"]), str(row["reference_group"])): row
        for row in records
    }
    summaries: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for group in groups:
        for k in KS:
            image_values: dict[str, dict[str, float]] = {image: {} for image in images}
            for system in SYSTEMS:
                for image in images:
                    trial_values = []
                    for trial in trials:
                        rows = [
                            row for key, row in by_key.items()
                            if key[0] == system and key[1] == image and key[2] == trial and key[3] == group
                        ]
                        if len(rows) != 5:
                            raise ValueError(f"expected five references for {system}/{image}/{trial}/{group}")
                        # The official evaluator averages Pass@K over the five
                        # human references; preserve that order exactly.
                        trial_values.append(sum(pass_at_k(n, int(row["winning_caption_count"]), k) for row in rows) / 5.0)
                    image_values[image][system] = sum(trial_values) / len(trial_values)
                vals = [image_values[image][system] for image in images]
                lo, hi = bootstrap_mean(vals, seed=bootstrap_seed + len(summaries) * 17 + k, replicates=bootstrap_replicates)
                summaries.append({
                    "system_id": system,
                    "official_reference_group": group,
                    "k": k,
                    "candidate_count": n,
                    "images": len(images),
                    "trials": len(trials),
                    "pass_at_k": sum(vals) / len(vals),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "per_trial": {
                        str(trial): sum(
                            sum(pass_at_k(n, int(row["winning_caption_count"]), k) for row in [r for key, r in by_key.items() if key[0] == system and key[1] == image and key[2] == trial and key[3] == group]) / 5.0
                            for image in images
                        ) / len(images)
                        for trial in trials
                    },
                })
            differences = [image_values[image]["latent_bridge"] - image_values[image]["text_homer_context_replay"] for image in images]
            lo, hi = paired_bootstrap(differences, seed=bootstrap_seed + 10000 + len(paired) * 19 + k, replicates=bootstrap_replicates)
            paired.append({
                "comparison": "latent_bridge_minus_text_homer_context_replay",
                "official_reference_group": group,
                "k": k,
                "images": len(images),
                "trials": len(trials),
                "mean_difference": sum(differences) / len(differences),
                "ci95_low": lo,
                "ci95_high": hi,
                "fraction_image_difference_gt_0": sum(value > 0 for value in differences) / len(differences),
            })
    return {
        "schema_version": 1,
        "protocol": "homer_unbiased_pass_at_k_paired_image_bootstrap",
        "statistical_unit": "image_id",
        "formula": "1 - C(n-c,k) / C(n,k)",
        "candidate_count": n,
        "k_values": list(KS),
        "systems": list(SYSTEMS),
        "reference_groups": list(groups),
        "images": len(images),
        "trials": len(trials),
        "bootstrap": {"seed": bootstrap_seed, "replicates": bootstrap_replicates, "hierarchy": "image"},
        "summaries": summaries,
        "paired_comparison": paired,
    }


def render_markdown(result: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines = [
        "# v3.5 HOMER primary comparison",
        "",
        "This report is generated only after an exact one-to-one external A/B judgment set.",
        "It uses the official unbiased Pass@K estimator and paired image-level bootstrap.",
        "",
        f"- Evaluation ID: `{provenance.get('evaluation_id')}`",
        f"- Evaluator: `{provenance.get('evaluator', {}).get('model_id')}` (temperature `{provenance.get('evaluator', {}).get('temperature')}`)",
        f"- Images: {result['images']}; trials: {result['trials']}; candidates: {result['candidate_count']}",
        "",
        "## Pass@K",
        "",
        "| System | Reference group | K | Pass@K | 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["summaries"]:
        lines.append(f"| {row['system_id']} | {row['official_reference_group']} | {row['k']} | {row['pass_at_k']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |")
    lines += ["", "## Paired latent minus text", "", "| Reference group | K | Mean difference | 95% CI | Images with positive difference |", "|---|---:|---:|---:|---:|"]
    for row in result["paired_comparison"]:
        lines.append(f"| {row['official_reference_group']} | {row['k']} | {row['mean_difference']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | {row['fraction_image_difference_gt_0']:.4f} |")
    lines += ["", "No causal or quality claim is implied by a non-significant difference; interpret alongside the generation and evaluator provenance.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize HOMER primary Pass@K comparison")
    parser.add_argument("--private-mapping", required=True, type=Path)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20250308)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    try:
        records_path = args.output_json.with_suffix(".records.jsonl")
        summary = aggregate(args.private_mapping, args.judgments, records_path)
        with records_path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        result = compute(records, bootstrap_seed=args.bootstrap_seed, bootstrap_replicates=args.bootstrap_replicates)
        result["aggregation"] = summary
        result["provenance"] = provenance
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result, provenance), encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2) from exc
    print(json.dumps({"status": "complete", "records": len(records), "output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
