#!/usr/bin/env python3
"""Map canonical blinded rater votes to policies and report robust consensus."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def policy(system: str) -> str:
    match = re.fullmatch(r"(.+)_s\d+", system)
    if not match:
        raise ValueError(f"invalid seeded system {system!r}")
    return match.group(1)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values); pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bootstrap(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed); n = len(values)
    draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def vote(values: list[str]) -> str:
    counts = Counter(values); ranked = counts.most_common()
    return ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else "unresolved"


def summarize(name: str, rater_names: list[str], raters: dict[str, dict[str, Any]], keys: dict[str, Any], public: dict[str, Any], samples: int, seed: int) -> dict[str, Any]:
    trials = []
    for pair_id in sorted(keys):
        key = keys[pair_id]; overall = vote([raters[r][pair_id]["overall"] for r in rater_names])
        mapped = "unresolved" if overall == "unresolved" else "tie" if overall == "Tie" else policy(key[f"group_{overall}_system"])
        absolute = {}
        for side in "AB":
            label = vote([raters[r][pair_id][f"absolute_{side}"] for r in rater_names])
            absolute[policy(key[f"group_{side}_system"])] = label
        trials.append({"pair_id": pair_id, "image_id": public[pair_id]["image_id"], "seed": int(re.search(r"_s(\d+)$", key["group_A_system"]).group(1)), "winner": mapped, "absolute": absolute})
    def score(winner: str) -> float:
        return 1.0 if winner == "dpo" else 0.0 if winner == "sft" else 0.5
    by_seed, by_image = defaultdict(list), defaultdict(list)
    for row in trials: by_seed[row["seed"]].append(row); by_image[row["image_id"]].append(row)
    seed_stats = {}
    for generation_seed, rows in sorted(by_seed.items()):
        counts = Counter(row["winner"] for row in rows)
        resolved = [row for row in rows if row["winner"] != "unresolved"]
        seed_stats[str(generation_seed)] = {
            "counts": {label: counts[label] for label in ("dpo", "tie", "sft", "unresolved")},
            "neutral_imputed_dpo_score": statistics.mean(score(row["winner"]) for row in rows),
            "resolved_only_dpo_score": statistics.mean(score(row["winner"]) for row in resolved) if resolved else None,
        }
    image_scores = [statistics.mean(score(row["winner"]) for row in rows) for rows in by_image.values()]
    counts = Counter(row["winner"] for row in trials); resolved = [row for row in trials if row["winner"] != "unresolved"]
    absolute_counts = {p: Counter(row["absolute"].get(p, "unresolved") for row in trials) for p in ("dpo", "sft")}
    majority_good = {}
    for p in ("dpo", "sft"):
        majority_good[p] = sum(sum(row["absolute"].get(p) == "good" for row in rows) >= 2 for rows in by_image.values())
    return {
        "name": name, "raters": rater_names, "trials": len(trials),
        "counts": {label: counts[label] for label in ("dpo", "tie", "sft", "unresolved")},
        "neutral_imputed_dpo_score": statistics.mean(score(row["winner"]) for row in trials),
        "resolved_only_dpo_score": statistics.mean(score(row["winner"]) for row in resolved),
        "unresolved_sensitivity_bounds": [sum(score(r["winner"]) if r["winner"] != "unresolved" else 0 for r in trials) / len(trials), sum(score(r["winner"]) if r["winner"] != "unresolved" else 1 for r in trials) / len(trials)],
        "image_clustered_neutral_95ci": bootstrap(image_scores, samples, seed),
        "per_seed": seed_stats,
        "absolute_group_counts": {p: dict(absolute_counts[p]) for p in absolute_counts},
        "majority_good_images": majority_good,
        "trial_results": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--rater", action="append", required=True, metavar="NAME=CANONICAL_JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    args = parser.parse_args()
    public = {r["pair_id"]: r for r in read_jsonl(args.public)}
    keys = {r["pair_id"]: r for r in json.load(args.key.open())["key"]}
    raters = {}
    for spec in args.rater:
        name, path = spec.split("=", 1); raters[name] = json.load(Path(path).open())["decisions"]
    if any(set(rows) != set(keys) for rows in raters.values()) or set(public) != set(keys):
        raise ValueError("public/key/rater pair IDs differ")
    names = list(raters)
    report = {"all_three_majority": summarize("all_three_majority", names, raters, keys, public, args.bootstrap_samples, args.bootstrap_seed)}
    if "codex" in raters and "llm_judge_2" in raters:
        report["position_unbiased_two_judge_agreement"] = summarize("position_unbiased_two_judge_agreement", ["codex", "llm_judge_2"], raters, keys, public, args.bootstrap_samples, args.bootstrap_seed + 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    compact = {k: {x: y for x, y in v.items() if x != "trial_results"} for k, v in report.items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
