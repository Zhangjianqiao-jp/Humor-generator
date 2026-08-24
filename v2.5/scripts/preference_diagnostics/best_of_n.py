#!/usr/bin/env python3
"""Generate or analyse nested Best-of-N caption pools.

Generation is deliberately separated from judging. A single N=max(N_VALUES)
pool is sampled per image, and smaller N values use prefixes of that same pool.
This makes the curves nested and avoids attributing independent sampling noise
to N. The scorer input may be the output of judge_sft_candidates_qwen.py or a
candidate JSONL whose candidates are objects containing the requested score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_lora_sft import run_generation
from src.preference.diagnostics import line_plot_png, mean, read_jsonl, sha256, write_csv, write_json

DEFAULT_N = (1, 2, 4, 8, 16, 32)


def candidate_scores(row: dict[str, Any], score_field: str) -> list[float]:
    judged = row.get("judged_candidates")
    if isinstance(judged, list):
        ordered = sorted(judged, key=lambda item: int(item.get("index", 0)))
        return [float(item[score_field]) for item in ordered if item.get(score_field) is not None]
    candidates = row.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        return [float(item[score_field]) for item in candidates if item.get(score_field) is not None]
    scores = row.get("scores")
    if isinstance(scores, list):
        return [float(value) for value in scores]
    return []


def analyse(
    score_rows: list[dict[str, Any]],
    n_values: tuple[int, ...],
    score_field: str,
    threshold: float,
    score_min: float = 1.0,
    score_max: float = 5.0,
) -> list[dict[str, Any]]:
    if not n_values or any(n < 1 for n in n_values) or tuple(sorted(set(n_values))) != n_values:
        raise ValueError("N values must be unique positive integers in ascending order")
    per_image = []
    for index, row in enumerate(score_rows):
        scores = candidate_scores(row, score_field)
        if any(not score_min <= score <= score_max for score in scores):
            raise ValueError(
                f"row {index} image_id={row.get('image_id')} has scores outside "
                f"[{score_min}, {score_max}]"
            )
        if len(scores) < max(n_values):
            raise ValueError(
                f"row {index} image_id={row.get('image_id')} has {len(scores)} scored candidates; "
                f"need at least {max(n_values)}"
            )
        per_image.append(scores)
    if not per_image:
        raise ValueError("score file contains no rows")
    results = []
    for n in n_values:
        maxima = [max(scores[:n]) for scores in per_image]
        candidate_means = [mean(scores[:n]) for scores in per_image]
        results.append(
            {
                "n": n,
                "images": len(per_image),
                "h_max": mean(maxima),
                "h_mean": mean(candidate_means),
                "p_good": mean([float(value >= threshold) for value in maxima]),
                "threshold": threshold,
                "score_field": score_field,
            }
        )
    return results


def parse_n_values(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/preference_diagnostics/best_of_n"))
    parser.add_argument("--n-values", default=",".join(map(str, DEFAULT_N)))
    parser.add_argument("--score-jsonl", type=Path, help="Judged candidate pool used for analysis.")
    parser.add_argument("--score-field", default="humor")
    parser.add_argument("--good-threshold", type=float, default=4.0)
    parser.add_argument("--score-min", type=float, default=1.0)
    parser.add_argument("--score-max", type=float, default=5.0)
    parser.add_argument("--generate", action="store_true", help="Generate the max-N candidate pool before analysis.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--temperature", type=float, default=None, help="Recorded override; edit config for generation.")
    parser.add_argument("--top-p", type=float, default=None, help="Recorded override; edit config for generation.")
    parser.add_argument("--top-k", type=int, default=None, help="Recorded only; current generator has no top-k argument.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--policy-scope", choices=("captioner", "planner", "joint"), default="captioner")
    args = parser.parse_args()
    n_values = parse_n_values(args.n_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = args.output_dir / "candidates.jsonl"

    if args.generate:
        missing = [name for name in ("config", "adapter", "input_jsonl") if getattr(args, name) is None]
        if missing:
            raise ValueError(f"--generate requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
        prompt_override = args.prompt
        if args.prompt_file is not None:
            if prompt_override is not None:
                raise ValueError("use only one of --prompt and --prompt-file")
            prompt_override = args.prompt_file.read_text(encoding="utf-8").strip()
        run_generation(
            config_path=args.config,
            adapter_dir=args.adapter,
            input_jsonl=args.input_jsonl,
            output_jsonl=generated_path,
            num_candidates=max(n_values),
            limit=args.limit,
            prompt_override=prompt_override,
            unique_images=True,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            prompt_template=None,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    manifest: dict[str, Any] = {
        "policy_scope": args.policy_scope,
        "nested_prefix_design": True,
        "n_values": list(n_values),
        "seed": args.seed,
        "config": None if args.config is None else str(args.config),
        "adapter": None if args.adapter is None else str(args.adapter),
        "input_jsonl": None if args.input_jsonl is None else str(args.input_jsonl),
        "candidate_jsonl": str(generated_path) if generated_path.exists() else None,
        "score_jsonl": None if args.score_jsonl is None else str(args.score_jsonl),
        "score_field": args.score_field,
        "good_threshold": args.good_threshold,
        "generation_overrides": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    if args.config is not None:
        resolved_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        manifest["resolved_generation_config"] = resolved_config.get("generation", {})
    for key in ("config", "input_jsonl", "score_jsonl"):
        path = getattr(args, key, None)
        if path is not None and path.exists():
            manifest[f"{key}_sha256"] = sha256(path)

    if args.score_jsonl is not None:
        results = analyse(
            read_jsonl(args.score_jsonl), n_values, args.score_field, args.good_threshold,
            score_min=args.score_min, score_max=args.score_max,
        )
        write_csv(args.output_dir / "best_of_n.csv", results)
        x = [float(row["n"]) for row in results]
        line_plot_png(
            args.output_dir / "best_of_n_humor.png",
            x,
            {"H_max": [float(row["h_max"]) for row in results], "H_mean": [float(row["h_mean"]) for row in results]},
            "Best-of-N humor capability",
            "N",
            args.score_field,
        )
        line_plot_png(
            args.output_dir / "best_of_n_success_rate.png",
            x,
            {"P_good": [float(row["p_good"]) for row in results]},
            "Fraction with at least one high-quality caption",
            "N",
            "success rate",
        )
        manifest["results"] = results
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        manifest["status"] = "candidate generation only; run an independent judge, then pass --score-jsonl"
        print(manifest["status"])
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
