#!/usr/bin/env python3
"""Reproduce the published Humor in AI preference-pair construction.

Zhang et al. (NeurIPS 2024) sample a chosen caption from the top half of a
contest ranking, a lower-ranked rejected caption, and retain the pair when
the mean-rating gap exceeds three times the combined reported precision.

This adapter keeps that published chosen/rejected rule while attaching the
image and compact-plan prompt used by this repository's Qwen2.5-VL captioner.
It creates data only; it never starts preference training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


PAPER = "Zhang et al., Humor in AI, NeurIPS 2024"
PAPER_URL = "https://arxiv.org/abs/2406.10522"
DATASET = "yguooo/newyorker_caption_ranking"
LICENSE = "CC-BY-NC-4.0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assistant_caption(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return content.strip()
        for item in content:
            if item.get("type") == "text":
                return str(item.get("text") or "").strip()
    raise ValueError(f"SFT row has no assistant caption: {row.get('image_id')}")


def user_prompt(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return content.strip()
        texts = [str(item.get("text") or "").strip() for item in content if item.get("type") == "text"]
        if texts:
            return "\n".join(text for text in texts if text)
    raise ValueError(f"SFT row has no user prompt: {row.get('image_id')}")


def load_contexts(path: Path) -> dict[int, dict[str, str]]:
    contexts: dict[int, dict[str, str]] = {}
    for row in read_jsonl(path):
        contest = int(row.get("meta", {}).get("contest_number") or str(row["image_id"]).split("_")[-1])
        context = {
            "image": str(row["image"]),
            "image_id": str(row["image_id"]),
            "prompt": user_prompt(row),
        }
        previous = contexts.setdefault(contest, context)
        if previous != context:
            raise ValueError(f"Contest {contest} has inconsistent image/prompt contexts in {path}")
    return contexts


def load_ranking(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            caption = str(row.get("caption") or "").strip()
            if not caption:
                continue
            rows.append(
                {
                    # Release files through contest 659 contain an explicit
                    # zero-based rank. Later files omit it but are already
                    # sorted by non-increasing mean score, so their CSV row
                    # index is the corresponding rank.
                    "rank": int(row["rank"]) if row.get("rank") not in (None, "") else row_index,
                    "caption": caption,
                    "mean": float(row["mean"]),
                    "precision": float(row["precision"]),
                    "votes": int(row["votes"]),
                    "funny_votes": int(row["funny"]),
                }
            )
    rows.sort(key=lambda row: row["rank"])
    if len(rows) < 2:
        raise ValueError(f"Ranking has fewer than two captions: {path}")
    return rows


def sample_published_pairs(
    ranking: list[dict[str, Any]],
    pair_count: int,
    max_attempts: int,
    rng: random.Random,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], int]:
    """Match the sampling and three-sigma filter in the published preprocess.py."""
    sampled: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    attempts = 0
    top_half_last = int(0.5 * len(ranking))
    while len(sampled) < pair_count and attempts < max_attempts:
        chosen_index = rng.randint(0, top_half_last)
        rejected_index = rng.randint(chosen_index + 1, len(ranking) - 1)
        chosen = ranking[chosen_index]
        rejected = ranking[rejected_index]
        uncertainty = math.sqrt(chosen["precision"] ** 2 + rejected["precision"] ** 2)
        margin = chosen["mean"] - rejected["mean"]
        attempts += 1
        if margin > 3.0 * uncertainty:
            z_margin = margin / uncertainty if uncertainty > 0 else float("inf")
            sampled.append((chosen, rejected, z_margin))
    return sampled, attempts


def relative_length_difference(left: str, right: str) -> float:
    denominator = max(len(left), len(right), 1)
    return abs(len(left) - len(right)) / denominator


def select_train_ready_pairs(
    rows: list[dict[str, Any]],
    per_contest: int,
    max_relative_length_difference: float,
) -> list[dict[str, Any]]:
    """Choose deduplicated, length-matched hard pairs without changing labels."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str, str]] = set()
    for row in rows:
        key = (int(row["contest_number"]), row["chosen"], row["rejected"])
        if key in seen:
            continue
        seen.add(key)
        if relative_length_difference(row["chosen"], row["rejected"]) > max_relative_length_difference:
            continue
        grouped[int(row["contest_number"])].append(row)

    selected: list[dict[str, Any]] = []
    for contest in sorted(grouped):
        # Lower z-margin pairs are harder while every candidate still passed
        # the published three-sigma confidence requirement.
        candidates = sorted(
            grouped[contest],
            key=lambda row: (row["z_margin"], row["score_margin"], row["chosen_rank"], row["rejected_rank"]),
        )
        for selection_index, source in enumerate(candidates[:per_contest]):
            row = dict(source)
            row["pair_id"] = f"published_nips24_selected_{row['source_split']}_{contest}_{selection_index:03d}"
            row["selection"] = {
                "view": "train_ready_hard_length_matched",
                "max_relative_length_difference": max_relative_length_difference,
                "order": "ascending_z_margin",
                "labels_unchanged": True,
            }
            selected.append(row)
    return selected


def build_split(
    split: str,
    context_path: Path,
    ranking_dir: Path,
    output_path: Path,
    pairs_per_contest: int,
    max_attempts: int,
    selected_per_contest: int,
    max_relative_length_difference: float,
    rng: random.Random,
) -> dict[str, Any]:
    contexts = load_contexts(context_path)
    output: list[dict[str, Any]] = []
    missing_rankings: list[int] = []
    attempts_by_contest: dict[str, int] = {}
    short_contests: dict[str, int] = {}
    unique_pairs: dict[int, set[tuple[str, str]]] = defaultdict(set)

    for contest in sorted(contexts):
        ranking_path = ranking_dir / f"{contest}.csv"
        if not ranking_path.exists():
            missing_rankings.append(contest)
            continue
        ranking = load_ranking(ranking_path)
        pairs, attempts = sample_published_pairs(ranking, pairs_per_contest, max_attempts, rng)
        attempts_by_contest[str(contest)] = attempts
        if len(pairs) < pairs_per_contest:
            short_contests[str(contest)] = len(pairs)
        context = contexts[contest]
        for sample_index, (chosen, rejected, z_margin) in enumerate(pairs):
            unique_pairs[contest].add((chosen["caption"], rejected["caption"]))
            output.append(
                {
                    "pair_id": f"published_nips24_{split}_{contest}_{sample_index:04d}",
                    "image": context["image"],
                    "image_id": context["image_id"],
                    "contest_number": contest,
                    "prompt": context["prompt"],
                    "chosen": chosen["caption"],
                    "rejected": rejected["caption"],
                    "chosen_score": chosen["mean"],
                    "rejected_score": rejected["mean"],
                    "score_margin": chosen["mean"] - rejected["mean"],
                    "chosen_precision": chosen["precision"],
                    "rejected_precision": rejected["precision"],
                    "z_margin": z_margin,
                    "chosen_rank": chosen["rank"],
                    "rejected_rank": rejected["rank"],
                    "chosen_votes": chosen["votes"],
                    "rejected_votes": rejected["votes"],
                    "pair_type": "H2",
                    "source": PAPER,
                    "source_url": PAPER_URL,
                    "source_dataset": DATASET,
                    "source_split": split,
                    "license": LICENSE,
                    "construction": "published_top_half_lower_rank_three_sigma",
                }
            )

    write_jsonl(output_path, output)
    selected = select_train_ready_pairs(
        output,
        per_contest=selected_per_contest,
        max_relative_length_difference=max_relative_length_difference,
    )
    selected_path = output_path.with_name(f"{output_path.stem}_selected.jsonl")
    write_jsonl(selected_path, selected)
    return {
        "split": split,
        "context_file": str(context_path),
        "context_sha256": sha256(context_path),
        "output_file": str(output_path),
        "output_sha256": sha256(output_path),
        "selected_output_file": str(selected_path),
        "selected_output_sha256": sha256(selected_path),
        "contests": len(contexts),
        "pairs": len(output),
        "unique_caption_pairs": sum(len(values) for values in unique_pairs.values()),
        "duplicate_samples": len(output) - sum(len(values) for values in unique_pairs.values()),
        "selected_pairs": len(selected),
        "selected_contests": len({row["contest_number"] for row in selected}),
        "missing_rankings": missing_rankings,
        "short_contests": short_contests,
        "attempts_by_contest": attempts_by_contest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/newyorker_caption_ranking"))
    parser.add_argument("--context-dir", type=Path, default=Path("data/processed/newyorker_compact_sft_v2"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/newyorker_published_dpo_pairs")
    )
    parser.add_argument("--pairs-per-contest", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=200000)
    parser.add_argument("--selected-per-contest", type=int, default=16)
    parser.add_argument("--max-relative-length-difference", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    if args.pairs_per_contest < 1 or args.max_attempts < 1 or args.selected_per_contest < 1:
        raise ValueError("Pair and attempt limits must be positive")
    if not 0 <= args.max_relative_length_difference <= 1:
        raise ValueError("Relative length difference must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    reports = []
    context_names = {
        "train": "caption_train.jsonl",
        "validation": "caption_validation.jsonl",
        "test": "caption_test.jsonl",
    }
    for split, filename in context_names.items():
        reports.append(
            build_split(
                split=split,
                context_path=args.context_dir / filename,
                ranking_dir=args.raw_dir / "ranking" / "source",
                output_path=args.output_dir / f"{split}.jsonl",
                pairs_per_contest=args.pairs_per_contest,
                max_attempts=args.max_attempts,
                selected_per_contest=args.selected_per_contest,
                max_relative_length_difference=args.max_relative_length_difference,
                rng=rng,
            )
        )

    split_contests = {
        report["split"]: {
            int(row["contest_number"])
            for row in read_jsonl(Path(report["output_file"]))
        }
        for report in reports
    }
    overlaps = {
        "train_validation": sorted(split_contests["train"] & split_contests["validation"]),
        "train_test": sorted(split_contests["train"] & split_contests["test"]),
        "validation_test": sorted(split_contests["validation"] & split_contests["test"]),
    }
    manifest = {
        "source": PAPER,
        "source_url": PAPER_URL,
        "source_dataset": DATASET,
        "license": LICENSE,
        "seed": args.seed,
        "pairs_per_contest": args.pairs_per_contest,
        "max_attempts_per_contest": args.max_attempts,
        "selected_per_contest": args.selected_per_contest,
        "selected_max_relative_length_difference": args.max_relative_length_difference,
        "published_rule": "chosen index from top half; rejected lower-ranked; mean gap > 3*combined precision",
        "local_adaptation": "The published caption preference is paired with this project's image + compact-plan SFT context.",
        "pair_type_limitation": "These are score-derived H2 pairs; they do not supply H1/H3/H4 labels.",
        "commercial_use": "Not permitted by the dataset license.",
        "splits": reports,
        "contest_overlap": overlaps,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(overlaps.values()):
        raise RuntimeError(f"Contest leakage detected: {overlaps}")
    print(json.dumps({"output_dir": str(args.output_dir), "splits": reports, "overlap": overlaps}, indent=2))


if __name__ == "__main__":
    main()
