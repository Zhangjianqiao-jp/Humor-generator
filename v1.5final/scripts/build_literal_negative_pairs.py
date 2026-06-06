#!/usr/bin/env python
from __future__ import annotations

import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_positives(path: Path, max_per_image: int, min_pos_score: float) -> dict[str, list[dict[str, Any]]]:
    positives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        image_id = str(row.get("image_id") or Path(str(row.get("image", ""))).stem)
        caption = clean(row.get("caption") or "")
        if not caption:
            continue
        score = float(row.get("score") or 0.0)
        if score < min_pos_score:
            continue
        if len(positives[image_id]) < max_per_image:
            positives[image_id].append(row)
    return positives


def build_pairs(
    literal_jsonl: Path,
    positive_jsonl: Path,
    output_jsonl: Path,
    max_positives_per_image: int,
    max_literals_per_image: int,
    max_pairs_per_image: int,
    min_pos_score: float,
    neg_score_offset: float,
    loss_weight: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    positives = load_positives(positive_jsonl, max_per_image=max_positives_per_image, min_pos_score=min_pos_score)
    pairs: list[dict[str, Any]] = []
    skipped_no_positive = 0
    skipped_no_literal = 0
    for row in read_jsonl(literal_jsonl):
        image_id = str(row.get("image_id") or Path(str(row.get("image", ""))).stem)
        pos_rows = positives.get(image_id) or []
        if not pos_rows:
            skipped_no_positive += 1
            continue
        literals = [clean(item) for item in (row.get("literal_captions") or []) if clean(item)]
        literals = literals[:max_literals_per_image]
        if not literals:
            skipped_no_literal += 1
            continue
        candidates = []
        for pos in pos_rows:
            pos_caption = clean(pos.get("caption") or "")
            pos_score = float(pos.get("score") or 0.0)
            # This score is metadata only for current pairwise training. Keep the gap modest
            # because literal captions are image-grounded negatives, not totally bad captions.
            neg_score = max(0.0, pos_score - neg_score_offset)
            for literal in literals:
                candidates.append(
                    {
                        "image": str(row.get("image") or pos.get("image")),
                        "image_id": image_id,
                        "positive": pos_caption,
                        "negative": literal,
                        "pos_score": pos_score,
                        "neg_score": neg_score,
                        "pos_rank_pct": pos.get("rank_pct"),
                        "neg_rank_pct": None,
                        "score_gap": pos_score - neg_score,
                        "pair_type": "strong_positive_vs_literal_caption",
                        "negative_type": "literal_caption",
                        "loss_weight": loss_weight,
                        "pos_source": "oxford_strong_positive",
                        "neg_source": "qwen_literal_caption",
                    }
                )
        rng.shuffle(candidates)
        pairs.extend(candidates[:max_pairs_per_image])
    rng.shuffle(pairs)
    write_jsonl(output_jsonl, pairs)
    summary = {
        "literal_jsonl": str(literal_jsonl),
        "positive_jsonl": str(positive_jsonl),
        "output_jsonl": str(output_jsonl),
        "num_pairs": len(pairs),
        "num_images": len({row["image_id"] for row in pairs}),
        "max_positives_per_image": max_positives_per_image,
        "max_literals_per_image": max_literals_per_image,
        "max_pairs_per_image": max_pairs_per_image,
        "min_pos_score": min_pos_score,
        "neg_score_offset": neg_score_offset,
        "loss_weight": loss_weight,
        "skipped_no_positive": skipped_no_positive,
        "skipped_no_literal": skipped_no_literal,
    }
    summary_path = output_jsonl.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[literal-pairs] saved {len(pairs)} pairs to {output_jsonl}")
    return summary


def main() -> None:
    parser = ArgumentParser(description="Build reranker pairwise training data with Qwen literal caption negatives.")
    parser.add_argument("--literal-jsonl", type=Path, default=Path("data/processed/reranker_hard_negatives/literal_captions_qwen.jsonl"))
    parser.add_argument("--positive-jsonl", type=Path, default=Path("data/processed/reranker_score_pools_strict/strong_positive.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/processed/reranker_hard_negatives/literal_pairs.jsonl"))
    parser.add_argument("--max-positives-per-image", type=int, default=2)
    parser.add_argument("--max-literals-per-image", type=int, default=5)
    parser.add_argument("--max-pairs-per-image", type=int, default=8)
    parser.add_argument("--min-pos-score", type=float, default=4.0)
    parser.add_argument("--neg-score-offset", type=float, default=1.0)
    parser.add_argument("--loss-weight", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_pairs(
        literal_jsonl=args.literal_jsonl,
        positive_jsonl=args.positive_jsonl,
        output_jsonl=args.output_jsonl,
        max_positives_per_image=args.max_positives_per_image,
        max_literals_per_image=args.max_literals_per_image,
        max_pairs_per_image=args.max_pairs_per_image,
        min_pos_score=args.min_pos_score,
        neg_score_offset=args.neg_score_offset,
        loss_weight=args.loss_weight,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
