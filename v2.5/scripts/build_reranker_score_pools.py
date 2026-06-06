#!/usr/bin/env python
from __future__ import annotations

import json
import math
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPACE_RE = r"\s+"


def clean_caption(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def quantiles(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def row_to_item(row: pd.Series, pool: str) -> dict[str, Any]:
    return {
        "image": str(row["image"]),
        "image_id": str(row["image_id"]),
        "caption": str(row["caption"]),
        "score": float(row["score"]),
        "rank_pct": float(row["rank_pct"]),
        "low_rank_pct": float(row["low_rank_pct"]),
        "image_caption_count": int(row["image_caption_count"]),
        "image_score_min": float(row["image_score_min"]),
        "image_score_max": float(row["image_score_max"]),
        "image_score_spread": float(row["image_score_spread"]),
        "pool": pool,
    }


def cap_pool(df: pd.DataFrame, max_per_image: int, sort_ascending: bool) -> pd.DataFrame:
    if max_per_image <= 0:
        return df.iloc[0:0].copy()
    return (
        df.sort_values(["image_id", "score", "caption"], ascending=[True, sort_ascending, True])
        .groupby("image_id", sort=False)
        .head(max_per_image)
        .reset_index(drop=True)
    )


def make_pairs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    pair_type: str,
    loss_weight: float,
    max_pairs_per_image: int,
    min_score_gap: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    right_by_image = {image_id: group for image_id, group in right.groupby("image_id")}
    for image_id, left_group in left.groupby("image_id"):
        right_group = right_by_image.get(image_id)
        if right_group is None or right_group.empty:
            continue
        candidates: list[tuple[pd.Series, pd.Series]] = []
        for _, pos in left_group.iterrows():
            for _, neg in right_group.iterrows():
                if float(pos["score"]) - float(neg["score"]) >= min_score_gap:
                    candidates.append((pos, neg))
        rng.shuffle(candidates)
        for pos, neg in candidates[:max_pairs_per_image]:
            pairs.append(
                {
                    "image": str(pos["image"]),
                    "image_id": str(image_id),
                    "positive": str(pos["caption"]),
                    "negative": str(neg["caption"]),
                    "pos_score": float(pos["score"]),
                    "neg_score": float(neg["score"]),
                    "pos_rank_pct": float(pos["rank_pct"]),
                    "neg_rank_pct": float(neg["rank_pct"]),
                    "pos_low_rank_pct": float(pos["low_rank_pct"]),
                    "neg_low_rank_pct": float(neg["low_rank_pct"]),
                    "score_gap": float(pos["score"] - neg["score"]),
                    "pair_type": pair_type,
                    "loss_weight": float(loss_weight),
                    "pos_source": "oxford_score_pool",
                    "neg_source": "oxford_score_pool",
                }
            )
    return pairs


def load_and_label(args: Any) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    images = pd.read_csv(args.image_csv, low_memory=False)
    captions = pd.read_csv(args.caption_csv, low_memory=False)

    required_image_cols = {args.image_id_col, args.image_path_col}
    required_caption_cols = {args.caption_image_id_col, args.caption_col, args.score_col}
    missing_image = required_image_cols - set(images.columns)
    missing_caption = required_caption_cols - set(captions.columns)
    if missing_image:
        raise ValueError(f"image CSV missing columns: {sorted(missing_image)}")
    if missing_caption:
        raise ValueError(f"caption CSV missing columns: {sorted(missing_caption)}")

    images = images[[args.image_id_col, args.image_path_col]].copy()
    images[args.image_id_col] = images[args.image_id_col].astype(str)
    images[args.image_path_col] = images[args.image_path_col].astype(str)
    images = images.rename(columns={args.image_id_col: "image_id", args.image_path_col: "image"})

    captions = captions[[args.caption_image_id_col, args.caption_col, args.score_col]].copy()
    captions = captions.rename(
        columns={
            args.caption_image_id_col: "image_id",
            args.caption_col: "caption",
            args.score_col: "score",
        }
    )
    captions["image_id"] = captions["image_id"].astype(str)
    captions["caption"] = captions["caption"].apply(clean_caption)
    captions["score"] = pd.to_numeric(captions["score"], errors="coerce")
    before = len(captions)
    captions = captions.dropna(subset=["score"])
    captions = captions[captions["caption"].str.len().between(args.min_caption_chars, args.max_caption_chars)]
    captions = captions[~captions["caption"].str.match(r"^https?://\S+$", case=False, na=False)]
    captions["_caption_norm"] = captions["caption"].str.lower()
    captions = captions.drop_duplicates(subset=["image_id", "_caption_norm"])

    captions["rank_pct"] = captions.groupby("image_id")["score"].rank(pct=True, method="max")
    captions["low_rank_pct"] = captions.groupby("image_id")["score"].rank(pct=True, method="min")
    captions["image_caption_count"] = captions.groupby("image_id")["caption"].transform("size")
    captions["image_score_min"] = captions.groupby("image_id")["score"].transform("min")
    captions["image_score_max"] = captions.groupby("image_id")["score"].transform("max")
    captions["image_score_spread"] = captions["image_score_max"] - captions["image_score_min"]

    merged = captions.merge(images, on="image_id", how="inner")
    if args.require_existing_image:
        merged = merged[merged["image"].apply(lambda value: Path(value).exists())]
    merged = merged[merged["image_caption_count"] >= args.min_captions_per_image].copy()
    scored_images = merged[merged["image_score_spread"] >= args.min_image_score_spread].copy()

    strong_pos = scored_images[
        (scored_images["rank_pct"] >= args.strong_pos_rank_min)
        & (scored_images["score"] >= args.strong_pos_score_min)
    ]
    weak_pos = scored_images[
        (scored_images["rank_pct"] >= args.weak_pos_rank_min)
        & (scored_images["rank_pct"] < args.weak_pos_rank_max)
        & (scored_images["score"] >= args.weak_pos_score_min)
    ]
    strong_neg = scored_images[
        (scored_images["low_rank_pct"] <= args.strong_neg_low_rank_max)
        & (scored_images["score"] <= args.strong_neg_score_max)
    ]
    weak_neg = scored_images[
        (scored_images["rank_pct"] >= args.weak_neg_rank_min)
        & (scored_images["rank_pct"] <= args.weak_neg_rank_max)
        & (scored_images["score"] <= args.weak_neg_score_max)
    ]

    pools = {
        "strong_positive": cap_pool(strong_pos, args.max_strong_positive_per_image, sort_ascending=False),
        "weak_positive": cap_pool(weak_pos, args.max_weak_positive_per_image, sort_ascending=False),
        "strong_negative": cap_pool(strong_neg, args.max_strong_negative_per_image, sort_ascending=True),
        "weak_negative": cap_pool(weak_neg, args.max_weak_negative_per_image, sort_ascending=True),
    }
    merged.attrs["raw_caption_rows"] = before
    merged.attrs["valid_caption_rows"] = len(captions)
    merged.attrs["merged_rows"] = len(merged)
    merged.attrs["scored_rows"] = len(scored_images)
    return merged, pools


def build_summary(args: Any, all_rows: pd.DataFrame, pools: dict[str, pd.DataFrame], pairs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {
        "caption_csv": str(args.caption_csv),
        "image_csv": str(args.image_csv),
        "output_dir": str(args.output_dir),
        "thresholds": {
            "min_caption_chars": args.min_caption_chars,
            "max_caption_chars": args.max_caption_chars,
            "min_captions_per_image": args.min_captions_per_image,
            "min_image_score_spread": args.min_image_score_spread,
            "strong_pos_rank_min": args.strong_pos_rank_min,
            "strong_pos_score_min": args.strong_pos_score_min,
            "weak_pos_rank_min": args.weak_pos_rank_min,
            "weak_pos_rank_max": args.weak_pos_rank_max,
            "weak_pos_score_min": args.weak_pos_score_min,
            "strong_neg_low_rank_max": args.strong_neg_low_rank_max,
            "strong_neg_score_max": args.strong_neg_score_max,
            "weak_neg_rank_min": args.weak_neg_rank_min,
            "weak_neg_rank_max": args.weak_neg_rank_max,
            "weak_neg_score_max": args.weak_neg_score_max,
            "strong_pair_min_score_gap": args.strong_pair_min_score_gap,
            "weak_pair_min_score_gap": args.weak_pair_min_score_gap,
        },
        "counts": {
            "raw_caption_rows": int(all_rows.attrs.get("raw_caption_rows", 0)),
            "valid_caption_rows": int(all_rows.attrs.get("valid_caption_rows", 0)),
            "merged_rows": int(all_rows.attrs.get("merged_rows", len(all_rows))),
            "scored_rows_after_spread_filter": int(all_rows.attrs.get("scored_rows", 0)),
            "images": int(all_rows["image_id"].nunique()) if not all_rows.empty else 0,
        },
        "score_quantiles_all_valid": quantiles(all_rows["score"]),
        "pools": {},
        "pairs": {},
    }
    for name, pool in pools.items():
        summary["pools"][name] = {
            "rows": int(len(pool)),
            "images": int(pool["image_id"].nunique()) if not pool.empty else 0,
            "score_quantiles": quantiles(pool["score"]),
            "rank_pct_quantiles": quantiles(pool["rank_pct"]),
        }
    for name, pair_rows in pairs.items():
        summary["pairs"][name] = {
            "rows": len(pair_rows),
            "images": len({row["image_id"] for row in pair_rows}),
            "score_gap_quantiles": quantiles(pd.Series([row["score_gap"] for row in pair_rows])),
            "loss_weight": pair_rows[0]["loss_weight"] if pair_rows else None,
        }
    return summary


def main() -> None:
    parser = ArgumentParser(description="Build adjustable Oxford-HIC score pools for humor reranker training.")
    parser.add_argument("--caption-csv", type=Path, default=Path("/home/zhang.jianqiao/datasets/hic-data/oxford_hic_data.csv"))
    parser.add_argument("--image-csv", type=Path, default=Path("/home/zhang.jianqiao/datasets/hic-data/oxford_hic_image_info.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/reranker_score_pools"))
    parser.add_argument("--image-id-col", type=str, default="image_id")
    parser.add_argument("--image-path-col", type=str, default="image_path")
    parser.add_argument("--caption-image-id-col", type=str, default="image_id")
    parser.add_argument("--caption-col", type=str, default="caption")
    parser.add_argument("--score-col", type=str, default="funny_score")
    parser.add_argument("--min-caption-chars", type=int, default=5)
    parser.add_argument("--max-caption-chars", type=int, default=220)
    parser.add_argument("--min-captions-per-image", type=int, default=5)
    parser.add_argument("--min-image-score-spread", type=float, default=1.0)
    parser.add_argument("--require-existing-image", action="store_true")

    parser.add_argument("--strong-pos-rank-min", type=float, default=0.90)
    parser.add_argument("--strong-pos-score-min", type=float, default=2.0)
    parser.add_argument("--weak-pos-rank-min", type=float, default=0.70)
    parser.add_argument("--weak-pos-rank-max", type=float, default=0.90)
    parser.add_argument("--weak-pos-score-min", type=float, default=1.0)
    parser.add_argument("--strong-neg-low-rank-max", type=float, default=0.20)
    parser.add_argument("--strong-neg-score-max", type=float, default=0.0)
    parser.add_argument("--weak-neg-rank-min", type=float, default=0.20)
    parser.add_argument("--weak-neg-rank-max", type=float, default=0.55)
    parser.add_argument("--weak-neg-score-max", type=float, default=2.0)

    parser.add_argument("--max-strong-positive-per-image", type=int, default=5)
    parser.add_argument("--max-weak-positive-per-image", type=int, default=5)
    parser.add_argument("--max-strong-negative-per-image", type=int, default=5)
    parser.add_argument("--max-weak-negative-per-image", type=int, default=5)
    parser.add_argument("--max-strong-pairs-per-image", type=int, default=5)
    parser.add_argument("--max-weak-pairs-per-image", type=int, default=5)
    parser.add_argument("--strong-pair-min-score-gap", type=float, default=2.0)
    parser.add_argument("--weak-pair-min-score-gap", type=float, default=1.0)
    parser.add_argument("--strong-loss-weight", type=float, default=1.0)
    parser.add_argument("--weak-loss-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    all_rows, pools = load_and_label(args)
    strong_pairs = make_pairs(
        pools["strong_positive"],
        pools["strong_negative"],
        pair_type="strong_positive_vs_strong_negative",
        loss_weight=args.strong_loss_weight,
        max_pairs_per_image=args.max_strong_pairs_per_image,
        min_score_gap=args.strong_pair_min_score_gap,
        seed=args.seed,
    )
    weak_pairs = make_pairs(
        pools["weak_positive"],
        pools["weak_negative"],
        pair_type="weak_positive_vs_weak_negative",
        loss_weight=args.weak_loss_weight,
        max_pairs_per_image=args.max_weak_pairs_per_image,
        min_score_gap=args.weak_pair_min_score_gap,
        seed=args.seed + 17,
    )
    pairs = {"strong_pairs": strong_pairs, "weak_pairs": weak_pairs}
    summary = build_summary(args, all_rows, pools, pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for pool_name, pool in pools.items():
            rows = [row_to_item(row, pool_name) for _, row in pool.iterrows()]
            write_jsonl(args.output_dir / f"{pool_name}.jsonl", rows)
        write_jsonl(args.output_dir / "strong_pairs.jsonl", strong_pairs)
        write_jsonl(args.output_dir / "weak_pairs.jsonl", weak_pairs)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[reranker-pools] saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
