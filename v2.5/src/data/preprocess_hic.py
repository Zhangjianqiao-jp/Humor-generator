from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.split import split_image_ids
from src.utils.io import write_jsonl

DEFAULT_PROMPT = "Generate a short humorous caption for this image."
URL_ONLY_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessConfig:
    image_csv: Path
    caption_csv: Path
    image_base_dir: Path | None
    output_dir: Path
    image_id_col: str
    image_path_col: str | None
    image_url_col: str | None
    caption_image_id_col: str
    caption_col: str
    score_col: str
    prompt: str = DEFAULT_PROMPT
    min_caption_chars: int = 5
    max_caption_chars: int = 220
    min_score: float | None = None
    rank_percentile_threshold: float = 0.85
    max_captions_per_image: int = 5
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    require_existing_image: bool = False


def _normalize_caption(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return SPACE_RE.sub(" ", text).strip()


def _is_valid_caption(text: str, min_chars: int, max_chars: int) -> bool:
    if len(text) < min_chars or len(text) > max_chars:
        return False
    if URL_ONLY_RE.match(text):
        return False
    return True


def _resolve_image_column(config: PreprocessConfig) -> str:
    image_col = config.image_path_col or config.image_url_col
    if image_col is None:
        raise ValueError("Set either image_path_col or image_url_col.")
    return image_col


def preprocess_hic_dataset(config: PreprocessConfig) -> dict[str, int]:
    images = pd.read_csv(config.image_csv, low_memory=False)
    captions = pd.read_csv(config.caption_csv, low_memory=False)
    image_col = _resolve_image_column(config)

    required_image_cols = {config.image_id_col, image_col}
    required_caption_cols = {config.caption_image_id_col, config.caption_col, config.score_col}
    missing_image = required_image_cols - set(images.columns)
    missing_caption = required_caption_cols - set(captions.columns)
    if missing_image:
        raise ValueError(f"Image CSV missing columns: {sorted(missing_image)}")
    if missing_caption:
        raise ValueError(f"Caption CSV missing columns: {sorted(missing_caption)}")

    captions = captions[[config.caption_image_id_col, config.caption_col, config.score_col]].copy()
    captions[config.caption_image_id_col] = captions[config.caption_image_id_col].astype(str)
    captions["_caption"] = captions[config.caption_col].apply(_normalize_caption)
    captions = captions[
        captions["_caption"].apply(
            lambda text: _is_valid_caption(text, config.min_caption_chars, config.max_caption_chars)
        )
    ]

    captions[config.score_col] = pd.to_numeric(captions[config.score_col], errors="coerce")
    captions = captions.dropna(subset=[config.score_col])
    if config.min_score is not None:
        captions = captions[captions[config.score_col] >= config.min_score]

    captions["_cap_norm"] = captions["_caption"].str.lower()
    captions = captions.drop_duplicates(subset=[config.caption_image_id_col, "_cap_norm"])
    captions["_rank_pct"] = captions.groupby(config.caption_image_id_col)[config.score_col].rank(
        pct=True, method="max"
    )
    captions = captions[captions["_rank_pct"] >= config.rank_percentile_threshold]
    captions = (
        captions.sort_values([config.caption_image_id_col, config.score_col], ascending=[True, False])
        .groupby(config.caption_image_id_col)
        .head(config.max_captions_per_image)
    )

    image_subset = images[[config.image_id_col, image_col]].copy()
    image_subset[config.image_id_col] = image_subset[config.image_id_col].astype(str)
    image_subset[image_col] = image_subset[image_col].astype(str)
    if config.image_base_dir and config.image_path_col:
        image_subset[image_col] = image_subset[image_col].apply(
            lambda value: str(Path(value)) if Path(value).is_absolute() else str(config.image_base_dir / value)
        )
    merged = captions.merge(
        image_subset,
        left_on=config.caption_image_id_col,
        right_on=config.image_id_col,
        how="inner",
    )

    merged["image_id"] = merged[config.image_id_col].astype(str)
    merged["image"] = merged[image_col].astype(str)
    if config.require_existing_image and config.image_path_col:
        merged = merged[merged["image"].apply(lambda p: Path(p).exists())]

    splits = split_image_ids(
        merged["image_id"].tolist(),
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.seed,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    all_rows: list[dict[str, Any]] = []
    for split_name, split_ids in splits.items():
        rows = []
        subset = merged[merged["image_id"].isin(split_ids)]
        for _, row in subset.iterrows():
            item = {
                "image": row["image"],
                "image_id": row["image_id"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": row["image"]},
                            {"type": "text", "text": config.prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": row["_caption"]}],
                    },
                ],
                "meta": {
                    "score": float(row[config.score_col]),
                    "rank_pct": float(row["_rank_pct"]),
                    "source": "OxfordTVG-HIC",
                    "version": "v1.5",
                },
            }
            rows.append(item)
        write_jsonl(config.output_dir / f"sft_{split_name}.jsonl", rows)
        counts[split_name] = len(rows)
        all_rows.extend(rows)

    write_jsonl(config.output_dir / "sft_sample_100.jsonl", all_rows[:100])
    return counts
