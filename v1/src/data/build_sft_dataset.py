from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.split import split_image_ids
from src.utils.io import write_jsonl
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)
PROMPT = "Generate a short humorous caption for this image."
URL_ONLY_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)


def _is_valid_caption(text: Any, min_len: int = 5, max_len: int = 220) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t or len(t) < min_len or len(t) > max_len:
        return False
    if URL_ONLY_RE.match(t):
        return False
    return True


def build_sft_dataset(image_csv: Path, caption_csv: Path, output_dir: Path, image_id_col: str, image_path_col: str | None, image_url_col: str | None,
                      caption_image_id_col: str, caption_col: str, score_col: str, threshold: float = 0.85, max_per_image: int = 5,
                      train_ratio: float = 0.9, val_ratio: float = 0.05, test_ratio: float = 0.05, seed: int = 42) -> dict[str, int]:
    images = pd.read_csv(image_csv)
    caps = pd.read_csv(caption_csv)

    path_col = image_path_col or image_url_col
    if path_col is None:
        raise ValueError("Provide image_path_col or image_url_col")

    caps = caps[[caption_image_id_col, caption_col, score_col]].copy()
    caps[score_col] = pd.to_numeric(caps[score_col], errors="coerce")
    invalid_score_n = int(caps[score_col].isna().sum())
    if invalid_score_n:
        LOGGER.warning("Dropping %d rows with non-numeric score values in column '%s'.", invalid_score_n,
                       score_col)
    caps = caps.dropna(subset=[score_col])
    caps = caps[caps[caption_col].apply(_is_valid_caption)]
    caps["_cap_norm"] = caps[caption_col].str.strip().str.lower()
    caps = caps.drop_duplicates(subset=[caption_image_id_col, "_cap_norm"])
    caps["rank_pct"] = caps.groupby(caption_image_id_col)[score_col].rank(pct=True, method="max")
    caps = caps[caps["rank_pct"] >= threshold]
    caps = caps.sort_values([caption_image_id_col, score_col], ascending=[True, False]).groupby(
        caption_image_id_col).head(max_per_image)
    merged = caps.merge(images[[image_id_col, path_col]], left_on=caption_image_id_col, right_on=image_id_col, how="inner")
    merged["image_id"] = merged[image_id_col].astype(str)
    merged["image"] = merged[path_col].astype(str)

    id_splits = split_image_ids(merged["image_id"].tolist(), train_ratio, val_ratio, test_ratio, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    all_rows = []
    for split_name, ids in id_splits.items():
        sub = merged[merged["image_id"].isin(ids)]
        rows = []
        for _, r in sub.iterrows():
            row = {
                "image": r["image"],
                "image_id": r["image_id"],
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": r["image"]}, {"type": "text", "text": PROMPT}]},
                    {"role": "assistant", "content": [{"type": "text", "text": str(r[caption_col]).strip()}]},
                ],
                "meta": {"score": float(r[score_col]), "rank_pct": float(r["rank_pct"]), "source": "OxfordTVG-HIC"},
            }
            rows.append(row)
        write_jsonl(output_dir / f"sft_{split_name}.jsonl", rows)
        counts[split_name] = len(rows)
        all_rows.extend(rows)
    write_jsonl(output_dir / "sft_sample_100.jsonl", all_rows[:100])
    LOGGER.info("Saved SFT dataset counts: %s", counts)
    return counts
