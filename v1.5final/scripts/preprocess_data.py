#!/usr/bin/env python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess_hic import DEFAULT_PROMPT, PreprocessConfig, preprocess_hic_dataset


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


def load_config(path: Path) -> PreprocessConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = raw["data"]
    filtering = raw.get("filtering", {})
    split = raw.get("split", {})
    output = raw.get("output", {})

    return PreprocessConfig(
        image_csv=Path(data["image_csv"]),
        caption_csv=Path(data["caption_csv"]),
        image_base_dir=_path(data.get("image_base_dir")),
        output_dir=Path(output.get("output_dir", "data/processed")),
        image_id_col=data["image_id_col"],
        image_path_col=data.get("image_path_col"),
        image_url_col=data.get("image_url_col"),
        caption_image_id_col=data["caption_image_id_col"],
        caption_col=data["caption_col"],
        score_col=data["score_col"],
        prompt=data.get("prompt", DEFAULT_PROMPT),
        min_caption_chars=filtering.get("min_caption_chars", 5),
        max_caption_chars=filtering.get("max_caption_chars", 220),
        min_score=filtering.get("min_score"),
        rank_percentile_threshold=filtering.get("rank_percentile_threshold", 0.85),
        max_captions_per_image=filtering.get("max_captions_per_image", 5),
        train_ratio=split.get("train_ratio", 0.9),
        val_ratio=split.get("val_ratio", 0.05),
        test_ratio=split.get("test_ratio", 0.05),
        seed=split.get("seed", 42),
        require_existing_image=filtering.get("require_existing_image", False),
    )


def main() -> None:
    parser = ArgumentParser(description="Preprocess OxfordTVG-HIC data into V1.5 SFT JSONL files.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_preprocess.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    counts = preprocess_hic_dataset(config)
    print(f"Saved V1.5 SFT data to {config.output_dir}: {counts}")


if __name__ == "__main__":
    main()
