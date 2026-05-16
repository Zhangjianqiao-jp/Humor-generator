#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

from src.data.build_sft_dataset import build_sft_dataset


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--image-csv", type=Path, required=True)
    p.add_argument("--caption-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--image-id-col", required=True)
    p.add_argument("--image-path-col")
    p.add_argument("--image-url-col")
    p.add_argument("--caption-image-id-col", required=True)
    p.add_argument("--caption-col", required=True)
    p.add_argument("--score-col", required=True)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--max-per-image", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_sft_dataset(args.image_csv, args.caption_csv, args.output_dir, args.image_id_col, args.image_path_col, args.image_url_col,
                      args.caption_image_id_col, args.caption_col, args.score_col, args.threshold, args.max_per_image, seed=args.seed)
