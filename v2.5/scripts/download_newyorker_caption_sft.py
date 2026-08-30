#!/usr/bin/env python
"""Download and clean the research-only New Yorker caption ranking data.

The source is pinned to a Hugging Face commit.  It produces a separate SFT
dataset because its CC BY-NC 4.0 / academic-only terms must not be silently
mixed with other data.  For every cartoon, the highest ranked 3% of valid
human captions are retained; scores are not comparable across contests.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import write_jsonl

REPO_ID = "yguooo/newyorker_caption_ranking"
REVISION = "1cd70477b6a99a473690a25a2fed359f75184c64"
LICENSE = "CC BY-NC 4.0; dataset card restricts direct use to academic research and prohibits commercial training/products."
DEFAULT_PROMPT = "Generate one short, natural, image-specific humorous caption for this image. Do not explain."
SPACE_RE = re.compile(r"\s+")
URL_ONLY_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)


def normalise_caption(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip()


def valid_caption(value: str, max_caption_chars: int) -> bool:
    return 5 <= len(value) <= max_caption_chars and not URL_ONLY_RE.fullmatch(value)


def download_source(download_dir: Path) -> Path:
    return Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            local_dir=download_dir,
            allow_patterns=[
                "README.md",
                "gpt4o_description/*.jsonl",
                "ranking/*.jsonl",
                "ranking/source/*.csv",
                "cartoons/source/*.jpg",
            ],
        )
    )


def description_splits(source_dir: Path) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for split in ("train", "validation", "test"):
        path = source_dir / "gpt4o_description" / f"{split}.jsonl"
        contest_ids: set[int] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                contest_ids.add(int(json.loads(line)["contest_number"]))
        result[split] = contest_ids
    return result


def all_contest_splits(source_dir: Path, seed: int) -> dict[str, set[int]]:
    images = {int(path.stem) for path in (source_dir / "cartoons" / "source").glob("*.jpg")}
    rankings = {int(path.stem) for path in (source_dir / "ranking" / "source").glob("*.csv")}
    contests = sorted(images & rankings)
    if len(contests) < 3:
        raise ValueError("Need at least three contests to create image-disjoint splits.")
    random.Random(seed).shuffle(contests)
    train_end = int(len(contests) * 0.90)
    val_end = train_end + int(len(contests) * 0.05)
    return {
        "train": set(contests[:train_end]),
        "val": set(contests[train_end:val_end]),
        "test": set(contests[val_end:]),
    }


def readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def make_row(image: Path, contest: int, caption: str, source: dict[str, str], split: str, prompt: str) -> dict[str, Any]:
    return {
        "image": str(image),
        "image_id": f"nycc_{contest}",
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": str(image)}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ],
        "meta": {
            "source": "New Yorker Caption Ranking Dataset",
            "source_repo": REPO_ID,
            "source_revision": REVISION,
            "source_split": split,
            "contest_number": contest,
            "rank": int(source["rank"]),
            "score": float(source["mean"]),
            "votes": int(source["votes"]),
            "funny_votes": int(source["funny"]),
            "license": "CC-BY-NC-4.0",
            "selection": "per_cartoon_top_fraction_by_source_rank",
        },
    }


def build_sft(
    source_dir: Path,
    output_dir: Path,
    top_fraction: float,
    max_caption_chars: int,
    prompt: str,
    all_contests: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    splits = all_contest_splits(source_dir, seed) if all_contests else description_splits(source_dir)
    output: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    balanced: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    counters: Counter[str] = Counter()
    for split, contests in splits.items():
        for contest in sorted(contests):
            image = source_dir / "cartoons" / "source" / f"{contest}.jpg"
            ranking = source_dir / "ranking" / "source" / f"{contest}.csv"
            if not image.exists() or not readable_image(image):
                counters["dropped_missing_or_unreadable_image"] += 1
                continue
            if not ranking.exists():
                counters["dropped_missing_ranking_file"] += 1
                continue
            valid: list[tuple[int, str, dict[str, str]]] = []
            seen: set[str] = set()
            with ranking.open(encoding="utf-8", newline="") as handle:
                for source in csv.DictReader(handle):
                    caption = normalise_caption(source.get("caption") or "")
                    if not valid_caption(caption, max_caption_chars):
                        counters["dropped_invalid_caption"] += 1
                        continue
                    try:
                        rank = int(source["rank"])
                    except (KeyError, TypeError, ValueError):
                        # The release leaves rank empty for its long tail of
                        # unranked submissions.  They have no reliable place
                        # in a top-3%-by-rank SFT set.
                        counters["dropped_unranked_submission"] += 1
                        continue
                    try:
                        float(source["mean"])
                        int(source["votes"])
                        int(source["funny"])
                    except (KeyError, TypeError, ValueError):
                        counters["dropped_invalid_rating"] += 1
                        continue
                    key = caption.lower()
                    if key in seen:
                        counters["dropped_duplicate_caption_within_cartoon"] += 1
                        continue
                    seen.add(key)
                    valid.append((rank, caption, source))
            valid.sort(key=lambda item: (item[0], item[1].lower()))
            keep = math.floor(len(valid) * top_fraction)
            if keep == 0 and valid:
                keep = 1
            selected = valid[:keep]
            rows = [make_row(image, contest, caption, source, split, prompt) for _, caption, source in selected]
            output[split].extend(rows)
            if rows:
                balanced[split].append(rows[0])
            counters["source_valid_pairs"] += len(valid)
            counters["selected_pairs"] += len(rows)
            counters["selected_cartoons"] += int(bool(rows))

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        write_jsonl(output_dir / f"sft_{split}.jsonl", rows)
        write_jsonl(output_dir / f"sft_{split}_one_per_image.jsonl", balanced[split])
    manifest = {
        "dataset": "New Yorker Caption Ranking Dataset",
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "license_and_use_restriction": LICENSE,
        "selection": "top fraction separately within each cartoon by the source rank column (rank 0 is best)",
        "split_strategy": (
            f"deterministic image-disjoint 90/5/5 split over all downloaded cartoons (seed={seed})"
            if all_contests
            else "official gpt4o_description train/validation/test contest assignment from the source release"
        ),
        "top_fraction": top_fraction,
        "max_caption_chars": max_caption_chars,
        "prompt": prompt,
        "rows_by_split": {split: len(rows) for split, rows in output.items()},
        "one_per_image_rows_by_split": {split: len(rows) for split, rows in balanced.items()},
        "filter_counts": dict(sorted(counters.items())),
        "training_note": "Use the full SFT files with image-balanced sampling, or the one_per_image files for a conservative caption SFT baseline.",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = ArgumentParser(description="Download and turn high-rated NYCC captions into research-only SFT JSONL.")
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw/newyorker_caption_ranking"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/newyorker_top3pct_sft"))
    parser.add_argument("--top-fraction", type=float, default=0.03)
    parser.add_argument("--max-caption-chars", type=int, default=220)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="UTF-8 file containing the exact task prompt; takes precedence over --prompt.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--all-contests",
        action="store_true",
        help="Use every downloaded cartoon and create a deterministic image-disjoint 90/5/5 split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source_dir = args.download_dir if args.skip_download else download_source(args.download_dir)
    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Prompt file is empty: {args.prompt_file}")
    manifest = build_sft(
        source_dir,
        args.output_dir,
        args.top_fraction,
        args.max_caption_chars,
        prompt,
        all_contests=args.all_contests,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
