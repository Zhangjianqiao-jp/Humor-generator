#!/usr/bin/env python
"""Build an exact global top-score subset from existing SFT JSONL files.

The source split of every example is preserved.  The selection is made over
all valid source rows, so ``--top-fraction 0.03`` means exactly the highest
3% of the locally available corpus rather than 3% independently per split.
"""
from __future__ import annotations

import json
import math
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.sft_dataset import extract_caption, extract_image_path
from src.utils.io import read_jsonl, write_jsonl

SPACE_RE = re.compile(r"\s+")
URL_ONLY_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
LAUGHTER_ONLY = {"(laughter)", "[laughter]", "laughter", "laugh", "laughs"}


def normalise_caption(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip()


def valid_caption(value: str, max_caption_chars: int) -> bool:
    return (
        5 <= len(value) <= max_caption_chars
        and not URL_ONLY_RE.fullmatch(value)
        and value.lower().strip() not in LAUGHTER_ONLY
    )


def score_from_row(row: dict[str, Any]) -> float | None:
    value = row.get("meta", {}).get("score")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def build_top_score_sft(
    inputs: dict[str, Path],
    output_dir: Path,
    top_fraction: float,
    max_caption_chars: int,
    verify_images: bool,
) -> dict[str, Any]:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")

    candidates: list[tuple[float, str, str, int, str, dict[str, Any]]] = []
    counters: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for split, path in inputs.items():
        for source_index, row in enumerate(read_jsonl(path)):
            counters[f"{split}_input_rows"] += 1
            image = extract_image_path(row)
            caption = extract_caption(row)
            score = score_from_row(row)
            if not image:
                counters["dropped_missing_image_field"] += 1
                continue
            if not isinstance(caption, str):
                counters["dropped_missing_caption"] += 1
                continue
            caption = normalise_caption(caption)
            if not valid_caption(caption, max_caption_chars):
                counters["dropped_invalid_caption"] += 1
                continue
            if score is None:
                counters["dropped_invalid_score"] += 1
                continue
            image_path = Path(image)
            image_id = str(row.get("image_id") or image_path.stem)
            dedupe_key = (image_id, caption.lower())
            if dedupe_key in seen:
                counters["dropped_duplicate_image_caption"] += 1
                continue
            seen.add(dedupe_key)
            row = dict(row)
            row["image"] = str(image)
            row["image_id"] = image_id
            row["meta"] = {
                **dict(row.get("meta") or {}),
                "score": score,
                "selection": "global_top_score",
                "source_split": split,
            }
            candidates.append((score, image_id, caption.lower(), source_index, split, row))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    target_count = math.floor(len(candidates) * top_fraction)
    if target_count == 0:
        raise ValueError("The requested top fraction selects zero valid rows.")
    selected: list[tuple[float, str, str, int, str, dict[str, Any]]] = []
    for candidate in candidates:
        if verify_images and not Path(str(candidate[-1]["image"])).exists():
            counters["dropped_missing_image_selected_or_replacement"] += 1
            continue
        selected.append(candidate)
        if len(selected) == target_count:
            break
    if len(selected) != target_count:
        raise ValueError("Not enough readable image-caption rows to satisfy the requested top fraction.")
    cutoff_score = selected[-1][0]
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in inputs}
    for _, _, _, _, split, row in selected:
        by_split[split].append(row)
    for split in by_split:
        by_split[split].sort(key=lambda row: (str(row["image_id"]), -float(row["meta"]["score"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in by_split.items():
        write_jsonl(output_dir / f"sft_{split}.jsonl", rows)
    manifest = {
        "dataset": "OxfordTVG-HIC locally available processed subset",
        "selection": "exact global top fraction by meta.score, deterministic score/image/caption tie break",
        "top_fraction": top_fraction,
        "candidate_count_before_selected_image_check": len(candidates),
        "selected_count": target_count,
        "cutoff_score": cutoff_score,
        "selected_by_source_split": {split: len(rows) for split, rows in by_split.items()},
        "input_paths": {split: str(path) for split, path in inputs.items()},
        "verify_images": verify_images,
        "max_caption_chars": max_caption_chars,
        "filter_counts": dict(sorted(counters.items())),
        "important_scope_note": (
            "The repository contains processed HIC JSONL rather than the original 2.9M-pair release; "
            "this is the top fraction of the locally available valid rows."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = ArgumentParser(description="Create an exact top-score SFT subset from existing JSONL splits.")
    parser.add_argument("--train", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/sft_val.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/sft_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/hic_top3pct_sft"))
    parser.add_argument("--top-fraction", type=float, default=0.03)
    parser.add_argument("--max-caption-chars", type=int, default=220)
    parser.add_argument("--skip-image-verification", action="store_true")
    args = parser.parse_args()
    manifest = build_top_score_sft(
        inputs={"train": args.train, "val": args.val, "test": args.test},
        output_dir=args.output_dir,
        top_fraction=args.top_fraction,
        max_caption_chars=args.max_caption_chars,
        verify_images=not args.skip_image_verification,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
