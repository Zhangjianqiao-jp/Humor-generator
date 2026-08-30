#!/usr/bin/env python
"""Create image-conditioned DPO pairs from New Yorker within-cartoon ranks."""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl

SPACE_RE = re.compile(r"\s+")


def clean_caption(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip()


def get_text(row: dict[str, Any], role: str) -> str:
    for message in row["messages"]:
        if message.get("role") == role:
            for item in message["content"]:
                if item.get("type") == "text":
                    return str(item.get("text") or "")
    return ""


def ranked_captions(path: Path) -> list[tuple[int, str, dict[str, str]]]:
    rows: list[tuple[int, str, dict[str, str]]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            caption = clean_caption(str(row.get("caption") or ""))
            try:
                rank = int(row["rank"])
            except (KeyError, TypeError, ValueError):
                continue
            if len(caption) < 5 or caption.lower() in seen:
                continue
            seen.add(caption.lower())
            rows.append((rank, caption, row))
    return sorted(rows, key=lambda item: (item[0], item[1].lower()))


def build_pairs(compact_dir: Path, raw_dir: Path, output_dir: Path, rejected_percentile: float) -> dict[str, Any]:
    if not 0 < rejected_percentile < 1:
        raise ValueError("rejected_percentile must be in (0, 1).")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        compact_rows = read_jsonl(compact_dir / f"caption_{split}.jsonl")
        choices: dict[int, list[dict[str, Any]]] = {}
        for row in compact_rows:
            choices.setdefault(int(row["meta"]["contest_number"]), []).append(row)
        pairs: list[dict[str, Any]] = []
        for contest, rows in sorted(choices.items()):
            ranked = ranked_captions(raw_dir / "ranking" / "source" / f"{contest}.csv")
            if len(ranked) < 4:
                continue
            rejected_index = max(1, math.floor(len(ranked) * rejected_percentile))
            rejected_rank, rejected, rejected_meta = ranked[rejected_index]
            for row in rows:
                chosen = get_text(row, "assistant")
                chosen_rank = int(row["meta"]["rank"])
                if chosen_rank >= rejected_rank or chosen.lower() == rejected.lower():
                    continue
                prompt = get_text(row, "user")
                pairs.append(
                    {
                        "image": row["image"],
                        "image_id": row["image_id"],
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "meta": {
                            "source": "New Yorker Caption Ranking Dataset",
                            "contest_number": contest,
                            "chosen_rank": chosen_rank,
                            "rejected_rank": rejected_rank,
                            "rank_gap": rejected_rank - chosen_rank,
                            "rejected_votes": int(rejected_meta.get("votes") or 0),
                            "compact": row["meta"]["compact"],
                            "pair_type": "same_image_rank_gap",
                        },
                    }
                )
        write_jsonl(output_dir / f"dpo_{split}.jsonl", pairs)
        counts[split] = len(pairs)
    manifest = {
        "source_compact_dir": str(compact_dir),
        "source_ranking_dir": str(raw_dir / "ranking" / "source"),
        "rejected_percentile": rejected_percentile,
        "pair_selection": "chosen=top-3%-rank caption; rejected=caption at fixed lower within-cartoon rank percentile",
        "pairs_by_split": counts,
        "warning": "DPO pairs encode ranking preference, not independent pairwise votes; retain a held-out image split for evaluation.",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = ArgumentParser(description="Build New Yorker compact-conditioned DPO preference pairs.")
    parser.add_argument("--compact-dir", type=Path, default=Path("data/processed/newyorker_compact_sft"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/newyorker_caption_ranking"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/newyorker_compact_dpo"))
    parser.add_argument("--rejected-percentile", type=float, default=0.50)
    args = parser.parse_args()
    print(json.dumps(build_pairs(args.compact_dir, args.raw_dir, args.output_dir, args.rejected_percentile), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
