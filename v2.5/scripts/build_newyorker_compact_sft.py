#!/usr/bin/env python
"""Create compact-planner and plan-conditioned caption SFT data from NYCC.

The compact target is deliberately short and auditable.  All three fields are
derived only from the release's GPT-4o visual descriptions.  In particular,
ANGLE is a *strategy class*, never text from a human caption; putting a gold
caption (or a paraphrase of it) in the planner input would leak the answer to
the captioner and invalidate SFT/DPO evaluation.
"""
from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl

PLANNER_PROMPT = """Study the image and return a compact humor plan.\nReturn exactly three lines and no explanation:\nANCHOR: key visible object or action\nCONTRAST: violated expectation\nANGLE: concise punchline direction"""
CAPTION_PROMPT = "Generate one short, natural, image-specific humorous caption. Do not explain."
SPACE_RE = re.compile(r"\s+")
FIRST_SENTENCE_RE = re.compile(r"^.*?[.!?](?=\s|$)")


def compact_text(value: str) -> str:
    """Return a complete first sentence instead of a mid-phrase word slice."""
    normalized = SPACE_RE.sub(" ", value).strip()
    match = FIRST_SENTENCE_RE.match(normalized)
    return match.group(0) if match else normalized


def descriptions(raw_dir: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        with (raw_dir / "gpt4o_description" / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["source_split"] = split
                records[int(row["contest_number"])] = row
    return records


def angle_strategy(desc: dict[str, Any]) -> str:
    """Choose a non-caption, reusable punchline direction from the description."""
    text = " ".join(
        [str(desc.get("canny") or ""), str(desc.get("uncanny") or ""), str(desc.get("location") or "")]
    ).lower()
    if any(word in text for word in ("therap", "doctor", "hospital", "medical", "pharmacy")):
        return "deadpan professional framing"
    if any(word in text for word in ("office", "meeting", "business", "work", "briefcase")):
        return "bureaucratic workplace framing"
    if any(word in text for word in ("restaurant", "waiter", "chef", "food", "dining", "pizza")):
        return "literal service-industry framing"
    if any(word in text for word in ("caveman", "dinosaur", "fairytale", "medieval", "ancient", "pirate")):
        return "anachronism treated as routine"
    if any(word in text for word in ("animal", "dog", "cat", "mouse", "fish", "dragon", "wave", "vegetable")):
        return "treat the nonhuman as ordinary"
    return "dry literal escalation"


def build_compact(desc: dict[str, Any]) -> str:
    anchor = compact_text(str(desc["canny"]))
    contrast = compact_text(str(desc["uncanny"]))
    angle = angle_strategy(desc)
    return f"ANCHOR: {anchor}\nCONTRAST: {contrast}\nANGLE: {angle}"


def replace_user_text(row: dict[str, Any], text: str) -> dict[str, Any]:
    copied = json.loads(json.dumps(row))
    for message in copied["messages"]:
        if message.get("role") == "user":
            for item in message.get("content", []):
                if item.get("type") == "text":
                    item["text"] = text
                    return copied
    raise ValueError("SFT row has no user text content.")


def build_datasets(
    high_score_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    planner_prompt: str,
    caption_prompt: str,
) -> dict[str, Any]:
    desc_by_contest = descriptions(raw_dir)
    high_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        rows = read_jsonl(high_score_dir / f"sft_{split}.jsonl")
        high_rows[split] = rows

    output_dir.mkdir(parents=True, exist_ok=True)
    compact_by_contest: dict[int, str] = {}
    planner_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, rows in high_rows.items():
        seen: set[int] = set()
        for source_row in rows:
            contest = int(source_row["meta"]["contest_number"])
            if contest in seen or contest not in desc_by_contest:
                continue
            seen.add(contest)
            compact = build_compact(desc_by_contest[contest])
            compact_by_contest[contest] = compact
            image = str(source_row["image"])
            planner_rows[split].append(
                {
                    "image": image,
                    "image_id": source_row["image_id"],
                    "messages": [
                        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": planner_prompt}]},
                        {"role": "assistant", "content": [{"type": "text", "text": compact}]},
                    ],
                    "meta": {
                        **source_row["meta"],
                        "task": "compact_humor_planning",
                        "label_source": "release_gpt4o_description_only",
                        "auto_compact": True,
                    },
                }
            )

    caption_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dropped_without_compact = 0
    for split, rows in high_rows.items():
        for row in rows:
            contest = int(row["meta"]["contest_number"])
            compact = compact_by_contest.get(contest)
            if compact is None:
                dropped_without_compact += 1
                continue
            conditioned_prompt = f"{caption_prompt}\n\nHumor plan:\n{compact}"
            conditioned = replace_user_text(row, conditioned_prompt)
            conditioned["meta"] = {
                **conditioned["meta"],
                "task": "compact_conditioned_humor_captioning",
                "compact": compact,
                "compact_label_source": "release_gpt4o_description_only",
                "auto_compact": True,
            }
            caption_rows[split].append(conditioned)

    for split in ("train", "validation", "test"):
        write_jsonl(output_dir / f"planner_{split}.jsonl", planner_rows[split])
        write_jsonl(output_dir / f"caption_{split}.jsonl", caption_rows[split])
    manifest = {
        "source_high_score_dir": str(high_score_dir),
        "source_description_dir": str(raw_dir / "gpt4o_description"),
        "planner_prompt": planner_prompt,
        "caption_prompt": caption_prompt,
        "compact_schema": ["ANCHOR", "CONTRAST", "ANGLE"],
        "compact_text_policy": "first_complete_sentence",
        "planner_rows_by_split": {split: len(planner_rows[split]) for split in ("train", "validation", "test")},
        "caption_rows_by_split": {split: len(caption_rows[split]) for split in ("train", "validation", "test")},
        "dropped_high_score_rows_without_description": dropped_without_compact,
        "label_warning": "Compact labels are automatic bootstrap labels from visual descriptions only; ANGLE is a strategy class, not a caption. Audit before treating them as gold supervision.",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = ArgumentParser(description="Build compact planner and conditioned-caption SFT JSONL from New Yorker data.")
    parser.add_argument("--high-score-dir", type=Path, default=Path("data/processed/newyorker_top3pct_sft"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/newyorker_caption_ranking"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/newyorker_compact_sft"))
    parser.add_argument("--planner-prompt", type=str, default=PLANNER_PROMPT)
    parser.add_argument("--caption-prompt", type=str, default=CAPTION_PROMPT)
    args = parser.parse_args()
    print(json.dumps(build_datasets(args.high_score_dir, args.raw_dir, args.output_dir, args.planner_prompt, args.caption_prompt), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
