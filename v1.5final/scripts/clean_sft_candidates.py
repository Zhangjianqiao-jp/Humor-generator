#!/usr/bin/env python
from __future__ import annotations

import json
import re
import statistics
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.sft_dataset import clean_generated_caption
from src.utils.io import read_jsonl, write_jsonl

PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
BAD_PREFIXES = ("caption:", "candidate:", "answer:", "humorous caption:")
EXPLANATION_MARKERS = (
    "because ",
    "the joke",
    "this is funny",
    "it is funny",
    "why it's funny",
    "why it is funny",
    "explanation",
)
GENERIC_PATTERNS = (
    "i don't know",
    "what are you doing",
    "that's funny",
    "this is awkward",
    "that moment when",
    "oh my god",
    "i'm sorry",
    "i can't help it",
    "i'll take care of it",
    "i've been looking for you",
    "i'm glad you're here",
)
GENERIC_EXACT = {
    "laughter",
    "laughs",
    "gentlemen",
    "oh my god",
    "dad",
    "hey",
    "sorry",
}
VIOLENT_PATTERNS = (
    "kill",
    "die",
    "dead",
    "shoot",
    "gun",
    "blood",
)


def normalize_for_dedupe(text: str) -> str:
    text = text.strip().lower()
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def has_repeated_ngram(text: str, n: int = 3, max_repeats: int = 2) -> bool:
    tokens = normalize_for_dedupe(text).split()
    if len(tokens) < n * max_repeats:
        return False
    counts: Counter[tuple[str, ...]] = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return any(count >= max_repeats for count in counts.values())


def has_repeated_phrase(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b(\w+)(?:\s+\1\b){2,}", lowered):
        return True
    parts = [part.strip() for part in re.split(r"[.!?;]+", lowered) if part.strip()]
    counts = Counter(parts)
    return any(count >= 2 and len(part.split()) >= 3 for part, count in counts.items())


def candidate_drop_reason(
    candidate: str,
    prompt: str,
    max_chars: int,
    max_words: int,
    drop_generic: bool,
    drop_violent: bool,
) -> str | None:
    stripped = candidate.strip()
    lowered = stripped.lower()
    if not stripped:
        return "empty"
    if prompt and prompt.strip().lower() in lowered:
        return "prompt_echo"
    if any(lowered.startswith(prefix) for prefix in BAD_PREFIXES):
        return "caption_prefix"
    if len([line for line in stripped.splitlines() if line.strip()]) > 1:
        return "multiline"
    if len(stripped) > max_chars:
        return "too_long_chars"
    if len(stripped.split()) > max_words:
        return "too_long_words"
    if any(marker in lowered for marker in EXPLANATION_MARKERS):
        return "explanation"
    if has_repeated_phrase(stripped):
        return "repeated_phrase"
    if has_repeated_ngram(stripped):
        return "repeated_ngram"
    normalized = normalize_for_dedupe(stripped)
    if drop_generic and (
        normalized in GENERIC_EXACT
        or any(pattern in lowered for pattern in GENERIC_PATTERNS)
        or any(normalize_for_dedupe(pattern) in normalized for pattern in GENERIC_PATTERNS)
    ):
        return "generic"
    if drop_violent and any(pattern in lowered for pattern in VIOLENT_PATTERNS):
        return "violent_pattern"
    return None


def candidate_flags(candidate: str) -> dict[str, bool]:
    lowered = candidate.lower()
    return {
        "generic": normalize_for_dedupe(candidate) in GENERIC_EXACT or any(pattern in lowered for pattern in GENERIC_PATTERNS) or any(normalize_for_dedupe(pattern) in normalize_for_dedupe(candidate) for pattern in GENERIC_PATTERNS),
        "violent_pattern": any(pattern in lowered for pattern in VIOLENT_PATTERNS),
        "repeated_phrase": has_repeated_phrase(candidate),
        "repeated_ngram": has_repeated_ngram(candidate),
    }


def collect_rows(rows: list[dict[str, Any]], group_by_image: bool) -> list[dict[str, Any]]:
    if not group_by_image:
        return rows

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        image = str(row.get("image") or "")
        if image not in grouped:
            grouped[image] = {
                "image": image,
                "image_id": row.get("image_id"),
                "gold_captions": [],
                "prompt": row.get("prompt"),
                "candidates": [],
            }
        gold = str(row.get("gold_caption") or "").strip()
        if gold and gold not in grouped[image]["gold_captions"]:
            grouped[image]["gold_captions"].append(gold)
        grouped[image]["candidates"].extend(row.get("candidates") or [])
    return list(grouped.values())


def clean_row(
    row: dict[str, Any],
    max_chars: int,
    max_words: int,
    target_candidates: int,
    min_candidates: int,
    drop_generic: bool,
    drop_violent: bool,
) -> tuple[dict[str, Any], Counter[str]]:
    prompt = str(row.get("prompt") or "")
    seen: set[str] = set()
    kept: list[str] = []
    drop_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()

    for raw_candidate in row.get("candidates") or []:
        candidate = clean_generated_caption(str(raw_candidate), prompt=prompt)
        for flag, value in candidate_flags(candidate).items():
            if value:
                flag_counts[flag] += 1
        reason = candidate_drop_reason(
            candidate=candidate,
            prompt=prompt,
            max_chars=max_chars,
            max_words=max_words,
            drop_generic=drop_generic,
            drop_violent=drop_violent,
        )
        if reason is not None:
            drop_counts[reason] += 1
            continue
        norm = normalize_for_dedupe(candidate)
        if norm in seen:
            drop_counts["duplicate"] += 1
            continue
        seen.add(norm)
        kept.append(candidate)
        if target_candidates > 0 and len(kept) >= target_candidates:
            break

    if len(kept) < min_candidates:
        drop_counts["below_min_candidates"] += 1

    cleaned = {
        "image": row.get("image"),
        "image_id": row.get("image_id"),
        "gold_caption": row.get("gold_caption"),
        "gold_captions": row.get("gold_captions"),
        "prompt": prompt,
        "candidates": kept,
        "diagnostics": {
            "raw_candidate_count": len(row.get("candidates") or []),
            "kept_candidate_count": len(kept),
            "drop_counts": dict(drop_counts),
            "flag_counts_before_filter": dict(flag_counts),
        },
    }
    return cleaned, drop_counts


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def clean_candidates(
    input_jsonl: Path,
    output_jsonl: Path,
    summary_json: Path | None,
    group_by_image: bool,
    max_chars: int,
    max_words: int,
    target_candidates: int,
    min_candidates: int,
    drop_generic: bool,
    drop_violent: bool,
) -> None:
    rows = read_jsonl(input_jsonl)
    rows = collect_rows(rows, group_by_image=group_by_image)

    cleaned_rows: list[dict[str, Any]] = []
    total_drop_counts: Counter[str] = Counter()
    for row in rows:
        cleaned, drop_counts = clean_row(
            row=row,
            max_chars=max_chars,
            max_words=max_words,
            target_candidates=target_candidates,
            min_candidates=min_candidates,
            drop_generic=drop_generic,
            drop_violent=drop_violent,
        )
        cleaned_rows.append(cleaned)
        total_drop_counts.update(drop_counts)

    kept_counts = [len(row["candidates"]) for row in cleaned_rows]
    raw_counts = [row["diagnostics"]["raw_candidate_count"] for row in cleaned_rows]
    summary = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "group_by_image": group_by_image,
        "num_output_rows": len(cleaned_rows),
        "total_raw_candidates": sum(raw_counts),
        "total_kept_candidates": sum(kept_counts),
        "avg_raw_candidates_per_row": mean([float(x) for x in raw_counts]),
        "avg_kept_candidates_per_row": mean([float(x) for x in kept_counts]),
        "rows_below_min_candidates": sum(1 for count in kept_counts if count < min_candidates),
        "rows_zero_candidates": sum(1 for count in kept_counts if count == 0),
        "drop_counts": dict(total_drop_counts),
        "settings": {
            "max_chars": max_chars,
            "max_words": max_words,
            "target_candidates": target_candidates,
            "min_candidates": min_candidates,
            "drop_generic": drop_generic,
            "drop_violent": drop_violent,
        },
    }

    write_jsonl(output_jsonl, cleaned_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary to {summary_json}")
    print(f"Saved cleaned candidates to {output_jsonl}")


def main() -> None:
    parser = ArgumentParser(description="Clean, dedupe, and diagnose generated humorous caption candidates.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--group-by-image", action="store_true", help="Merge repeated rows for the same image before cleaning.")
    parser.add_argument("--max-chars", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=22)
    parser.add_argument("--target-candidates", type=int, default=10)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--drop-generic", action="store_true")
    parser.add_argument("--drop-violent", action="store_true", help="Optional strict filter for violent words; off by default.")
    args = parser.parse_args()
    clean_candidates(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        group_by_image=args.group_by_image,
        max_chars=args.max_chars,
        max_words=args.max_words,
        target_candidates=args.target_candidates,
        min_candidates=args.min_candidates,
        drop_generic=args.drop_generic,
        drop_violent=args.drop_violent,
    )


if __name__ == "__main__":
    main()
