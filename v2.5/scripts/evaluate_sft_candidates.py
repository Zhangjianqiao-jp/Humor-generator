#!/usr/bin/env python
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from argparse import ArgumentParser
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    "when you",
    "that moment when",
)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def token_f1(a: str, b: str) -> float:
    a_tokens = normalize_text(a).split()
    b_tokens = normalize_text(b).split()
    if not a_tokens or not b_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in a_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in b_tokens:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = normalize_text(text).replace(" ", "_")
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def candidate_flags(candidate: str, prompt: str, max_chars: int) -> dict[str, bool]:
    stripped = candidate.strip()
    lowered = stripped.lower()
    lines = [line for line in stripped.splitlines() if line.strip()]
    return {
        "empty": not stripped,
        "too_long": len(stripped) > max_chars,
        "multiline": len(lines) > 1,
        "caption_prefix": any(lowered.startswith(prefix) for prefix in BAD_PREFIXES),
        "prompt_echo": bool(prompt and prompt.strip().lower() in lowered),
        "explains": any(marker in lowered for marker in EXPLANATION_MARKERS),
        "generic_pattern": any(pattern in lowered for pattern in GENERIC_PATTERNS),
    }


def format_ok(flags: dict[str, bool]) -> bool:
    hard_bad = ["empty", "too_long", "multiline", "caption_prefix", "prompt_echo", "explains"]
    return not any(flags[name] for name in hard_bad)


def get_gold_captions(row: dict[str, Any]) -> list[str]:
    golds: list[str] = []
    if row.get("gold_caption"):
        golds.append(str(row["gold_caption"]).strip())
    for gold in row.get("gold_captions") or []:
        gold = str(gold).strip()
        if gold and gold not in golds:
            golds.append(gold)
    if row.get("gold"):
        gold = str(row["gold"]).strip()
        if gold and gold not in golds:
            golds.append(gold)
    return golds


def score_candidate(
    candidate: str,
    golds: list[str],
    prompt: str,
    max_chars: int,
    similarity_threshold: float,
    candidate_index: int,
) -> dict[str, Any]:
    best = {"seq_ratio": 0.0, "token_f1": 0.0, "char3_jaccard": 0.0, "max_text_similarity": 0.0, "best_gold_caption": ""}
    for gold in golds:
        seq = sequence_ratio(candidate, gold)
        f1 = token_f1(candidate, gold)
        c3 = jaccard(char_ngrams(candidate), char_ngrams(gold))
        max_sim = max(seq, f1, c3)
        if max_sim > best["max_text_similarity"]:
            best = {
                "seq_ratio": seq,
                "token_f1": f1,
                "char3_jaccard": c3,
                "max_text_similarity": max_sim,
                "best_gold_caption": gold,
            }
    flags = candidate_flags(candidate, prompt=prompt, max_chars=max_chars)
    return {
        "candidate_index": candidate_index,
        "candidate": candidate,
        **best,
        "text_match_score": int(best["max_text_similarity"] >= similarity_threshold),
        "format_ok": format_ok(flags),
        "flags": flags,
        "num_chars": len(candidate.strip()),
        "num_words": len(candidate.strip().split()),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def evaluate_candidates(
    input_jsonl: Path,
    output_jsonl: Path | None,
    summary_json: Path | None,
    similarity_threshold: float,
    max_chars: int,
    print_samples: int,
) -> None:
    rows = read_jsonl(input_jsonl)
    evaluated_rows: list[dict[str, Any]] = []
    all_candidate_metrics: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        golds = get_gold_captions(row)
        prompt = str(row.get("prompt") or "")
        candidates = row.get("candidates") or []
        if not candidates and row.get("generated_caption"):
            candidates = [row["generated_caption"]]
        candidate_metrics = [
            score_candidate(
                str(candidate),
                golds,
                prompt,
                max_chars,
                similarity_threshold=similarity_threshold,
                candidate_index=candidate_index,
            )
            for candidate_index, candidate in enumerate(candidates, start=1)
        ]
        all_candidate_metrics.extend(candidate_metrics)
        best = max(candidate_metrics, key=lambda item: item["max_text_similarity"], default=None)
        evaluated_rows.append(
            {
                "image": row.get("image"),
                "image_id": row.get("image_id"),
                "gold_caption": golds[0] if golds else "",
                "gold_captions": golds,
                "prompt": prompt,
                "best_by_text_similarity": None if best is None else best["candidate"],
                "max_text_similarity": 0.0 if best is None else best["max_text_similarity"],
                "gold_match_score": int(best is not None and best["max_text_similarity"] >= similarity_threshold),
                "num_text_matched_candidates": sum(item["text_match_score"] for item in candidate_metrics),
                "num_candidates": len(candidate_metrics),
                "num_format_ok": sum(1 for item in candidate_metrics if item["format_ok"]),
                "candidates": candidate_metrics,
            }
        )

    row_count = len(evaluated_rows)
    candidate_count = len(all_candidate_metrics)
    by_index: dict[int, list[dict[str, Any]]] = {}
    for item in all_candidate_metrics:
        by_index.setdefault(int(item.get("candidate_index", 0)), []).append(item)

    summary = {
        "input_jsonl": str(input_jsonl),
        "num_rows": row_count,
        "num_candidates": candidate_count,
        "similarity_threshold": similarity_threshold,
        "scoring_rule": "image_score=1 if any candidate has max_text_similarity >= similarity_threshold else 0",
        "total_model_score": sum(row["gold_match_score"] for row in evaluated_rows),
        "total_possible_score": row_count,
        "model_score_rate": mean([row["gold_match_score"] for row in evaluated_rows]),
        "gold_match_rate": mean([row["gold_match_score"] for row in evaluated_rows]),
        "candidate_text_match_count": sum(item["text_match_score"] for item in all_candidate_metrics),
        "candidate_text_match_rate": mean([item["text_match_score"] for item in all_candidate_metrics]),
        "avg_text_matched_candidates_per_image": mean([row["num_text_matched_candidates"] for row in evaluated_rows]),
        "avg_max_text_similarity_per_image": mean([row["max_text_similarity"] for row in evaluated_rows]),
        "avg_seq_ratio": mean([item["seq_ratio"] for item in all_candidate_metrics]),
        "avg_token_f1": mean([item["token_f1"] for item in all_candidate_metrics]),
        "avg_char3_jaccard": mean([item["char3_jaccard"] for item in all_candidate_metrics]),
        "format_ok_rate": mean([float(item["format_ok"]) for item in all_candidate_metrics]),
        "avg_chars": mean([item["num_chars"] for item in all_candidate_metrics]),
        "by_candidate_index": {
            str(index): {
                "count": len(items),
                "avg_max_text_similarity": mean([item["max_text_similarity"] for item in items]),
                "text_match_rate": mean([item["text_match_score"] for item in items]),
                "format_ok_rate": mean([float(item["format_ok"]) for item in items]),
            }
            for index, items in sorted(by_index.items())
            if index > 0
        },
        "flag_rates": {},
    }
    flag_names = sorted({name for item in all_candidate_metrics for name in item["flags"]})
    for name in flag_names:
        summary["flag_rates"][name] = mean([float(item["flags"].get(name, False)) for item in all_candidate_metrics])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if print_samples > 0:
        print("\nLowest text-similarity samples:")
        for row in sorted(evaluated_rows, key=lambda item: item["max_text_similarity"])[:print_samples]:
            print("=" * 80)
            print(f"Image: {row.get('image')}")
            print(f"Gold: {row.get('gold_caption')}")
            print(f"Best similarity: {row.get('max_text_similarity'):.3f}")
            print(f"Best candidate: {row.get('best_by_text_similarity')}")

    if output_jsonl is not None:
        write_jsonl(output_jsonl, evaluated_rows)
        print(f"Saved per-row evaluation to {output_jsonl}")
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary to {summary_json}")


def main() -> None:
    parser = ArgumentParser(description="Evaluate generated SFT candidate captions with cheap text/format diagnostics.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--max-chars", type=int, default=160)
    parser.add_argument("--print-samples", type=int, default=5)
    args = parser.parse_args()
    evaluate_candidates(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        similarity_threshold=args.similarity_threshold,
        max_chars=args.max_chars,
        print_samples=args.print_samples,
    )


if __name__ == "__main__":
    main()
