#!/usr/bin/env python3
"""Summarize the auxiliary 7B Base-vs-SFT held-out evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl

TOKENS = re.compile(r"\b\w+(?:['’]\w+)?\b")
GENERIC = re.compile(r"\b(?:pov|bro|meanwhile|lol|lmao|bruh)\b|[💀😂]", re.IGNORECASE)


def generation_metrics(path: Path) -> dict[str, float]:
    rows = read_jsonl(path)
    captions = [str(x).strip() for row in rows for x in row.get("candidates", []) if str(x).strip()]
    tokenized = [[token.lower() for token in TOKENS.findall(text)] for text in captions]
    unigrams = [token for tokens in tokenized for token in tokens]
    bigrams = [tuple(tokens[i : i + 2]) for tokens in tokenized for i in range(len(tokens) - 1)]
    return {
        "num_images": len(rows),
        "num_captions": len(captions),
        "mean_words": sum(len(tokens) for tokens in tokenized) / max(len(captions), 1),
        "distinct_1": len(set(unigrams)) / max(len(unigrams), 1),
        "distinct_2": len(set(bigrams)) / max(len(bigrams), 1),
        "generic_template_rate": sum(bool(GENERIC.search(text)) for text in captions) / max(len(captions), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/7b-generator-eval"))
    parser.add_argument("--output", type=Path, default=Path("results/7b_generator/sft_base_comparison.json"))
    args = parser.parse_args()
    systems = {}
    for name in ("base", "sft"):
        directory = args.root / name
        judge = json.loads((directory / "judge_summary.json").read_text(encoding="utf-8"))
        systems[name] = {
            "qualified_rate": judge["qualified_rate"],
            "hallucination_rate": judge["hallucination_rate"],
            "avg_scores": judge["avg_scores"],
            **generation_metrics(directory / "generations.jsonl"),
        }
    base, sft = systems["base"], systems["sft"]
    payload = {
        "scope": "7B-only Base vs SFT auxiliary evaluation",
        "systems": systems,
        "sft_minus_base": {
            "qualified_rate": sft["qualified_rate"] - base["qualified_rate"],
            "hallucination_rate": sft["hallucination_rate"] - base["hallucination_rate"],
            **{
                key: sft["avg_scores"][key] - base["avg_scores"][key]
                for key in sorted(base["avg_scores"])
            },
        },
        "decision_note": "Do not select SFT from this auxiliary judge alone; complete the blind group packet.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
