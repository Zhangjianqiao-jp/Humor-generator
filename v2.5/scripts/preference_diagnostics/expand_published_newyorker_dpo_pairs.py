#!/usr/bin/env python3
"""Build a nested, diversity-aware expansion of the published-rule DPO pairs.

The existing selected pairs are retained verbatim as a low-data subset. Extra
training pairs come only from the already materialized Zhang et al. (NeurIPS
2024) three-sigma candidate pool. Labels are never generated or reversed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return int(row["contest_number"]), str(row["chosen"]), str(row["rejected"])


def relative_length_difference(row: dict[str, Any]) -> float:
    chosen = str(row["chosen"])
    rejected = str(row["rejected"])
    return abs(len(chosen) - len(rejected)) / max(len(chosen), len(rejected), 1)


def expand_train(
    candidates: list[dict[str, Any]],
    base: list[dict[str, Any]],
    per_contest: int,
    max_relative_length_difference: float,
    seed: int,
) -> list[dict[str, Any]]:
    by_contest: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_candidates: set[tuple[int, str, str]] = set()
    for row in candidates:
        key = pair_key(row)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        if float(row["z_margin"]) < 3.0:
            continue
        if relative_length_difference(row) > max_relative_length_difference:
            continue
        by_contest[key[0]].append(row)

    base_by_contest: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in base:
        base_by_contest[int(row["contest_number"])].append(row)

    expanded: list[dict[str, Any]] = []
    for contest in sorted(base_by_contest):
        initial = list(base_by_contest[contest])
        if len(initial) > per_contest:
            raise ValueError(f"Contest {contest}: base size exceeds requested expansion")
        selected_keys = {pair_key(row) for row in initial}
        caption_use: Counter[str] = Counter()
        for row in initial:
            caption_use[str(row["chosen"])] += 1
            caption_use[str(row["rejected"])] += 1

        rng = random.Random(seed + contest)
        tie_break = {pair_key(row): rng.random() for row in by_contest[contest]}
        remaining = [row for row in by_contest[contest] if pair_key(row) not in selected_keys]

        while len(initial) < per_contest:
            if not remaining:
                raise ValueError(f"Contest {contest}: not enough eligible unique pairs")
            best_index = min(
                range(len(remaining)),
                key=lambda index: (
                    int(caption_use[str(remaining[index]["chosen"])] > 0)
                    + int(caption_use[str(remaining[index]["rejected"])] > 0),
                    caption_use[str(remaining[index]["chosen"])]
                    + caption_use[str(remaining[index]["rejected"])],
                    int(remaining[index]["chosen_rank"]),
                    tie_break[pair_key(remaining[index])],
                ),
            )
            source = remaining.pop(best_index)
            row = dict(source)
            row["pair_id"] = (
                f"published_nips24_expanded_train_{contest}_{len(initial):03d}"
            )
            row["selection"] = {
                "view": "expanded_nested_diversity_aware",
                "base_pairs_retained": len(base_by_contest[contest]),
                "target_pairs_per_contest": per_contest,
                "max_relative_length_difference": max_relative_length_difference,
                "labels_unchanged": True,
            }
            initial.append(row)
            selected_keys.add(pair_key(row))
            caption_use[str(row["chosen"])] += 1
            caption_use[str(row["rejected"])] += 1

        expanded.extend(initial)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/processed/newyorker_published_dpo_pairs")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/newyorker_published_dpo_pairs_expanded64"),
    )
    parser.add_argument("--train-pairs-per-contest", type=int, default=64)
    parser.add_argument("--max-relative-length-difference", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    base_train = read_jsonl(args.input_dir / "train_selected.jsonl")
    expanded_train = expand_train(
        read_jsonl(args.input_dir / "train.jsonl"),
        base_train,
        per_contest=args.train_pairs_per_contest,
        max_relative_length_difference=args.max_relative_length_difference,
        seed=args.seed,
    )
    write_jsonl(args.output_dir / "train_selected.jsonl", expanded_train)

    # Keep held-out data byte-for-byte equivalent at the row level so data-size
    # comparisons change only the training pairs.
    held_out: dict[str, list[dict[str, Any]]] = {}
    for split in ("validation", "test"):
        held_out[split] = read_jsonl(args.input_dir / f"{split}_selected.jsonl")
        write_jsonl(args.output_dir / f"{split}_selected.jsonl", held_out[split])

    base_keys = {pair_key(row) for row in base_train}
    expanded_keys = {pair_key(row) for row in expanded_train}
    if not base_keys <= expanded_keys:
        raise RuntimeError("Expanded training data is not a strict superset of the pilot data")

    manifest = {
        "source": "Zhang et al., Humor in AI, NeurIPS 2024",
        "source_dataset": "yguooo/newyorker_caption_ranking",
        "source_candidate_manifest": str(args.input_dir / "manifest.json"),
        "source_candidate_manifest_sha256": sha256(args.input_dir / "manifest.json"),
        "construction": "nested diversity-aware expansion from published-rule three-sigma pairs",
        "labels_generated": False,
        "labels_changed": False,
        "seed": args.seed,
        "train_pairs_per_contest": args.train_pairs_per_contest,
        "max_relative_length_difference": args.max_relative_length_difference,
        "pilot_train_pairs_retained": len(base_train),
        "splits": {},
        "license": "CC-BY-NC-4.0",
    }
    for split, rows in {
        "train": expanded_train,
        "validation": held_out["validation"],
        "test": held_out["test"],
    }.items():
        path = args.output_dir / f"{split}_selected.jsonl"
        manifest["splits"][split] = {
            "path": str(path),
            "sha256": sha256(path),
            "pairs": len(rows),
            "contests": len({int(row["contest_number"]) for row in rows}),
            "unique_chosen": len({str(row["chosen"]) for row in rows}),
            "unique_rejected": len({str(row["rejected"]) for row in rows}),
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
