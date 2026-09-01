#!/usr/bin/env python3
"""Select official New Yorker preferences with strong chosen captions and mixed difficulty.

This changes only local pair selection. Preference directions and scores remain
those reconstructed from the crowd rankings of Zhang et al. (NeurIPS 2024).
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def length_gap(row: dict[str, Any]) -> float:
    a, b = str(row["chosen"]), str(row["rejected"])
    return abs(len(a) - len(b)) / max(len(a), len(b), 1)


def tier(row: dict[str, Any], ranking_size: int) -> str | None:
    chosen_pct = (int(row["chosen_rank"]) + 1) / ranking_size
    rejected_pct = (int(row["rejected_rank"]) + 1) / ranking_size
    z = float(row["z_margin"])
    if chosen_pct <= 0.10 and rejected_pct >= 0.50 and z >= 4.5:
        return "clear"
    if chosen_pct <= 0.20 and rejected_pct >= 0.40 and z >= 3.5:
        return "medium"
    if chosen_pct <= 0.25 and rejected_pct - chosen_pct >= 0.20 and z >= 3.0:
        return "hard"
    return None


def choose(rows: list[dict[str, Any]], quota: int, usage: Counter[str], used: set[tuple[str, str]], difficulty: str) -> list[dict[str, Any]]:
    candidates = [r for r in rows if (r["chosen"], r["rejected"]) not in used]
    if difficulty == "hard":
        candidates.sort(key=lambda r: (abs(float(r["z_margin"]) - 3.0), int(r["chosen_rank"]), -float(r["score_margin"])))
    else:
        candidates.sort(key=lambda r: (int(r["chosen_rank"]), -float(r["score_margin"]), -float(r["z_margin"])))
    selected = []
    for cap in (2, 3, 10**9):
        while len(selected) < quota:
            eligible = [r for r in candidates if max(usage[str(r["chosen"])], usage[str(r["rejected"])]) < cap]
            if not eligible:
                break
            row = min(eligible, key=lambda r: (usage[str(r["chosen"])] + usage[str(r["rejected"])], candidates.index(r)))
            candidates.remove(row); selected.append(row); used.add((row["chosen"], row["rejected"]))
            usage[str(row["chosen"])] += 1; usage[str(row["rejected"])] += 1
        if len(selected) == quota:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/newyorker_published_dpo_pairs_fullsplit64"))
    parser.add_argument("--ranking-dir", type=Path, default=Path("data/raw/newyorker_caption_ranking/ranking/source"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/newyorker_published_dpo_pairs_quality64"))
    parser.add_argument("--pairs-per-contest", type=int, default=64)
    parser.add_argument("--max-relative-length-difference", type=float, default=0.35)
    args = parser.parse_args()

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input_dir / "train.jsonl"):
        if float(row["z_margin"]) >= 3 and length_gap(row) <= args.max_relative_length_difference:
            grouped[int(row["contest_number"])].append(row)
    output, shortages = [], {}
    desired = {"clear": round(args.pairs_per_contest * 0.375), "medium": round(args.pairs_per_contest * 0.375)}
    desired["hard"] = args.pairs_per_contest - desired["clear"] - desired["medium"]
    for contest in sorted(grouped):
        ranking_path = args.ranking_dir / f"{contest}.csv"
        ranking_size = max(sum(1 for _ in ranking_path.open(encoding="utf-8-sig")) - 1, 1)
        pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen = set()
        for row in grouped[contest]:
            key = (str(row["chosen"]), str(row["rejected"]))
            if key in seen:
                continue
            seen.add(key)
            label = tier(row, ranking_size)
            if label:
                pools[label].append(row)
        usage: Counter[str] = Counter(); used: set[tuple[str, str]] = set(); selected = []
        for label in ("clear", "medium", "hard"):
            picked = choose(pools[label], desired[label], usage, used, label)
            for source in picked:
                row = dict(source); row["quality_tier"] = label; selected.append(row)
        # Fill any shortage from all eligible top-quartile pairs; never relax label direction,
        # three-sigma confidence, chosen-quality ceiling, or length matching.
        eligible = [r for values in pools.values() for r in values]
        fill = choose(eligible, args.pairs_per_contest - len(selected), usage, used, "hard")
        for source in fill:
            row = dict(source); row["quality_tier"] = f"{tier(source, ranking_size)}_quota_fallback"; selected.append(row)
        if len(selected) < args.pairs_per_contest:
            shortages[str(contest)] = len(selected)
        for index, row in enumerate(selected):
            row["pair_id"] = f"published_nips24_quality64_train_{contest}_{index:03d}"
            row["selection"] = {
                "view": "quality_stratified_length_matched",
                "target_mix": desired,
                "chosen_rank_percentile_max": 0.25,
                "z_margin_min": 3.0,
                "max_relative_length_difference": args.max_relative_length_difference,
                "caption_reuse_soft_cap": 2,
                "labels_unchanged": True,
            }
            output.append(row)
    write_jsonl(args.output_dir / "train_selected.jsonl", output)
    for split in ("validation", "test"):
        write_jsonl(args.output_dir / f"{split}_selected.jsonl", read_jsonl(args.input_dir / f"{split}_selected.jsonl"))
    counts = Counter(str(row["quality_tier"]) for row in output)
    manifest = {
        "source": "Zhang et al., Humor in AI, NeurIPS 2024",
        "preference_labels_generated_or_changed": False,
        "purpose": "increase absolute chosen quality while retaining clear/medium/hard preference coverage",
        "target_mix_per_contest": desired,
        "pairs": len(output), "contests": len(grouped), "tier_counts": dict(counts), "shortages": shortages,
        "train_sha256": sha256(args.output_dir / "train_selected.jsonl"),
        "validation_sha256": sha256(args.output_dir / "validation_selected.jsonl"),
        "test_sha256": sha256(args.output_dir / "test_selected.jsonl"),
        "license": "CC-BY-NC-4.0",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
