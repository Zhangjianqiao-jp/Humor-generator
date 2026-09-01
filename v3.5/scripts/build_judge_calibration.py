#!/usr/bin/env python3
"""Build five non-test, crowd-ranked calibration pairs for the blind judge."""
from __future__ import annotations

from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/processed/latent_bridge_v35"
    )
    parser.add_argument(
        "--ranking-dir", type=Path,
        default=ROOT / "data/external/benchmarks/humor_in_ai/ranking/source",
    )
    parser.add_argument(
        "--image-dir", type=Path,
        default=ROOT / "data/external/benchmarks/humor_in_ai/cartoons/source",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/cache/judge_calibration.json"
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--minimum-mean-margin", type=float, default=0.5)
    args = parser.parse_args()

    train_contests = sorted({
        int(row["contest_number"])
        for row in read_jsonl(args.dataset / "train.jsonl")
        if row["dataset"] == "humor_in_ai"
    })
    forbidden = {
        int(row["contest_number"])
        for split in (
            "validation", "internal_test", "official_hia_unseen_test",
            "official_hia_seen_diagnostic",
        )
        for row in read_jsonl(args.dataset / f"{split}.jsonl")
    }
    candidates = []
    for contest in train_contests:
        ranking = args.ranking_dir / f"{contest}.csv"
        image = args.image_dir / f"{contest}.jpg"
        if contest in forbidden or not ranking.is_file() or not image.is_file():
            continue
        with ranking.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if str(row.get("caption", "")).strip()]
        if len(rows) < 20:
            continue
        for offset, row in enumerate(rows):
            row["_rank"] = int(row["rank"]) if row.get("rank", "").strip() else offset
        rows.sort(key=lambda row: int(row["_rank"]))
        winner = rows[0]
        lower = [
            row for row in rows[len(rows) // 2 :]
            if float(winner["mean"]) - float(row["mean"]) >= args.minimum_mean_margin
        ]
        if not lower:
            continue
        # Use the closest still-clear preference, rather than a trivial bottom
        # caption that would calibrate only obvious-error detection.
        loser = max(lower, key=lambda row: float(row["mean"]))
        margin = float(winner["mean"]) - float(loser["mean"])
        if margin < args.minimum_mean_margin:
            continue
        candidates.append((contest, image.resolve(), winner, loser, margin))
    if len(candidates) < 5:
        raise RuntimeError(f"only {len(candidates)} eligible calibration contests")
    rng = random.Random(args.seed)
    selected = rng.sample(candidates, 5)
    examples, provenance = [], []
    for index, (contest, image, winner, loser, margin) in enumerate(selected):
        winner_on_a = index % 2 == 0
        examples.append({
            "image": str(image),
            "caption_A": winner["caption"] if winner_on_a else loser["caption"],
            "caption_B": loser["caption"] if winner_on_a else winner["caption"],
            "answer": "A" if winner_on_a else "B",
        })
        provenance.append({
            "contest_number": contest,
            "winner_rank": int(winner["_rank"]),
            "loser_rank": int(loser["_rank"]),
            "winner_mean": float(winner["mean"]),
            "loser_mean": float(loser["mean"]),
            "mean_margin": margin,
            "source": "yguooo/newyorker_caption_ranking crowd ranking",
            "license": "CC-BY-NC-4.0",
        })
    serialized = json.dumps(examples, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "examples": len(examples),
        "test_clusters_used": 0,
        "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "provenance": provenance,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
