#!/usr/bin/env python3
"""Blind 10-candidate pools as three disjoint Group-of-3 trials per image/seed."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    grouped = defaultdict(dict)
    for row in read_jsonl(args.input):
        grouped[(row["cluster_id"], int(row["seed"]), row["image"])][row["system"]] = row["candidates"]
    packet, key = [], []
    for (cluster, seed, image), systems in sorted(grouped.items()):
        if len(systems) != 2:
            raise ValueError(f"expected exactly two systems for {cluster}/{seed}")
        names = sorted(systems)
        if any(len(systems[name]) < 10 for name in names):
            raise ValueError(f"10-candidate contract failed for {cluster}/{seed}")
        for trial in range(3):
            token = f"{cluster}|{seed}|{trial}|{args.seed}"
            rng = random.Random(int(hashlib.sha256(token.encode()).hexdigest()[:16], 16))
            order = list(names); rng.shuffle(order)
            groups = {}
            for label, name in zip(("A", "B"), order):
                values = list(systems[name][trial*3:(trial+1)*3]); rng.shuffle(values)
                groups[label] = values
            blind_id = hashlib.sha256(token.encode()).hexdigest()[:16]
            packet.append({
                "blind_id": blind_id, "image": image, "group_A": groups["A"], "group_B": groups["B"],
                "instructions": "Judge overall A/B/Tie and absolute good/weak/bad for each group.",
            })
            key.append({"blind_id": blind_id, "cluster_id": cluster, "generation_seed": seed, "trial": trial, "A": order[0], "B": order[1], "unused_candidate_index": 10})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "blind_packet.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in packet))
    (args.output_dir / "identity_key.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in key))
    manifest = {"items": len(packet), "trials_per_image_seed": 3, "candidates_per_group": 3, "candidate_reuse_within_image_seed": False, "tenth_candidate_reserved_for_best_of_10_metrics": True}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
