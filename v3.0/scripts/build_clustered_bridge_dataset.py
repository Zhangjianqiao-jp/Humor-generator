#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from humor_generator_v3.data.clustered import build_clustered_bridge_dataset


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "d1334f295cc1a8f8f6dc67ba7e846c5939dddcec"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--captions-per-source", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/latent_bridge_v3")
    parser.add_argument(
        "--tracked-manifest",
        type=Path,
        default=ROOT / "manifests/image_clustered_dataset.json",
    )
    args = parser.parse_args()
    result = build_clustered_bridge_dataset(
        official_description_root=ROOT / f"data/external/homer_official/{COMMIT}/data/datasets",
        humor_in_ai_root=ROOT / "data/external/benchmarks/humor_in_ai",
        electronic_sheep_root=ROOT / "data/external/benchmarks/electronic_sheep",
        adapter_seen_manifest=ROOT / "manifests/frozen_adapter_seen_clusters.json",
        output=args.output,
        seed=args.seed,
        captions_per_source=args.captions_per_source,
    )
    args.tracked_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.tracked_manifest.write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
