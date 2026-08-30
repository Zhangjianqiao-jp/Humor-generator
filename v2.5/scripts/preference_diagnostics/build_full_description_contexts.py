#!/usr/bin/env python3
"""Materialize one captioner prompt context for every official NYCC split image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_newyorker_compact_sft import CAPTION_PROMPT, build_compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/newyorker_caption_ranking")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/newyorker_full_description_contexts"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source": "yguooo/newyorker_caption_ranking:gpt4o_description",
        "prompt_policy": "description-only compact plan; no caption-derived fields",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        source = args.raw_dir / "gpt4o_description" / f"{split}.jsonl"
        output = args.output_dir / f"caption_{split}.jsonl"
        rows = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                desc = json.loads(line)
                contest = int(desc["contest_number"])
                image = args.raw_dir / "cartoons" / "source" / f"{contest}.jpg"
                ranking = args.raw_dir / "ranking" / "source" / f"{contest}.csv"
                if not image.is_file() or not ranking.is_file():
                    raise FileNotFoundError(f"Contest {contest}: image or ranking missing")
                compact = build_compact(desc)
                prompt = f"{CAPTION_PROMPT}\n\nHumor plan:\n{compact}"
                rows.append(
                    {
                        "image": str(image),
                        "image_id": f"nycc_{contest}",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": str(image)},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                        "meta": {
                            "contest_number": contest,
                            "source_split": split,
                            "compact": compact,
                            "compact_label_source": "release_gpt4o_description_only",
                            "auto_compact": True,
                        },
                    }
                )
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {"path": str(output), "contexts": len(rows)}

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
