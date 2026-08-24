#!/usr/bin/env python3
"""Build image-matched funny/weak/literal examples for the linear probe."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preference.diagnostics import read_jsonl, sha256, text_features, write_json, write_jsonl

FIRST_SENTENCE = re.compile(r"^.*?[.!?](?=\s|$)")


def descriptions(root: Path) -> dict[str, str]:
    output = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        for row in read_jsonl(path):
            value = " ".join(str(row.get("canny") or "").split())
            match = FIRST_SENTENCE.match(value)
            output[f"nycc_{int(row['contest_number'])}"] = match.group(0) if match else value
    return output


def relative_length(a: str, b: str) -> float:
    return abs(len(a) - len(b)) / max(len(a), len(b), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--description-dir", type=Path, default=Path("data/raw/newyorker_caption_ranking/gpt4o_description"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-literal-relative-length-difference", type=float, default=0.65)
    args = parser.parse_args()

    pairs = [row for row in read_jsonl(args.pairs) if row.get("pair_type") == "H2"]
    desc = descriptions(args.description_dir)
    best_by_image: dict[str, dict[str, Any]] = {}
    for row in pairs:
        image_id = str(row["image_id"])
        margin = float(row.get("score_margin") or 0.0)
        if image_id not in best_by_image or margin > float(best_by_image[image_id].get("score_margin") or 0.0):
            best_by_image[image_id] = row

    output = []
    skipped_length = 0
    skipped_description = 0
    for image_id, row in sorted(best_by_image.items()):
        literal = desc.get(image_id, "")
        if not literal:
            skipped_description += 1
            continue
        funny, weak = str(row["chosen"]), str(row["rejected"])
        if max(relative_length(literal, funny), relative_length(literal, weak)) > args.max_literal_relative_length_difference:
            skipped_length += 1
            continue
        common = {"image": row.get("image") or row.get("image_path"), "image_id": image_id, "prompt": row["prompt"]}
        output.extend(
            [
                {**common, "caption": funny, "label": "funny", "source": "New Yorker higher-score caption", "score": row.get("chosen_score")},
                {**common, "caption": weak, "label": "weak", "source": "New Yorker lower-score caption", "score": row.get("rejected_score")},
                {**common, "caption": literal, "label": "literal", "source": "released GPT-4o canny visual description", "score": None},
            ]
        )
    manifest = {
        "pairs": str(args.pairs),
        "pairs_sha256": sha256(args.pairs),
        "images": len(output) // 3,
        "examples": len(output),
        "examples_by_label": {label: sum(row["label"] == label for row in output) for label in ("funny", "weak", "literal")},
        "skipped_literal_length_mismatch": skipped_length,
        "skipped_missing_description": skipped_description,
        "max_literal_relative_length_difference": args.max_literal_relative_length_difference,
        "warning": "Literal examples are released GPT-4o visual descriptions, not human captions; probe results must be checked for residual style/length shortcuts.",
    }
    write_jsonl(args.output, output)
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
