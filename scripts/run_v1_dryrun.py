#!/usr/bin/env python
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.build_sft_dataset import build_sft_dataset
from src.evaluation.ranker import rank_candidates
from src.inference.generate_candidates import generate_candidates
from src.utils.io import read_jsonl, write_jsonl


def main() -> None:
    demo_dir = Path("data/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    img_dir = demo_dir / "images"
    img_dir.mkdir(exist_ok=True)

    images = []
    captions = []
    for i in range(6):
        path = img_dir / f"img_{i}.png"
        Image.new("RGB", (32, 32), color=(i * 30, 100, 120)).save(path)
        iid = f"img_{i}"
        images.append({"image_id": iid, "image_path": str(path)})
        for j in range(3):
            captions.append({"image_id": iid, "caption": f"Funny caption {j} for {iid} with chaos", "score": j + 1})

    image_csv = demo_dir / "images.csv"
    caption_csv = demo_dir / "captions.csv"
    pd.DataFrame(images).to_csv(image_csv, index=False)
    pd.DataFrame(captions).to_csv(caption_csv, index=False)

    build_sft_dataset(image_csv, caption_csv, Path("data/processed"), "image_id", "image_path", None, "image_id", "caption", "score")
    generate_candidates(Path("data/processed/sft_test.jsonl"), Path("outputs/generations/candidates.jsonl"), num_candidates=10, dry_run=True)

    cand_rows = read_jsonl(Path("outputs/generations/candidates.jsonl"))
    ranked = [rank_candidates(r["image"], r["image_id"], r["candidates"]) for r in cand_rows]
    out_path = Path("outputs/generations/ranked_top5.jsonl")
    write_jsonl(out_path, ranked)
    for row in ranked:
        print(row["image_id"], "TOP-5")
        for x in row["top5"]:
            print(" -", x["caption"], f"({x['total_score']:.2f})")


if __name__ == "__main__":
    main()
