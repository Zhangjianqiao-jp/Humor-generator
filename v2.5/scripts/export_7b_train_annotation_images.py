#!/usr/bin/env python3
"""Export the current 7B planner training images for blind human annotation."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/processed/newyorker_compact_viewpoint_sft/train.jsonl")
DEFAULT_OUTPUT = Path("data/annotations/nycc_planner_v2/packets/train_image_only")


def resolve_from_project(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []  # type: List[Dict[str, Any]]
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("image") or not row.get("image_id"):
                raise ValueError(f"{path}:{line_number} lacks image or image_id")
            rows.append(row)
    return rows


def export(input_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    rows = read_rows(input_path)
    image_ids = [str(row["image_id"]) for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Training JSONL contains duplicate image_id values")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = []  # type: List[Dict[str, Any]]

    for index, row in enumerate(rows, start=1):
        source_value = Path(str(row["image"]))
        source = resolve_from_project(source_value).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)

        image_id = str(row["image_id"])
        filename = f"{index:03d}_{image_id}{source.suffix.lower()}"
        destination = images_dir / filename
        source_hash = sha256(source)

        if destination.exists():
            if sha256(destination) != source_hash:
                raise ValueError(f"Existing export differs from source: {destination}")
        else:
            shutil.copy2(source, destination)

        manifest.append(
            {
                "index": index,
                "image_id": image_id,
                "filename": f"images/{filename}",
                "source_image": str(source_value),
                "sha256": source_hash,
            }
        )

    manifest_jsonl = output_dir / "manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest_csv = output_dir / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = resolve_from_project(args.input)
    output_dir = resolve_from_project(args.output_dir)
    manifest = export(input_path, output_dir)
    print(f"exported_images={len(manifest)}")
    print(f"output_dir={output_dir}")
    print(f"manifest={output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
