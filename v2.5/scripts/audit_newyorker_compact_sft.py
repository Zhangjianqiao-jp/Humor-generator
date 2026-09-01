#!/usr/bin/env python
"""Independently audit the clean New Yorker compact SFT corpus.

This intentionally does not import either dataset builder.  It reconstructs
the eligible captions from the pinned raw CSV files and verifies that the
materialised SFT files contain exactly the per-cartoon top three percent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from PIL import Image


SPLITS = ("train", "validation", "test")
SPACE_RE = re.compile(r"\s+")
FIRST_SENTENCE_RE = re.compile(r"^.*?[.!?](?=\s|$)")
URL_ONLY_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
SCHEMA = ("ANCHOR", "CONTRAST", "ANGLE")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def normalise(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip()


def first_sentence(value: str) -> str:
    value = normalise(value)
    match = FIRST_SENTENCE_RE.match(value)
    return match.group(0) if match else value


def message_text(row: dict[str, Any], role: str) -> str:
    values: list[str] = []
    for message in row.get("messages", []):
        if message.get("role") != role:
            continue
        for item in message.get("content", []):
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                values.append(item["text"])
    if len(values) != 1:
        raise ValueError(f"{row.get('image_id')}: expected one {role} text, got {len(values)}")
    return values[0]


def eligible_source_rows(path: Path, max_chars: int) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            caption = normalise(source.get("caption") or "")
            if not (5 <= len(caption) <= max_chars) or URL_ONLY_RE.fullmatch(caption):
                continue
            try:
                rank = int(source["rank"])
                score = float(source["mean"])
                votes = int(source["votes"])
                funny = int(source["funny"])
            except (KeyError, TypeError, ValueError):
                continue
            key = caption.casefold()
            if key in seen:
                continue
            seen.add(key)
            eligible.append(
                {"caption": caption, "rank": rank, "score": score, "votes": votes, "funny": funny}
            )
    eligible.sort(key=lambda row: (row["rank"], row["caption"].lower()))
    return eligible


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_image(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing image: {path}")
    with Image.open(path) as image:
        image.verify()


def audit(raw_dir: Path, top_dir: Path, compact_dir: Path) -> dict[str, Any]:
    top_manifest = json.loads((top_dir / "manifest.json").read_text(encoding="utf-8"))
    compact_manifest = json.loads((compact_dir / "manifest.json").read_text(encoding="utf-8"))
    fraction = float(top_manifest["top_fraction"])
    max_chars = int(top_manifest["max_caption_chars"])
    if not math.isclose(fraction, 0.03):
        raise ValueError(f"expected top_fraction=0.03, got {fraction}")
    if top_manifest.get("source_revision") != "1cd70477b6a99a473690a25a2fed359f75184c64":
        raise ValueError("source revision is not the reviewed pinned New Yorker snapshot")
    if "CC BY-NC 4.0" not in str(top_manifest.get("license_and_use_restriction")):
        raise ValueError("missing non-commercial license restriction")

    descriptions: dict[int, dict[str, Any]] = {}
    declared_split: dict[int, str] = {}
    for split in SPLITS:
        for row in read_jsonl(raw_dir / "gpt4o_description" / f"{split}.jsonl"):
            contest = int(row["contest_number"])
            if contest in descriptions:
                raise ValueError(f"contest {contest} occurs in multiple description splits")
            descriptions[contest] = row
            declared_split[contest] = split

    image_sets: dict[str, set[str]] = {}
    total_source_eligible = 0
    checked_images: set[Path] = set()
    file_hashes: dict[str, str] = {}
    rows_report: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        top_path = top_dir / f"sft_{split}.jsonl"
        planner_path = compact_dir / f"planner_{split}.jsonl"
        caption_path = compact_dir / f"caption_{split}.jsonl"
        top_rows = read_jsonl(top_path)
        planner_rows = read_jsonl(planner_path)
        caption_rows = read_jsonl(caption_path)
        if len(top_rows) != len(caption_rows):
            raise ValueError(f"{split}: compact captions do not preserve all selected top rows")

        top_by_contest: dict[int, list[dict[str, Any]]] = {}
        for row in top_rows:
            contest = int(row["meta"]["contest_number"])
            top_by_contest.setdefault(contest, []).append(row)
        if set(top_by_contest) != {int(row["meta"]["contest_number"]) for row in planner_rows}:
            raise ValueError(f"{split}: planner/caption image sets differ")

        plans: dict[str, str] = {}
        for row in planner_rows:
            contest = int(row["meta"]["contest_number"])
            if declared_split.get(contest) != split:
                raise ValueError(f"contest {contest}: split assignment changed")
            image_id = str(row["image_id"])
            if image_id in plans:
                raise ValueError(f"{split}: duplicate planner image {image_id}")
            plan = message_text(row, "assistant")
            lines = plan.splitlines()
            if len(lines) != 3 or tuple(line.split(":", 1)[0] for line in lines) != SCHEMA:
                raise ValueError(f"{image_id}: invalid compact planner schema")
            source_desc = descriptions[contest]
            if lines[0] != f"ANCHOR: {first_sentence(str(source_desc['canny']))}":
                raise ValueError(f"{image_id}: ANCHOR is not grounded in the source description")
            if lines[1] != f"CONTRAST: {first_sentence(str(source_desc['uncanny']))}":
                raise ValueError(f"{image_id}: CONTRAST is not grounded in the source description")
            if not lines[2].split(":", 1)[1].strip():
                raise ValueError(f"{image_id}: empty ANGLE")
            plans[image_id] = plan

        expected_caption_keys: list[tuple[int, str, int]] = []
        for contest, selected_rows in sorted(top_by_contest.items()):
            source_path = raw_dir / "ranking" / "source" / f"{contest}.csv"
            eligible = eligible_source_rows(source_path, max_chars)
            total_source_eligible += len(eligible)
            keep = max(1, math.floor(len(eligible) * fraction)) if eligible else 0
            expected = eligible[:keep]
            actual = [
                {
                    "caption": message_text(row, "assistant"),
                    "rank": int(row["meta"]["rank"]),
                    "score": float(row["meta"]["score"]),
                    "votes": int(row["meta"]["votes"]),
                    "funny": int(row["meta"]["funny_votes"]),
                }
                for row in selected_rows
            ]
            if actual != expected:
                raise ValueError(f"contest {contest}: selected rows are not exactly the source top 3%")
            expected_caption_keys.extend((contest, item["caption"].casefold(), item["rank"]) for item in expected)

        actual_caption_keys: list[tuple[int, str, int]] = []
        for top_row, row in zip(top_rows, caption_rows, strict=True):
            if message_text(top_row, "assistant") != message_text(row, "assistant"):
                raise ValueError(f"{split}: compact transformation changed a target caption")
            contest = int(row["meta"]["contest_number"])
            image_id = str(row["image_id"])
            caption = message_text(row, "assistant")
            prompt = message_text(row, "user")
            plan = plans[image_id]
            if not prompt.endswith(f"Humor plan:\n{plan}"):
                raise ValueError(f"{image_id}: caption prompt does not contain the matching plan")
            if caption.casefold() in prompt.casefold():
                raise ValueError(f"{image_id}: gold caption leaks into the prompt")
            if row["meta"].get("task") != "compact_conditioned_humor_captioning":
                raise ValueError(f"{image_id}: wrong task label")
            actual_caption_keys.append((contest, caption.casefold(), int(row["meta"]["rank"])))
            image = Path(str(row["image"]))
            if image not in checked_images:
                verify_image(image)
                checked_images.add(image)
        if sorted(actual_caption_keys) != sorted(expected_caption_keys):
            raise ValueError(f"{split}: materialised caption keys differ from raw-source selection")
        if len(actual_caption_keys) != len(set(actual_caption_keys)):
            raise ValueError(f"{split}: duplicate image-caption-rank rows")

        image_sets[split] = set(plans)
        rows_report[split] = {"planner": len(planner_rows), "caption": len(caption_rows)}
        for path in (top_path, planner_path, caption_path):
            file_hashes[str(path)] = sha256(path)

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = image_sets[left] & image_sets[right]
            if overlap:
                raise ValueError(f"image leakage between {left} and {right}: {sorted(overlap)[:3]}")

    if rows_report != {
        split: {
            "planner": int(compact_manifest["planner_rows_by_split"][split]),
            "caption": int(compact_manifest["caption_rows_by_split"][split]),
        }
        for split in SPLITS
    }:
        raise ValueError("manifest counts do not match materialised rows")

    return {
        "status": "pass",
        "source_revision": top_manifest["source_revision"],
        "selection": "exact per-cartoon top 3% reconstructed from raw ranking CSV",
        "license": top_manifest["license_and_use_restriction"],
        "split_image_disjoint": True,
        "gold_caption_prompt_leakage": 0,
        "verified_unique_images": len(checked_images),
        "source_eligible_rows_recomputed": total_source_eligible,
        "rows": rows_report,
        "sha256": file_hashes,
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/newyorker_caption_ranking"))
    parser.add_argument("--top-dir", type=Path, default=Path("data/processed/newyorker_top3pct_sft"))
    parser.add_argument("--compact-dir", type=Path, default=Path("data/processed/newyorker_compact_sft_v2"))
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = audit(args.raw_dir, args.top_dir, args.compact_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
