#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.humor_context import QwenVLHumorContextExtractor, STRUCTURED_HUMOR_PROMPTS
from src.training.sft_dataset import extract_image_path, resolve_image_path
from src.utils.io import read_jsonl

EXTRACTION_MODES = ("visual-facts", "structured-humor", "both")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def row_satisfies_mode(row: dict[str, Any], mode: str) -> bool:
    if row.get("failed"):
        return False
    if mode in ("visual-facts", "both") and not row.get("visual_facts"):
        return False
    if mode in ("structured-humor", "both") and "structured_humor" not in row:
        return False
    return True


def read_existing_keys(path: Path, mode: str) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row_satisfies_mode(row, mode):
                continue
            for key in keys_for_row(row):
                keys.add(key)
    return keys


def keys_for_row(row: dict[str, Any]) -> list[str]:
    keys = []
    image_id = row.get("image_id")
    if image_id:
        keys.append(f"id:{image_id}")
    image = row.get("image")
    if image:
        keys.append(f"image:{image}")
        keys.append(f"stem:{Path(str(image)).stem}")
    return keys


def resolve_row_image(row: dict[str, Any], image_root: Path | None) -> Path:
    raw_image = extract_image_path(row)
    if not raw_image:
        raise ValueError("row does not contain an image path")
    return resolve_image_path(str(raw_image), image_root)


def structured_humor_type(row: dict[str, Any]) -> str:
    structured = row.get("structured_humor") or {}
    mechanism = structured.get("humor_mechanism") if isinstance(structured, dict) else {}
    if not isinstance(mechanism, dict):
        return "none"
    return str(mechanism.get("type") or "none")


def structured_humor_useful(row: dict[str, Any]) -> bool:
    structured = row.get("structured_humor") or {}
    guidance = structured.get("generator_guidance") if isinstance(structured, dict) else {}
    return bool(isinstance(guidance, dict) and guidance.get("useful"))


def run_extract(
    config_path: Path,
    input_jsonl: Path | None,
    output_jsonl: Path | None,
    limit: int | None,
    overwrite: bool,
    skip_existing: bool,
    mode: str | None,
    structured_prompt_version: str | None,
) -> None:
    config = load_config(config_path)
    extractor_config = config["extractor"]
    data_config = config.get("data", {})
    output_config = config.get("output", {})

    extraction_mode = mode or str(extractor_config.get("mode", "visual-facts"))
    if extraction_mode not in EXTRACTION_MODES:
        raise ValueError(f"Unknown extraction mode: {extraction_mode}. Choices: {', '.join(EXTRACTION_MODES)}")

    prompt_version = structured_prompt_version or str(
        extractor_config.get("structured_humor_prompt_version", "structured-v1")
    )
    if prompt_version not in STRUCTURED_HUMOR_PROMPTS:
        choices = ", ".join(sorted(STRUCTURED_HUMOR_PROMPTS))
        raise ValueError(f"Unknown structured prompt version: {prompt_version}. Choices: {choices}")

    input_path = input_jsonl or Path(data_config.get("input_jsonl", "data/processed/sft_test.jsonl"))
    output_path = output_jsonl or Path(output_config.get("context_jsonl", "outputs/analysis/vlm_visual_facts.jsonl"))
    if limit is None:
        limit = data_config.get("limit")
    image_root = data_config.get("image_root")
    image_root_path = None if image_root in (None, "", "null") else Path(image_root)

    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[: int(limit)]

    if overwrite and output_path.exists():
        output_path.unlink()
    existing_keys = read_existing_keys(output_path, extraction_mode) if skip_existing else set()

    extractor = QwenVLHumorContextExtractor(
        model_name=str(extractor_config["model_name"]),
        device_map=str(extractor_config.get("device_map", "auto")),
        torch_dtype=str(extractor_config.get("torch_dtype", "auto")),
        trust_remote_code=bool(extractor_config.get("trust_remote_code", True)),
    )

    failures = 0
    skipped_existing = 0
    skipped_missing = 0
    structured_parse_errors = 0
    structured_useful = 0
    structured_non_none = 0
    type_counts: dict[str, int] = {}
    for index, row in enumerate(tqdm(rows, desc=f"extracting {extraction_mode}", dynamic_ncols=True)):
        row_keys = keys_for_row(row)
        if skip_existing and any(key in existing_keys for key in row_keys):
            skipped_existing += 1
            continue

        try:
            image_path = resolve_row_image(row, image_root_path)
            if not image_path.exists():
                skipped_missing += 1
                print(f"[extract-context] missing image row={index}: {image_path}")
                continue
            context = extractor.analyze_image(
                image_path=image_path,
                description_max_new_tokens=int(extractor_config.get("description_max_new_tokens", 96)),
                humor_max_new_tokens=int(
                    extractor_config.get("visual_facts_max_new_tokens", extractor_config.get("humor_max_new_tokens", 384))
                ),
                include_visual_facts=extraction_mode in ("visual-facts", "both"),
                include_structured_humor=extraction_mode in ("structured-humor", "both"),
                structured_humor_max_new_tokens=int(extractor_config.get("structured_humor_max_new_tokens", 768)),
                structured_humor_prompt_version=prompt_version,
                structured_humor_temperature=float(extractor_config.get("structured_humor_temperature", 0.0)),
                visual_facts_temperature=float(extractor_config.get("visual_facts_temperature", 0.0)),
            )
            output_row = {
                "image": str(image_path),
                "image_id": row.get("image_id") or image_path.stem,
                "source_index": index,
                "extraction_mode": extraction_mode,
                **context,
            }
        except Exception as exc:
            failures += 1
            output_row = {
                "image": row.get("image") or extract_image_path(row),
                "image_id": row.get("image_id"),
                "source_index": index,
                "extraction_mode": extraction_mode,
                "failed": True,
                "error": str(exc),
            }
            print(f"[extract-context] failed row={index}: {exc}")

        append_jsonl(output_path, output_row)
        if not output_row.get("failed"):
            for key in keys_for_row(output_row):
                existing_keys.add(key)
            if "structured_humor" in output_row:
                if output_row.get("structured_humor_parse_error"):
                    structured_parse_errors += 1
                if structured_humor_useful(output_row):
                    structured_useful += 1
                mechanism_type = structured_humor_type(output_row)
                type_counts[mechanism_type] = type_counts.get(mechanism_type, 0) + 1
                if mechanism_type != "none":
                    structured_non_none += 1

    print(f"[extract-context] saved to {output_path}")
    print(
        "[extract-context] "
        f"mode={extraction_mode} rows={len(rows)} skipped_existing={skipped_existing} "
        f"skipped_missing={skipped_missing} failures={failures}"
    )
    if extraction_mode in ("structured-humor", "both"):
        print(
            "[extract-context] "
            f"structured_prompt={prompt_version} temperature={float(extractor_config.get('structured_humor_temperature', 0.0))} "
            f"structured_non_none={structured_non_none} structured_useful={structured_useful} "
            f"structured_parse_errors={structured_parse_errors} type_counts={type_counts}"
        )


def main() -> None:
    parser = ArgumentParser(description="Extract conservative VLM visual facts and structured humor guidance.")
    parser.add_argument("--config", type=Path, default=Path("configs/vlm_guided_generation.yaml"))
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--mode", choices=EXTRACTION_MODES, default=None)
    parser.add_argument("--structured-prompt-version", choices=sorted(STRUCTURED_HUMOR_PROMPTS), default=None)
    args = parser.parse_args()
    run_extract(
        config_path=args.config,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        limit=args.limit,
        overwrite=args.overwrite,
        skip_existing=not args.no_skip_existing,
        mode=args.mode,
        structured_prompt_version=args.structured_prompt_version,
    )


if __name__ == "__main__":
    main()
