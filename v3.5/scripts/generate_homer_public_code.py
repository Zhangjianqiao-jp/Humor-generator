#!/usr/bin/env python3
"""Generate the current pretrained-7B HOMER protocol on explicit rows.

This entry point is intentionally separate from ``generate_formal_baseline``.
It never loads the historical SFT adapters and never collapses rows with
``first_rows_by_cluster``.  It is resumable, but it will refuse to run until
the pretrained route and the dataset manifest pass their checks.
"""
from __future__ import annotations

from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.homer.contracts import parse_conflicts
from humor_generator_v35.homer.public_code_pipeline import HomerPublicCodePipeline
from humor_generator_v35.homer.retrieval import OfficialCodeRetrieval
from humor_generator_v35.homer.omega import parse as parse_omega
from humor_generator_v35.qwen_backend import QwenBackend
from check_pretrained_route import check as check_pretrained_route


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_official_summary_retriever(root: Path):
    manifest = json.loads((root / "manifests/homer_official_assets.json").read_text())
    corpus = root / manifest["joke_corpus"]["local_path"]
    if sha256(corpus) != manifest["joke_corpus"]["sha256"]:
        raise RuntimeError("HOMER joke corpus hash mismatch")
    with corpus.open(encoding="utf-8", newline="") as handle:
        jokes = [row["Joke"] for row in csv.DictReader(handle)]
    import nltk
    nltk.data.path.insert(0, str(root / "artifacts/nltk_data"))
    # OfficialCodeRetrieval mirrors the pinned ``imaginator.py`` formulas and
    # entity-tree construction.  The older v3.5 augmenter remains available
    # for historical/latent comparisons, but is not the formal route.
    augmenter = OfficialCodeRetrieval(jokes)

    def retrieve(summary: dict[str, Any], description: str, conflict_text: str) -> Mapping[str, Any]:
        # ``OfficialCodeRetrieval`` expects the raw summary mapping, then
        # reproduces the public ``retrieval_humor_base`` entity-tree output.
        parse_conflicts(conflict_text)  # validate the same conflict contract before retrieval
        return augmenter.retrieve(summary, description=description, conflicts=conflict_text)

    return retrieve


def load_rows(
    dataset_root: Path,
    split: str,
    *,
    require_standard_description: bool,
) -> list[dict[str, Any]]:
    path = dataset_root / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty dataset split: {path}")
    required = {"row_id", "cluster_id", "dataset", "contest_number", "image", "image_sha256"}
    if require_standard_description:
        required.add("standard_description")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"dataset rows missing required fields: {missing}")
    for row in rows:
        image_value = row.get("image")
        image = Path(str(image_value))
        if not image.is_absolute():
            image = (ROOT / image).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"row {row.get('row_id')} image is missing: {image}")
        expected_hash = str(row.get("image_sha256", ""))
        if expected_hash and sha256(image) != expected_hash:
            raise ValueError(f"row {row.get('row_id')} image_sha256 mismatch: {image}")
        if require_standard_description and not str(row.get("standard_description", "")).strip():
            raise ValueError(f"row {row.get('row_id')} has an empty standard_description")
    return rows


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/homer_public_code_pretrained_7b.yaml")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/processed/homer_pretrained_7b_public_release_362",
        help="A source-aware population directory; the historical representative-row data is refused.",
    )
    parser.add_argument(
        "--allow-historical-pilot",
        action="store_true",
        help=(
            "Explicitly allow the superseded representative-row manifest for a parser/"
            "engineering pilot. This flag is forbidden for HOMER population claims."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "internal_test", "official_hia_unseen_test", "official_hia_seen_diagnostic"],
        default="validation",
    )
    parser.add_argument("--mode", choices=["standard_replay", "online_description"], default="standard_replay")
    parser.add_argument(
        "--omega",
        type=str,
        default=None,
        metavar="NARRATIVE_STRATEGY|LINGUISTIC_STYLE",
        help=(
            "Enable the explicitly named project Ω extension, for example "
            "setup_punchline|pun_wordplay. Omit this flag for the HOMER "
            "public-code baseline."
        ),
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260906])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-retrieval", action="store_true", help="Only for an explicitly non-scientific parser smoke")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    route_report = check_pretrained_route(args.config if args.config.name == "homer_public_code_pretrained_7b.yaml" else ROOT / "configs/homer_public_code_pretrained_7b.yaml")
    if route_report["status"] != "pass":
        raise RuntimeError(f"pretrained route check failed: {route_report['errors']}")
    historical_dataset = args.dataset.resolve() == (ROOT / "data/processed/latent_bridge_v35").resolve()
    if route_report.get("data_gate") != "ready" and not args.allow_historical_pilot:
        raise RuntimeError(
            "pretrained route data gate is blocked: "
            f"{route_report.get('data_errors', [])}; resolve the source-aware "
            "population/trace provenance before formal generation"
        )
    if historical_dataset and not args.allow_historical_pilot:
        raise RuntimeError(
            "the current pretrained-7B HOMER route refuses the historical "
            "latent_bridge_v35 representative-row dataset; build and pass a "
            "source-aware population split explicitly. Use --allow-historical-pilot "
            "only for a non-scientific parser/engineering smoke."
        )
    if args.allow_historical_pilot and args.max_rows is None:
        raise RuntimeError(
            "--allow-historical-pilot requires --max-rows so a historical run cannot "
            "silently become a full population claim"
        )
    if args.allow_historical_pilot and not historical_dataset:
        raise RuntimeError(
            "--allow-historical-pilot is only valid for the superseded "
            "data/processed/latent_bridge_v35 dataset"
        )
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")
    omega_cfg = config.get("protocol", {}).get("omega", {})
    omega_spec = None
    if args.omega is not None:
        if omega_cfg.get("enabled") is not True:
            raise RuntimeError("the selected config does not enable the Ω extension")
        omega_spec = parse_omega(args.omega)
        if omega_spec.narrative_strategy not in omega_cfg.get("narrative_strategies", []):
            raise ValueError("omega narrative strategy is not in the versioned config vocabulary")
        if omega_spec.linguistic_style not in omega_cfg.get("linguistic_styles", []):
            raise ValueError("omega linguistic style is not in the versioned config vocabulary")
    omega_record = None
    omega_sha256 = None
    if omega_spec is not None:
        omega_record = {
            "variant": omega_cfg.get("variant_name"),
            "narrative_strategy": omega_spec.narrative_strategy,
            "linguistic_style": omega_spec.linguistic_style,
            "prompt_text": omega_spec.prompt_text(),
        }
        omega_sha256 = hashlib.sha256(
            json.dumps(omega_record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
    rows = load_rows(
        args.dataset,
        args.split,
        require_standard_description=args.mode == "standard_replay",
    )
    if args.max_rows is not None:
        if args.max_rows < 1:
            raise ValueError("max-rows must be positive")
        rows = rows[: args.max_rows]

    model = config["model"]
    backend = QwenBackend.load(
        model["name"],
        revision=model["revision"],
        adapter=None,
        load_in_4bit=True,
        min_visual_tokens=256,
        max_visual_tokens=1280,
    )
    retriever = None if args.without_retrieval else build_official_summary_retriever(ROOT)
    pipeline = HomerPublicCodePipeline(
        backend,
        retriever=retriever,
        provider="qwen",
        require_retrieval=not args.without_retrieval,
        successors=int(config["protocol"]["association_successors"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, int]] = set()
    if args.output.is_file():
        completed = {
            (str(item["row_id"]), int(item["seed"]))
            for item in read_jsonl(args.output)
            if "row_id" in item and "seed" in item
        }
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            for seed in args.seeds:
                key = (str(row["row_id"]), seed)
                if key in completed:
                    continue
                result = pipeline.run(
                    image=str((ROOT / row["image"]).resolve()),
                    seed=seed,
                    standard_description=(
                        None
                        if args.mode == "online_description"
                        else str(row["standard_description"])
                    ),
                    omega=omega_spec.prompt_text() if omega_spec is not None else None,
                )
                record = {
                    "schema_version": 1,
                    "route": "pretrained_7b_bridge_only",
                    "split": args.split,
                    "row_id": row["row_id"],
                    "cluster_id": row["cluster_id"],
                    "contest_number": row.get("contest_number"),
                    "image": row["image"],
                    "image_sha256": row.get("image_sha256"),
                    "seed": seed,
                    "mode": args.mode,
                    "description": result.description,
                    "conflict_scripts": result.conflict_scripts,
                    "global_imagination": result.global_imagination,
                    "local_imagination": result.local_imagination,
                    "summary_imagination": result.summary_imagination,
                    "retrieved_imagination": result.retrieved_imagination,
                    "selected_conflict": result.selected_conflict,
                    "selected_entities": list(result.selected_entities),
                    "selected_paths": {key: list(value) for key, value in result.selected_paths.items()},
                    "caption": result.caption,
                    "explanation": result.explanation,
                    "raw_caption": result.raw_caption,
                    "omega": omega_record,
                    "request_count": result.request_count,
                    "call_log": list(result.call_log),
                    "provenance": {
                        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip(),
                        "config_sha256": sha256(args.config.resolve()),
                        "prompt_sha256": sha256(ROOT / "src/humor_generator_v35/homer/official_prompts.py"),
                        "omega_variant": omega_cfg.get("variant_name") if omega_spec is not None else None,
                        "omega_sha256": omega_sha256,
                        "model_manifest_sha256": sha256(ROOT / model["manifest"]),
                        "retrieval": "official_code_compatible_tfidf_wordnet_entity_tree" if retriever is not None else "disabled_parser_smoke_only",
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed.add(key)
    manifest = {
        "schema_version": 1,
        "route": "pretrained_7b_bridge_only",
        "split": args.split,
        "mode": args.mode,
        "rows": len(rows),
        "seeds": args.seeds,
        "retrieval": retriever is not None,
        "omega": (
            {"variant": omega_cfg.get("variant_name"), "value": args.omega, "sha256": omega_sha256}
            if omega_spec is not None
            else None
        ),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
