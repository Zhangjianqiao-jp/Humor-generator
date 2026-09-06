#!/usr/bin/env python3
"""Fail-closed check for the current pretrained-7B/bridge-only route."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def check(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text())
    model = config.get("model", {})
    errors: list[str] = []
    data_errors: list[str] = []
    if config.get("experiment", {}).get("route") != "qwen_pretrained_7b_only_bridge":
        errors.append("experiment.route is not qwen_pretrained_7b_only_bridge")
    if model.get("revision") != EXPECTED_REVISION:
        errors.append("Qwen2.5-VL revision is not the pinned local revision")
    for key in ("planner_adapter", "generator_adapter", "adapter"):
        if model.get(key) is not None:
            errors.append(f"{key} must be null for the pretrained-only route")
    for key in ("planner_frozen", "generator_frozen"):
        if model.get(key) is not True:
            errors.append(f"{key} must be true")
    if model.get("bridge_trainable") is not True:
        errors.append("bridge_trainable must be true")
    manifest_value = model.get("manifest")
    if not isinstance(manifest_value, str) or not (ROOT / manifest_value).is_file():
        errors.append("pretrained model manifest is missing")
    else:
        manifest = json.loads((ROOT / manifest_value).read_text())
        if manifest.get("revision") != EXPECTED_REVISION:
            errors.append("pretrained model manifest revision mismatch")
    asset_manifest_path = ROOT / "manifests/homer_official_assets.json"
    if asset_manifest_path.is_file():
        asset_manifest = json.loads(asset_manifest_path.read_text())
        source_hashes = asset_manifest.get("source_code_hashes", {})
        local_sources = {
            "local_official_prompts.py": ROOT / "src/humor_generator_v35/homer/official_prompts.py",
            "local_public_code_pipeline.py": ROOT / "src/humor_generator_v35/homer/public_code_pipeline.py",
            "local_homer_retrieval.py": ROOT / "src/humor_generator_v35/homer/retrieval.py",
        }
        for name, local_source in local_sources.items():
            if not local_source.is_file() or not source_hashes.get(name):
                errors.append(f"source hash is missing: {name}")
            elif hashlib.sha256(local_source.read_bytes()).hexdigest() != source_hashes[name]:
                errors.append(f"source hash mismatch: {name}")
    for relative in (
        "src/humor_generator_v35/homer/official_prompts.py",
        "src/humor_generator_v35/homer/public_code_pipeline.py",
        "manifests/homer_population_v2.json",
        "scripts/verify_homer_evaluation_assets.py",
    ):
        if not (ROOT / relative).is_file():
            errors.append(f"required public-code implementation is missing: {relative}")
    protocol = config.get("protocol", {})
    if protocol.get("summary_required") is not True:
        errors.append("summary_required must be true")
    if protocol.get("retrieval_implementation") != "official_code_compatible_tfidf_wordnet_entity_tree":
        errors.append("pretrained route must use the official-code-compatible retrieval adapter")
    if protocol.get("caption_contract") != ["##Caption", "##Explanation"]:
        errors.append("caption contract must require ##Caption and ##Explanation")
    evaluator = config.get("evaluation", {}).get("exact_comparable_track", {})
    canonical_evaluator = evaluator.get("canonical_evaluator_model")
    if canonical_evaluator != "gpt-5-chat-latest":
        errors.append("canonical HOMER evaluator must be gpt-5-chat-latest")
    evaluator_model = evaluator.get("evaluator_model")
    if not isinstance(evaluator_model, str) or not evaluator_model.strip():
        errors.append("exact comparable evaluator_model must be a non-empty model id")
    evaluator_substitution = evaluator_model != canonical_evaluator
    if evaluator_substitution and evaluator.get("allow_evaluator_substitution") is not True:
        errors.append("evaluator substitution requires allow_evaluator_substitution=true")
    if evaluator_substitution and evaluator.get("substitution_label_required") is not True:
        errors.append("evaluator substitution requires substitution_label_required=true")
    if int(protocol.get("association_successors", 0)) != 3:
        errors.append("main public-code successor count must remain three")
    hia_benchmark = config.get("evaluation", {}).get("official_evaluator_contract", {}).get("hia_original_benchmark", {})
    for key, expected in {
        "available_contests_declared": 358,
        "evaluator_holdout_contests": 91,
        "candidates_per_image": 10,
        "reference_groups": ["#1-10", "#200-209", "#1000-1009", "median"],
        "five_shot_prompt": True,
    }.items():
        if hia_benchmark.get(key) != expected:
            errors.append(f"HIA original benchmark contract mismatch for {key}")
    population_manifest = config.get("data", {}).get("population_manifest")
    if population_manifest != "manifests/homer_population_v2.json":
        errors.append("pretrained route must name the source-aware population manifest")
    data = config.get("data", {})
    if data.get("population_status") != "verified_365_contest_allowlist":
        data_errors.append(
            "population_status is not verified_365_contest_allowlist; official allow-list is pending"
        )
    for key in ("population_rows", "trace_index"):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            data_errors.append(f"data.{key} is not populated; formal route cannot fall back to historical rows")
        elif not (ROOT / value).is_file():
            data_errors.append(f"data.{key} does not exist: {value}")
    if data.get("bridge_training_manifest") is not None:
        value = str(data["bridge_training_manifest"])
        if "latent_bridge_v35" in value:
            data_errors.append("historical latent_bridge_v35 manifest is forbidden on the pretrained route")
    return {
        "status": "pass" if not errors else "fail",
        "config": str(config_path),
        "errors": errors,
        "data_gate": "ready" if not data_errors else "blocked",
        "data_errors": data_errors,
        "evaluator": {
            "model": evaluator_model,
            "canonical_model": canonical_evaluator,
            "substitution": evaluator_substitution,
            "substitution_label_required": evaluator.get("substitution_label_required") is True,
        },
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/homer_public_code_pretrained_7b.yaml",
    )
    parser.add_argument(
        "--require-data-ready",
        action="store_true",
        help="Exit non-zero unless the verified population rows and current trace index exist.",
    )
    args = parser.parse_args()
    report = check(args.config.resolve())
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" or (args.require_data_ready and report["data_gate"] != "ready"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
