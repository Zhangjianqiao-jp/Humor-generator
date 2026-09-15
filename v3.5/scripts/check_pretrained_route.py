#!/usr/bin/env python3
"""Fail-closed check for the current pretrained-7B/bridge-only route."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.training.public_bridge import load_context_index


EXPECTED_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATA_VERSION = "homer_pretrained_7b_public_release_362"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


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
        "manifests/homer_population_public_release_362.json",
        "scripts/verify_homer_public_release_362.py",
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
    if population_manifest != "manifests/homer_population_public_release_362.json":
        errors.append("pretrained route must name the pinned public-release population manifest")
    population_manifest_payload: dict[str, object] = {}
    if isinstance(population_manifest, str) and (ROOT / population_manifest).is_file():
        population_manifest_payload = json.loads((ROOT / population_manifest).read_text())
        if population_manifest_payload.get("population_status") != "verified_public_release_362":
            errors.append("public-release population manifest status is not verified_public_release_362")
        gate = population_manifest_payload.get("formal_route_gate", {})
        if gate.get("verified_public_release_allowlist") is not True:
            errors.append("public-release manifest does not verify its machine-readable allow-list")
        if gate.get("verified_365_contest_allowlist") is not False:
            errors.append("canonical 365 gate must remain explicitly false")
    data = config.get("data", {})
    if data.get("population_status") != "verified_public_release_362":
        data_errors.append(
            "population_status is not verified_public_release_362"
        )
    for key in ("population_rows",):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            data_errors.append(f"data.{key} is not populated; public-release route cannot fall back to historical rows")
        elif not (ROOT / value).is_file():
            data_errors.append(f"data.{key} does not exist: {value}")
    trace_errors: list[str] = []
    trace_value = data.get("trace_index")
    if not isinstance(trace_value, str) or not trace_value:
        trace_errors.append(
            "data.trace_index is pending; public-code generation is allowed, bridge training is not"
        )
    else:
        trace_path = ROOT / trace_value
        if not trace_path.is_file():
            trace_errors.append(f"data.trace_index does not exist: {trace_value}")
        else:
            trace_manifest_path = trace_path.parent / "manifest.json"
            if not trace_manifest_path.is_file():
                trace_errors.append(
                    "pretrained trace manifest is missing; index cannot be trusted"
                )
            else:
                try:
                    trace_manifest = json.loads(trace_manifest_path.read_text())
                except json.JSONDecodeError as exc:
                    trace_errors.append(f"pretrained trace manifest is invalid JSON: {exc}")
                else:
                    expected_input = data.get("trace_input_manifest")
                    expected_input_hash = (
                        hashlib.sha256((ROOT / expected_input).read_bytes()).hexdigest()
                        if isinstance(expected_input, str) and (ROOT / expected_input).is_file()
                        else None
                    )
                    if trace_manifest.get("status") != "pass":
                        trace_errors.append(
                            "pretrained trace manifest is not complete (status != pass)"
                        )
                    if trace_manifest.get("records_in_index") != 362:
                        trace_errors.append("pretrained trace index must contain 362 records")
                    if trace_manifest.get("failure_records") != 0:
                        trace_errors.append("pretrained trace cache reports generation failures")
                    if trace_manifest.get("data_version") != "homer_pretrained_7b_public_release_362":
                        trace_errors.append("pretrained trace data version mismatch")
                    if trace_manifest.get("revision") != EXPECTED_REVISION:
                        trace_errors.append("pretrained trace Planner revision mismatch")
                    if trace_manifest.get("adapter") is not None:
                        trace_errors.append("pretrained trace manifest must declare adapter=null")
                    if expected_input_hash is None or trace_manifest.get(
                        "trace_input_manifest_sha256"
                    ) != expected_input_hash:
                        trace_errors.append("pretrained trace input manifest hash mismatch")
    trace_input = data.get("trace_input_manifest")
    if not isinstance(trace_input, str) or not trace_input or not (ROOT / trace_input).is_file():
        trace_errors.append("data.trace_input_manifest is missing; current pretrained traces cannot be generated")
    if data.get("bridge_training_manifest") is not None:
        value = str(data["bridge_training_manifest"])
        if "latent_bridge_v35" in value:
            data_errors.append("historical latent_bridge_v35 manifest is forbidden on the pretrained route")

    # Context is deliberately separate from the online population/trace
    # gates.  The public text baseline can run without this post-trace cache,
    # while bridge training must fail closed until all 362
    # summary/retrieval/selection records have been validated.
    context_errors: list[str] = []
    context_value = data.get("bridge_context_index")
    context_records = 0
    context_index_hash: str | None = None
    contexts: dict[str, object] = {}
    if not isinstance(context_value, str) or not context_value:
        context_errors.append(
            "data.bridge_context_index is missing; bridge context has not been sealed"
        )
    else:
        context_path = ROOT / context_value
        if not context_path.is_file():
            context_errors.append(f"data.bridge_context_index does not exist: {context_value}")
        else:
            context_index_hash = hashlib.sha256(context_path.read_bytes()).hexdigest()
            try:
                contexts = load_context_index(context_path)
                context_records = len(contexts)
            except Exception as exc:
                context_errors.append(f"pretrained HOMER context index is invalid: {exc}")
            manifest_path = context_path.parent / "manifest.json"
            if not manifest_path.is_file():
                context_errors.append("pretrained HOMER context manifest is missing")
            else:
                try:
                    context_manifest = json.loads(manifest_path.read_text())
                except json.JSONDecodeError as exc:
                    context_manifest = {}
                    context_errors.append(
                        f"pretrained HOMER context manifest is invalid JSON: {exc}"
                    )
                if context_manifest.get("status") != "pass":
                    context_errors.append(
                        "pretrained HOMER context manifest is not complete (status != pass)"
                    )
                if context_manifest.get("records_in_index") != 362:
                    context_errors.append(
                        "pretrained HOMER context index must contain 362 records"
                    )
                if context_manifest.get("failure_records") != 0:
                    context_errors.append(
                        "pretrained HOMER context cache reports generation failures"
                    )
                if context_manifest.get("data_version") != DATA_VERSION:
                    context_errors.append("pretrained HOMER context data version mismatch")
                if (
                    context_manifest.get("model") != MODEL_NAME
                    or context_manifest.get("revision") != EXPECTED_REVISION
                ):
                    context_errors.append("pretrained HOMER context Planner identity mismatch")
                if context_manifest.get("adapter") is not None:
                    context_errors.append("pretrained HOMER context must declare adapter=null")
                trace_path = ROOT / str(data.get("trace_index", ""))
                input_path = ROOT / str(data.get("trace_input_manifest", ""))
                if trace_path.is_file() and context_manifest.get(
                    "trace_index_sha256"
                ) != hashlib.sha256(trace_path.read_bytes()).hexdigest():
                    context_errors.append("pretrained HOMER context trace-index hash mismatch")
                if input_path.is_file() and context_manifest.get(
                    "trace_input_manifest_sha256"
                ) != hashlib.sha256(input_path.read_bytes()).hexdigest():
                    context_errors.append("pretrained HOMER context trace-input hash mismatch")
            if context_records != 362:
                context_errors.append(
                    f"pretrained HOMER context index has {context_records} records; expected 362"
                )
            if trace_value and (ROOT / trace_value).is_file() and contexts:
                try:
                    trace_clusters = {
                        str(json.loads(line).get("cluster_id"))
                        for line in (ROOT / trace_value).read_text().splitlines()
                        if line.strip()
                    }
                except (json.JSONDecodeError, OSError):
                    trace_clusters = set()
                if trace_clusters and trace_clusters != set(contexts):
                    context_errors.append(
                        "pretrained HOMER context and trace cluster sets differ"
                    )

    # A complete context cache is still not a bridge dataset.  Keep the final
    # bridge gate blocked until the source-caption training view is present
    # and its own verified manifest points at this exact context hash.
    bridge_errors = [*data_errors, *trace_errors, *context_errors]
    bridge_manifest_value = data.get("bridge_training_manifest")
    if not isinstance(bridge_manifest_value, str) or not bridge_manifest_value:
        bridge_errors.append(
            "data.bridge_training_manifest is missing; bridge input view is not sealed"
        )
    else:
        bridge_manifest_path = ROOT / bridge_manifest_value
        if not bridge_manifest_path.is_file():
            bridge_errors.append(
                f"data.bridge_training_manifest does not exist: {bridge_manifest_value}"
            )
        else:
            try:
                bridge_manifest = json.loads(bridge_manifest_path.read_text())
            except json.JSONDecodeError as exc:
                bridge_manifest = {}
                bridge_errors.append(f"bridge training manifest is invalid JSON: {exc}")
            if bridge_manifest.get("status") != "verified_current_pretrained_bridge_training_view":
                bridge_errors.append(
                    "bridge training manifest is not the verified current pretrained view"
                )
            if bridge_manifest.get("data_version") != DATA_VERSION:
                bridge_errors.append("bridge training manifest data version mismatch")
            if context_index_hash and bridge_manifest.get(
                "context_index_sha256"
            ) != context_index_hash:
                bridge_errors.append("bridge training manifest context-index hash mismatch")
    return {
        "status": "pass" if not errors else "fail",
        "config": str(config_path),
        "errors": errors,
        # Population readiness is sufficient for the online public-code
        # baseline.  Cached hidden-state traces are a separate bridge gate.
        "data_gate": "ready" if not data_errors else "blocked",
        "data_errors": data_errors,
        "trace_gate": "ready" if not trace_errors else "blocked",
        "trace_errors": trace_errors,
        "context_gate": "ready" if not context_errors else "blocked",
        "context_errors": context_errors,
        "context_records": context_records,
        "context_index_sha256": context_index_hash,
        "bridge_data_gate": "ready" if not bridge_errors else "blocked",
        "bridge_data_errors": bridge_errors,
        "population_manifest": population_manifest_payload,
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
    parser.add_argument(
        "--require-context-ready",
        action="store_true",
        help="Also require the complete 362-record post-trace HOMER context cache.",
    )
    parser.add_argument(
        "--require-bridge-data-ready",
        action="store_true",
        help="Require the complete context cache and verified bridge training view.",
    )
    args = parser.parse_args()
    report = check(args.config.resolve())
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" or (
        args.require_data_ready
        and (report["data_gate"] != "ready" or report["trace_gate"] != "ready")
    ) or (
        args.require_context_ready
        and (
            report["data_gate"] != "ready"
            or report["trace_gate"] != "ready"
            or report["context_gate"] != "ready"
        )
    ) or (
        args.require_bridge_data_ready
        and report["bridge_data_gate"] != "ready"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
