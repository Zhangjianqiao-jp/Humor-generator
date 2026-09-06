#!/usr/bin/env python3
"""CPU-only gate for the current HOMER + latent-bridge caption route."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.training.public_bridge import (
    PUBLIC_BRIDGE_PROMPT_TRACK,
    load_context_index,
    validate_row_context,
)


DATA_VERSION = "homer_pretrained_7b_public_release_362"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
EXPECTED_DATASET = "data/processed/homer_pretrained_7b_bridge_362"
EXPECTED_TRACE_INPUT = "data/processed/homer_pretrained_7b_public_release_362/trace_inputs.jsonl"
EXPECTED_TRACE = "data/cache/homer_pretrained_7b_planner_traces/index.jsonl"
EXPECTED_CONTEXT = "data/cache/homer_pretrained_7b_homer_context/index.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    experiment = config.get("experiment", {})
    protocol = config.get("protocol", {})
    model = config.get("model", {})
    data = config.get("data", {})
    if experiment.get("route") != "homer_public_code_bridge":
        errors.append("experiment.route must be homer_public_code_bridge")
    if protocol.get("bridge_prompt_track") != PUBLIC_BRIDGE_PROMPT_TRACK:
        errors.append("bridge prompt track is not the current official-caption replacement track")
    if protocol.get("training_target_policy") != "source_caption_only_no_fabricated_explanation":
        errors.append("training target policy must avoid fabricated explanations")
    if model.get("revision") != REVISION:
        errors.append("Qwen revision is not the pinned pretrained revision")
    if model.get("model_manifest") != "manifests/local_qwen2_5_vl_7b.json":
        errors.append("model.model_manifest must be the pinned local Qwen manifest")
    for key in ("adapter", "planner_adapter", "generator_adapter"):
        if model.get(key) is not None:
            errors.append(f"model.{key} must be null")
    for key in ("frozen",):
        if model.get(key) is not True:
            errors.append(f"model.{key} must be true")
    if data.get("data_version") != DATA_VERSION:
        errors.append("data.data_version mismatch")
    if data.get("target_policy") != "source_caption_only_no_fabricated_explanation":
        errors.append("data.target_policy must record the source-caption-only policy")
    for key, expected in (
        ("dataset", EXPECTED_DATASET),
        ("trace_input_manifest", EXPECTED_TRACE_INPUT),
        ("trace_index", EXPECTED_TRACE),
        ("context_index", EXPECTED_CONTEXT),
    ):
        if data.get(key) != expected:
            errors.append(f"data.{key} must be {expected}")
        if isinstance(data.get(key), str) and "latent_bridge_v35" in data[key]:
            errors.append(f"data.{key} points to historical latent_bridge_v35")
    dataset = ROOT / EXPECTED_DATASET
    trace_index = ROOT / EXPECTED_TRACE
    context_index = ROOT / EXPECTED_CONTEXT
    for path, label in (
        (dataset / "manifest.json", "bridge dataset manifest"),
        (trace_index, "Planner trace index"),
        (context_index, "HOMER context index"),
    ):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
    dataset_payload: dict[str, Any] = {}
    if (dataset / "manifest.json").is_file():
        try:
            dataset_payload = json.loads((dataset / "manifest.json").read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"bridge dataset manifest is invalid JSON: {exc}")
        else:
            if dataset_payload.get("data_version") != DATA_VERSION:
                errors.append("bridge dataset manifest data_version mismatch")
            if dataset_payload.get("status") != "verified_current_pretrained_bridge_training_view":
                errors.append("bridge dataset manifest is not a verified current training view")
            if dataset_payload.get("final_evaluation_policy", "").find("complete_ranking_csv") < 0:
                errors.append("bridge dataset does not declare the full-ranking final-evaluation policy")
    trace_records: list[dict[str, Any]] = []
    if trace_index.is_file():
        trace_records = read_jsonl(trace_index)
        if len(trace_records) != 362:
            errors.append(f"Planner trace index must contain 362 records, got {len(trace_records)}")
        for record in trace_records:
            if record.get("planner_revision") != REVISION or record.get("planner_adapter") is not None:
                errors.append(f"trace {record.get('cluster_id')} is not adapter-free pinned Planner")
                break
    contexts: dict[str, Any] = {}
    if context_index.is_file():
        try:
            contexts = load_context_index(context_index)
        except Exception as exc:
            errors.append(f"HOMER context index validation failed: {exc}")
        else:
            if len(contexts) != 362:
                errors.append(f"HOMER context index must contain 362 records, got {len(contexts)}")
            context_manifest = context_index.parent / "manifest.json"
            if not context_manifest.is_file():
                errors.append(f"missing HOMER context manifest: {context_manifest}")
            else:
                try:
                    context_payload = json.loads(context_manifest.read_text())
                    if context_payload.get("status") != "pass":
                        errors.append("HOMER context manifest is not complete")
                    if context_payload.get("trace_index_sha256") != sha256(trace_index):
                        errors.append("HOMER context trace-index hash mismatch")
                    if context_payload.get("trace_input_manifest_sha256") != sha256(
                        ROOT / str(data.get("trace_input_manifest", ""))
                    ):
                        errors.append("HOMER context trace-input hash mismatch")
                except (json.JSONDecodeError, OSError) as exc:
                    errors.append(f"HOMER context manifest is invalid: {exc}")
            trace_by_cluster = {
                str(record.get("cluster_id")): record for record in trace_records
            }
            for cluster, context in contexts.items():
                trace = trace_by_cluster.get(cluster)
                if trace is None:
                    errors.append(f"HOMER context {cluster} has no matching Planner trace")
                    continue
                if context.trace_path != str(trace.get("trace_path")):
                    errors.append(f"HOMER context {cluster} trace path mismatch")
                if context.trace_sha256 != str(trace.get("trace_sha256")):
                    errors.append(f"HOMER context {cluster} trace hash mismatch")
    dataset_clusters: set[str] = set()
    row_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = dataset / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"missing bridge dataset split: {path}")
            continue
        rows = read_jsonl(path)
        row_counts[split] = len(rows)
        for row in rows:
            cluster = str(row.get("cluster_id", ""))
            dataset_clusters.add(cluster)
            if not str(row.get("caption", "")).strip():
                errors.append(f"{row.get('row_id')}: empty caption")
            if cluster not in contexts:
                errors.append(f"{row.get('row_id')}: missing HOMER context")
            elif contexts:
                try:
                    validate_row_context(row, contexts[cluster])
                except ValueError as exc:
                    errors.append(f"{row.get('row_id')}: {exc}")
            if "latent_bridge_v35" in json.dumps(row, ensure_ascii=False):
                errors.append(f"{row.get('row_id')}: historical data reference")
    if contexts and dataset_clusters - set(contexts):
        errors.append("bridge dataset contains clusters outside current context index")
    result = {
        "status": "pass" if not errors else "fail",
        "config": str(config_path),
        "prompt_track": protocol.get("bridge_prompt_track"),
        "data_version": DATA_VERSION,
        "revision": REVISION,
        "dataset": str(dataset),
        "dataset_manifest_sha256": (
            sha256(dataset / "manifest.json") if (dataset / "manifest.json").is_file() else None
        ),
        "trace_records": len(trace_records),
        "trace_index_sha256": sha256(trace_index) if trace_index.is_file() else None,
        "context_records": len(contexts),
        "context_index_sha256": sha256(context_index) if context_index.is_file() else None,
        "row_counts": row_counts,
        "errors": errors,
    }
    return result


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/pilot/cross_attention_caption_pretrained_public.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.config.resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
