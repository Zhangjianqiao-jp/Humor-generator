#!/usr/bin/env python3
"""Sealed semantic confirmation for the Phase A5 repair candidate.

This keeps A4's channel-isolated, donor-reconstruction and zero-bridge
protocol, but evaluates the A5 role-specific bridge on all unseen held-out
clusters (internal_test + official_hia_unseen_test).  It is deliberately
separate so A4's immutable artifacts and protocol are never overwritten.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.cross_attention_bridge import ReceiverCrossAttentionTask
from humor_generator_v35.training.formal_bridge import load_trace_index, prepare_example
from humor_generator_v35.latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from re_evaluate_failed_semantic_bridges import length_matched_channel_donors
from run_outer_semantic_confirmation_a4 import (
    CHANNELS,
    evaluate_channel,
    first_rows_by_cluster,
    load_bridge,
    summarize,
)


DEFAULT_SEEDS = (20260901, 20260902, 20260903)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def sealed_unseen_rows(dataset: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return one row per unseen image cluster, with overlap fail-closed."""
    internal = first_rows_by_cluster(read_jsonl(dataset / "internal_test.jsonl"))
    official = first_rows_by_cluster(read_jsonl(dataset / "official_hia_unseen_test.jsonl"))
    left = {str(row["cluster_id"]) for row in internal}
    right = {str(row["cluster_id"]) for row in official}
    overlap = sorted(left & right)
    if overlap:
        raise RuntimeError(f"sealed A5 test splits overlap: {overlap[:5]}")
    rows = sorted(internal + official, key=lambda row: str(row["cluster_id"]))
    if len(rows) != 121:
        raise RuntimeError(f"A5 unseen evaluation expects 121 clusters, found {len(rows)}")
    split_by_cluster = {
        **{str(row["cluster_id"]): "internal_test" for row in internal},
        **{str(row["cluster_id"]): "official_hia_unseen_test" for row in official},
    }
    return rows, split_by_cluster


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-index", type=Path,
        help="Sealed Planner trace index for the evaluation split; defaults to the training/validation index.",
    )
    parser.add_argument(
        "--donor-trace-index", type=Path,
        help=(
            "Trace index containing the validation donor traces.  The sealed "
            "test index intentionally contains only held-out targets, so the "
            "donor index is merged read-only for counterfactual matching."
        ),
    )
    parser.add_argument(
        "--trace-input-manifest", type=Path,
        help="Hash-locked planner input manifest corresponding to --trace-index.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-clusters", type=int, help="Deterministic smoke subset")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config["loss"].get("semantic_objective") != "channel_isolated_v5":
        raise RuntimeError("A5 evaluator requires semantic_objective=channel_isolated_v5")
    if config["bridge"].get("projection_mode") != "per_channel":
        raise RuntimeError("A5 evaluator requires projection_mode=per_channel")
    if config["loss"].get("channel_visibility") != "target_only":
        raise RuntimeError("A5 evaluator requires target-only visibility")
    if config["training"].get("semantic_prompt_include_image") is not False:
        raise RuntimeError("A5 evaluator requires an image-free semantic prompt")
    if float(config["loss"].get("semantic_target_alignment", 0.0)) <= 0:
        raise RuntimeError("A5 evaluator requires receiver-native target alignment")
    completion = checkpoint_path.parent / "complete.json"
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not completion.is_file() or json.loads(completion.read_text()).get("status") != "complete":
        raise RuntimeError("A5 checkpoint is not complete")
    if not manifest_path.is_file():
        raise RuntimeError("A5 checkpoint is missing run_manifest.json")
    run_manifest = json.loads(manifest_path.read_text())
    if run_manifest.get("semantic_objective") != "channel_isolated_v5":
        raise RuntimeError("checkpoint manifest is not an A5 run")
    if run_manifest.get("projection_mode") != "per_channel":
        raise RuntimeError("checkpoint manifest does not record per-channel projections")

    dataset = ROOT / config["data"]["dataset"]
    trace_index_path = (
        args.trace_index.resolve()
        if args.trace_index is not None
        else (ROOT / config["data"]["trace_index"]).resolve()
    )
    trace_input_manifest = (
        args.trace_input_manifest.resolve()
        if args.trace_input_manifest is not None
        else (dataset / "trace_inputs.jsonl").resolve()
    )
    if not trace_input_manifest.is_file():
        raise FileNotFoundError(f"trace input manifest not found: {trace_input_manifest}")
    rows, split_by_cluster = sealed_unseen_rows(dataset)
    input_rows = {
        str(item["cluster_id"]): item for item in read_jsonl(trace_input_manifest)
    }
    expected_input_ids = {str(row["cluster_id"]) for row in rows}
    if set(input_rows) != expected_input_ids:
        raise RuntimeError(
            "sealed trace input manifest must contain exactly the evaluated clusters: "
            f"expected={len(expected_input_ids)} found={len(input_rows)}"
        )
    for row in rows:
        item = input_rows[str(row["cluster_id"])]
        if str(item.get("image_sha256")) != str(row["image_sha256"]) or str(
            item.get("standard_description")
        ) != str(row["standard_description"]):
            raise RuntimeError(f"sealed trace input mismatch for {row['cluster_id']}")
    if args.max_clusters is not None:
        if not 2 <= args.max_clusters <= len(rows):
            raise ValueError("--max-clusters must be between 2 and 121")
        rows = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"v35-a5-outer-smoke:{row['cluster_id']}".encode()
            ).digest(),
        )[:args.max_clusters]
    traces = load_trace_index(trace_index_path)
    donor_trace_index_path = (
        args.donor_trace_index.resolve()
        if args.donor_trace_index is not None
        else trace_index_path
    )
    if donor_trace_index_path != trace_index_path:
        donor_traces = load_trace_index(donor_trace_index_path)
        overlap = sorted(set(traces) & set(donor_traces))
        for cluster in overlap:
            if traces[cluster] != donor_traces[cluster]:
                raise RuntimeError(f"conflicting trace records for merged cluster {cluster}")
        traces.update({cluster: record for cluster, record in donor_traces.items() if cluster not in traces})
    missing = sorted({str(row["cluster_id"]) for row in rows} - set(traces))
    if missing:
        raise RuntimeError(f"A5 outer traces missing: {missing[:5]}")
    input_hash = sha256(trace_input_manifest)
    for cluster_id in expected_input_ids:
        provenance = traces[cluster_id].get("provenance", {})
        if provenance.get("trace_input_manifest_sha256") != input_hash:
            raise RuntimeError(f"trace provenance input hash mismatch for {cluster_id}")

    # Donors are drawn from validation only: these clusters were not used to
    # fit the bridge, and the policy remains frozen throughout evaluation.
    donor_rows = read_jsonl(dataset / "validation.jsonl")
    donor_map, donor_diagnostics = length_matched_channel_donors(
        rows, donor_rows, root=ROOT, trace_index=traces
    )
    backend = QwenBackend.load(
        config["model"]["name"], revision=config["model"]["revision"],
        adapter=ROOT / config["model"]["adapter"], load_in_4bit=True,
        min_visual_tokens=int(config["model"]["min_visual_tokens"]),
        max_visual_tokens=int(config["model"]["max_visual_tokens"]),
    )
    if model_device(backend.model).type != "cuda":
        raise RuntimeError("A5 outer confirmation requires CUDA")
    backend.model.eval()
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    bridge = load_bridge(config, checkpoint_path, backend)
    if not isinstance(bridge, ReceiverDrivenCrossAttentionBridge):
        raise RuntimeError("A5 checkpoint did not load a cross-attention bridge")
    task = ReceiverCrossAttentionTask(
        backend, bridge, root=ROOT, trace_index=traces,
        loss_config=config["loss"], max_target_tokens=int(config["training"]["max_target_tokens"]),
        stage="semantic_reconstruction", semantic_prompt_include_image=False,
    )

    details: list[dict[str, Any]] = []
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for index, row in enumerate(rows, start=1):
            cluster_id = str(row["cluster_id"])
            example = prepare_example(row, traces[cluster_id], seed=seed)
            for channel in CHANNELS:
                result = evaluate_channel(
                    task, example, channel, donor_map[cluster_id][channel]
                )
                details.append({
                    "seed": seed, "evaluation_index": index,
                    "cluster_id": cluster_id,
                    "split_group": split_by_cluster[cluster_id],
                    "row_id": row["row_id"], **result,
                })
                print(json.dumps({"status": "evaluated", "protocol": "A5", **details[-1]}), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize(
        details, seed=int(config["training"]["seed"]),
        min_gap=float(config["gate"]["min_channel_validation_gap"]),
        min_fraction=float(config["gate"]["min_channel_fraction_gap_gt_0"]),
    )
    # The imported reducer is intentionally reused for identical statistics;
    # only the protocol identity and sealed split metadata differ.
    summary.update({
        "protocol": "A5_receiver_native_target_alignment_v5",
        "clusters": len({row["cluster_id"] for row in rows}),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_completion_sha256": sha256(completion),
        "checkpoint_run_manifest_sha256": sha256(manifest_path),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "dataset": str(dataset), "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "trace_index": str(trace_index_path), "trace_index_sha256": sha256(trace_index_path),
        "donor_trace_index": str(donor_trace_index_path),
        "donor_trace_index_sha256": sha256(donor_trace_index_path),
        "trace_input_manifest": str(trace_input_manifest),
        "trace_input_manifest_sha256": sha256(trace_input_manifest),
        "frozen_adapter_manifest_sha256": sha256(ROOT / "manifests/frozen_7b_adapters.json"),
        "local_model_manifest_sha256": sha256(ROOT / "manifests/local_qwen2_5_vl_7b.json"),
        "homer_prompt_source_sha256": sha256(ROOT / "src/humor_generator_v35/homer/prompts.py"),
        "receiver_prompt_source_sha256": sha256(ROOT / "src/humor_generator_v35/training/cross_attention_bridge.py"),
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256(Path(__file__).resolve()), "git_commit": git_commit(),
        "selection_policy": "all_internal_test_plus_official_hia_unseen_test",
        "subset_mode": "smoke" if args.max_clusters is not None else "sealed_unseen_confirmation",
        "donor_policy": "validation_split_length_matched_channel_donor",
        "donor_diagnostics": donor_diagnostics,
        "caption_generation": "excluded", "preference_learning": "excluded",
        "semantic_target_alignment": "receiver_final_layer_target_span_mean_cosine",
    })
    args.output.mkdir(parents=True, exist_ok=False)
    details_path = args.output / "cluster_seed_channel_details.jsonl"
    summary_path = args.output / "summary.json"
    details_path.write_text("".join(json.dumps(row) + "\n" for row in details))
    (args.output / "outer_cluster_ids.json").write_text(
        json.dumps({"clusters": sorted({row["cluster_id"] for row in rows}), "count": len(rows)}, indent=2) + "\n"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "protocol": summary["protocol"],
        "evaluator": summary["evaluator"], "evaluator_sha256": summary["evaluator_sha256"],
        "details_sha256": sha256(details_path), "summary_sha256": sha256(summary_path),
        "checkpoint_sha256": summary["checkpoint_sha256"], "config_sha256": summary["config_sha256"],
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "trace_index_sha256": summary["trace_index_sha256"],
        "target_only": True, "semantic_prompt_include_image": False,
        "zero_bridge_included": True, "projection_mode": "per_channel",
        "semantic_target_alignment": True, "donor_policy": summary["donor_policy"],
        "git_commit": summary["git_commit"], "subset_mode": summary["subset_mode"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
