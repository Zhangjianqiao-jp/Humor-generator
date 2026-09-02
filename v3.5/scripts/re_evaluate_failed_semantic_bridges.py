#!/usr/bin/env python3
"""Re-evaluate the completed v1/v2 bridge checkpoints with the corrected protocol.

This is a frozen-checkpoint evaluator, not a training entrypoint.  It keeps the
old v1 architecture intact, uses one representative caption per image cluster,
and measures same-image, single-channel counterfactual gaps.  A small pilot
cannot establish superiority; the report therefore distinguishes a clear
technical failure from an inconclusive semantic result.
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

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from humor_generator_v35.latent.legacy_cross_attention import (
    LegacyV1ReceiverDrivenCrossAttentionBridge,
)
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.cross_attention_bridge import ReceiverCrossAttentionTask
from humor_generator_v35.training.formal_bridge import (
    cluster_balanced_rows,
    hard_negative_cluster_map,
    load_trace_index,
    prepare_example,
)
from humor_generator_v35.training.losses import sequence_log_probability


CHANNELS = ("conflict", "local", "global")


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


def fixed_hash_sample_clusters(
    rows: list[dict[str, Any]], limit: int | None, *, seed: int, split: str,
) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    clusters = sorted({row["cluster_id"] for row in rows})
    if limit < 2 or limit > len(clusters):
        raise ValueError(f"invalid {split} cluster limit {limit} for {len(clusters)} clusters")
    selected = sorted(
        clusters,
        key=lambda cluster: hashlib.sha256(
            f"v35-pilot:{seed}:{split}:{cluster}".encode()
        ).digest(),
    )[:limit]
    allowed = set(selected)
    return [row for row in rows if row["cluster_id"] in allowed]


def bootstrap_interval(values: list[float], *, seed: int, draws: int = 10000) -> list[float]:
    if len(values) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    )
    return [means[int(draws * 0.025)], means[int(draws * 0.975) - 1]]


def build_bridge(method: str, width: int, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    kwargs = {
        "layer_indices": [int(value) for value in config["bridge"]["layer_indices"]],
        "bottleneck_dim": int(config["bridge"]["bottleneck_dim"]),
        "heads": int(config["bridge"]["heads"]),
        "gate_init": float(config["bridge"].get("gate_init", 0.1)),
    }
    if method == "v1":
        bridge = LegacyV1ReceiverDrivenCrossAttentionBridge(width, width, **kwargs)
    elif method == "v2":
        bridge = ReceiverDrivenCrossAttentionBridge(
            width, width, channel_fusion="learned", **kwargs
        )
    else:
        raise ValueError("method must be v1 or v2")
    return bridge.to(device)


@torch.no_grad()
def evaluate_cluster(
    task: ReceiverCrossAttentionTask,
    row: dict[str, Any],
    donor_cluster: str,
) -> dict[str, Any]:
    """Measure full-target log-probability under one-channel swaps.

    The image, prompt, target and the other two memory channels remain fixed.
    This makes the comparison valid for old full-plan checkpoints while still
    exposing which channel, if any, affects the Receiver's native output path.
    """
    example = prepare_example(row, task.trace_index[row["cluster_id"]], seed=0)
    prepared = task.prepare(example)
    embeddings, mask, positions, targets, _teacher_cpu, matched_states = prepared
    donor_states = task._states(donor_cluster)

    def score(states: dict[str, Any]) -> tuple[float, dict[str, float]]:
        logits = task._logits(embeddings, mask, positions, targets, states)
        logp = sequence_log_probability(logits, targets).detach().float().cpu()
        diagnostics = task.bridge.last_diagnostics
        diag = {
            "mean_gate": statistics.fmean(item.gate for item in diagnostics),
            "mean_relative_update_norm": statistics.fmean(
                item.relative_update_norm for item in diagnostics
            ),
        }
        return float(logp.mean()), diag

    matched_logp, matched_diag = score(matched_states)
    row_out: dict[str, Any] = {
        "cluster_id": row["cluster_id"],
        "row_id": row["row_id"],
        "donor_cluster_id": donor_cluster,
        "matched_logp": matched_logp,
        **matched_diag,
    }
    swapped_states = dict(matched_states)
    all_swapped = dict(donor_states)
    all_logp, _ = score(all_swapped)
    row_out["full_plan_swap_logp"] = all_logp
    row_out["full_plan_swap_gap"] = matched_logp - all_logp
    for channel in CHANNELS:
        swapped_states[channel] = donor_states[channel]
        counterfactual_logp, _ = score(swapped_states)
        row_out[f"counterfactual_logp_{channel}"] = counterfactual_logp
        row_out[f"gap_{channel}"] = matched_logp - counterfactual_logp
        swapped_states[channel] = matched_states[channel]
    return row_out


def summarize(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    intervals: dict[str, list[float]] = {}
    channel_summary: dict[str, Any] = {}
    for index, channel in enumerate(CHANNELS):
        values = [float(row[f"gap_{channel}"]) for row in rows]
        intervals[channel] = bootstrap_interval(values, seed=seed + index)
        channel_summary[channel] = {
            "mean_gap": statistics.fmean(values),
            "median_gap": statistics.median(values),
            "fraction_gap_gt_0": sum(value > 0 for value in values) / len(values),
            "fraction_gap_gt_0_02": sum(value > 0.02 for value in values) / len(values),
            "bootstrap_95ci": intervals[channel],
        }
    point_pass = all(
        summary["mean_gap"] >= 0 and summary["fraction_gap_gt_0"] >= 0.60
        for summary in channel_summary.values()
    )
    ci_positive = all(intervals[channel][0] > 0 for channel in CHANNELS)
    # This is deliberately a three-state pilot routing decision.  It is not a
    # claim that the old method is scientifically superior or inferior.
    status = (
        "strong_go" if point_pass and ci_positive
        else "go_to_outer_semantic_validation" if point_pass
        else "pilot_inconclusive"
    )
    return {
        "status": status,
        "clusters": len(rows),
        "point_gate_pass": point_pass,
        "all_channel_ci_lower_bounds_above_zero": ci_positive,
        "channels": channel_summary,
        "full_plan_swap": {
            "mean_gap": statistics.fmean(float(row["full_plan_swap_gap"]) for row in rows),
            "bootstrap_95ci": bootstrap_interval(
                [float(row["full_plan_swap_gap"]) for row in rows], seed=seed + 3
            ),
        },
        "interpretation": (
            "This corrected re-evaluation is cluster-balanced and counterfactual. "
            "pilot_inconclusive means the old checkpoint cannot be declared ineffective "
            "from 24 clusters; it requires the sealed outer semantic validation."
        ),
    }


def evaluate_method(
    method: str,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if config["training"].get("max_validation_clusters") != 24:
        raise ValueError("re-evaluation expects the preregistered 24-cluster validation subset")
    dataset = ROOT / config["data"]["dataset"]
    trace_path = ROOT / config["data"]["trace_index"]
    validation_rows = read_jsonl(dataset / "validation.jsonl")
    validation_rows = fixed_hash_sample_clusters(
        validation_rows,
        int(config["training"]["max_validation_clusters"]),
        seed=int(config["training"]["seed"]),
        split="validation",
    )
    representatives = cluster_balanced_rows(
        validation_rows, epoch=0, seed=int(config["training"]["seed"])
    )
    traces = load_trace_index(trace_path)
    cluster_ids = {row["cluster_id"] for row in representatives}
    missing = sorted(cluster_ids - set(traces))
    if missing:
        raise RuntimeError(f"missing traces for clusters: {missing[:5]}")
    donors, donor_diagnostics = hard_negative_cluster_map(validation_rows, traces)

    adapter = config["model"].get("adapter")
    backend = QwenBackend.load(
        config["model"]["name"],
        revision=config["model"]["revision"],
        adapter=None if adapter is None else ROOT / adapter,
        load_in_4bit=True,
        min_visual_tokens=int(config["model"]["min_visual_tokens"]),
        max_visual_tokens=int(config["model"]["max_visual_tokens"]),
    )
    device = model_device(backend.model)
    if device.type != "cuda":
        raise RuntimeError("corrected bridge re-evaluation requires CUDA")
    backend.model.eval()
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    width = int(backend.model.get_input_embeddings().weight.shape[1])
    bridge = build_bridge(method, width, config, device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "bridge" not in state:
        raise RuntimeError(f"checkpoint has no bridge state: {checkpoint_path}")
    bridge.load_state_dict(state["bridge"], strict=True)
    bridge.eval()
    task = ReceiverCrossAttentionTask(
        backend,
        bridge,  # type: ignore[arg-type]
        root=ROOT,
        trace_index=traces,
        loss_config=config["loss"],
        max_target_tokens=int(config["training"]["max_target_tokens"]),
        stage="semantic_reconstruction",
    )

    details = []
    for index, row in enumerate(representatives, start=1):
        result = evaluate_cluster(task, row, donors[row["cluster_id"]])
        result["evaluation_index"] = index
        details.append(result)
        print(json.dumps({"status": "evaluated", "method": method, **result}), flush=True)
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=False)
    detail_path = output_dir / "cluster_counterfactuals.jsonl"
    detail_path.write_text("".join(json.dumps(row) + "\n" for row in details))
    summary = summarize(details, seed=int(config["training"]["seed"]))
    summary.update({
        "method": method,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "trace_index_sha256": sha256(trace_path),
        "git_commit": git_commit(),
        "validation_cluster_ids": sorted(cluster_ids),
        "validation_cluster_ids_sha256": hashlib.sha256(
            "\n".join(sorted(cluster_ids)).encode()
        ).hexdigest(),
        "donor_policy": "description_tfidf_nearest_same-source_different-conflict",
        "donor_diagnostics": donor_diagnostics,
        "evaluation_protocol": "full-target_same-image_single-channel_counterfactual",
        "excluded": ["sealed_test", "caption_generation", "retraining"],
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "method": method,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "config_sha256": summary["config_sha256"],
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "trace_index_sha256": summary["trace_index_sha256"],
        "git_commit": summary["git_commit"],
        "cluster_details_sha256": sha256(detail_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--method", choices=("v1", "v2"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    summary = evaluate_method(
        args.method,
        args.config.resolve(),
        args.checkpoint.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
