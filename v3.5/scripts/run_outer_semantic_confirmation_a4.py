#!/usr/bin/env python3
"""Run the sealed A4 outer semantic confirmation.

This evaluator is intentionally separate from the historical A3 evaluator.  It
only measures the channel-isolated semantic protocol used by
``channel_isolated_v4``; it does not generate captions or perform preference
learning.  A non-pass is reported as inconclusive and never as a model
failure.  The primary unit is an image cluster, with three shared seeds kept
for provenance and seed-variance reporting.
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
from humor_generator_v35.latent.bridges import TypedLatentBridge
from humor_generator_v35.latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.cross_attention_bridge import ReceiverCrossAttentionTask
from humor_generator_v35.training.formal_bridge import (
    PreparedExample,
    load_trace_index,
    prepare_example,
)
from humor_generator_v35.training.losses import sequence_log_probability, token_cross_entropy
from humor_generator_v35.training.memory_safe import caption_only_logits
from re_evaluate_failed_semantic_bridges import (
    bootstrap_interval,
    length_matched_channel_donors,
    wilson_interval,
)


CHANNELS = ("conflict", "local", "global")
DEFAULT_SEEDS = (20260830, 20260831, 20260832)


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


def first_rows_by_cluster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["row_id"]):
        selected.setdefault(str(row["cluster_id"]), row)
    return [selected[key] for key in sorted(selected)]


def sealed_outer_clusters(
    validation_rows: list[dict[str, Any]], selected_for_early_stopping: set[str],
    max_clusters: int | None,
) -> list[str]:
    all_clusters = {str(row["cluster_id"]) for row in validation_rows}
    if len(all_clusters) != 64:
        raise RuntimeError(f"A4 outer protocol expects 64 validation clusters, found {len(all_clusters)}")
    if len(selected_for_early_stopping) != 24:
        raise RuntimeError(
            "A4 checkpoint must identify exactly 24 validation clusters used for early stopping"
        )
    if not selected_for_early_stopping <= all_clusters:
        missing = sorted(selected_for_early_stopping - all_clusters)
        raise RuntimeError(f"A4 early-stopping cluster list contains unknown IDs: {missing[:5]}")
    candidates = sorted(all_clusters - selected_for_early_stopping)
    if max_clusters is None:
        if len(candidates) != 40:
            raise RuntimeError(f"A4 sealed outer protocol expects 40 clusters, found {len(candidates)}")
        return candidates
    if not 2 <= max_clusters <= len(candidates):
        raise ValueError(f"--max-clusters must be between 2 and {len(candidates)}")
    # Smoke subsets are deterministic and must not depend on input row order.
    return sorted(
        candidates,
        key=lambda cluster: hashlib.sha256(f"v35-a4-outer-smoke:{cluster}".encode()).digest(),
    )[:max_clusters]


def load_bridge(
    config: dict[str, Any], checkpoint_path: Path, backend: QwenBackend,
) -> ReceiverDrivenCrossAttentionBridge:
    width = int(backend.model.get_input_embeddings().weight.shape[1])
    bridge_config = config["bridge"]
    bridge = ReceiverDrivenCrossAttentionBridge(
        width,
        width,
        layer_indices=[int(value) for value in bridge_config["layer_indices"]],
        bottleneck_dim=int(bridge_config["bottleneck_dim"]),
        heads=int(bridge_config["heads"]),
        gate_init=float(bridge_config.get("gate_init", 0.1)),
        channel_fusion=str(bridge_config["channel_fusion"]),
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "bridge" not in payload:
        raise RuntimeError(f"A4 checkpoint has no bridge state: {checkpoint_path}")
    bridge.load_state_dict(payload["bridge"], strict=True)
    bridge.to(model_device(backend.model)).eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    return bridge


def score_without_bridge(
    task: ReceiverCrossAttentionTask,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Score the same receiver prompt with the communication residual absent."""
    logits = caption_only_logits(
        task.backend.model,
        inputs_embeds=embeddings,
        attention_mask=mask,
        position_ids=positions,
        caption_tokens=int(targets.shape[1]),
    )
    nll = float(token_cross_entropy(logits, targets).float().cpu())
    logp = float(sequence_log_probability(logits, targets).mean().float().cpu())
    return logp, nll


@torch.no_grad()
def evaluate_channel(
    task: ReceiverCrossAttentionTask,
    example: PreparedExample,
    channel: str,
    donor_cluster: str,
) -> dict[str, Any]:
    """Evaluate matched, isolated counterfactual, donor, and zero-bridge paths."""
    prepared = task.prepare_semantic_channel(example, channel)
    embeddings, mask, positions, targets, _teacher, matched_states = prepared
    donor_states = task._states(donor_cluster)
    counterfactual_states = dict(matched_states)
    counterfactual_states[channel] = donor_states[channel]
    active = (channel,)

    matched_logits = task._logits(
        embeddings, mask, positions, targets, matched_states, active_channels=active
    )
    matched_nll = float(token_cross_entropy(matched_logits, targets).float().cpu())
    matched_logp = float(sequence_log_probability(matched_logits, targets).mean().float().cpu())
    del matched_logits

    counterfactual_logits = task._logits(
        embeddings, mask, positions, targets, counterfactual_states, active_channels=active
    )
    counterfactual_logp = float(
        sequence_log_probability(counterfactual_logits, targets).mean().float().cpu()
    )
    del counterfactual_logits

    donor_prepared = task.prepare_semantic_target(
        example, channel, donor_states[channel].semantics
    )
    donor_embeddings, donor_mask, donor_positions, donor_targets, _, _ = donor_prepared
    donor_logits = task._logits(
        donor_embeddings,
        donor_mask,
        donor_positions,
        donor_targets,
        counterfactual_states,
        active_channels=active,
    )
    donor_nll = float(token_cross_entropy(donor_logits, donor_targets).float().cpu())
    donor_logp = float(sequence_log_probability(donor_logits, donor_targets).mean().float().cpu())
    del donor_logits

    zero_logp, zero_nll = score_without_bridge(task, embeddings, mask, positions, targets)
    target_length = int(matched_states[channel].states.shape[1])
    donor_length = int(donor_states[channel].states.shape[1])
    return {
        "channel": channel,
        "donor_cluster_id": donor_cluster,
        "matched_logp": matched_logp,
        "counterfactual_logp": counterfactual_logp,
        "gap": matched_logp - counterfactual_logp,
        "matched_nll": matched_nll,
        "donor_logp": donor_logp,
        "donor_nll": donor_nll,
        "zero_bridge_logp": zero_logp,
        "zero_bridge_nll": zero_nll,
        "zero_bridge_gap": 0.0,
        "target_length": target_length,
        "donor_length": donor_length,
        "length_delta": donor_length - target_length,
        "active_channels": list(active),
        "semantic_prompt_include_image": False,
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("correlation vectors must have equal non-zero length")
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = sum((a - left_mean) ** 2 for a in left)
    right_scale = sum((b - right_mean) ** 2 for b in right)
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale) ** 0.5


def summarize(
    details: list[dict[str, Any]],
    *,
    seed: int,
    min_gap: float,
    min_fraction: float,
) -> dict[str, Any]:
    clusters = sorted({str(row["cluster_id"]) for row in details})
    seeds = sorted({int(row["seed"]) for row in details})
    by_key = {(int(row["seed"]), str(row["cluster_id"]), str(row["channel"])): row for row in details}
    expected = len(clusters) * len(seeds) * len(CHANNELS)
    if len(by_key) != expected or len(details) != expected:
        raise RuntimeError("A4 outer details must contain exactly one row per seed/cluster/channel")

    channels: dict[str, Any] = {}
    seed_summary: dict[str, dict[str, float]] = defaultdict(dict)
    for channel_index, channel in enumerate(CHANNELS):
        cluster_values = {
            cluster: statistics.fmean(
                float(by_key[(seed_value, cluster, channel)]["gap"]) for seed_value in seeds
            )
            for cluster in clusters
        }
        values = list(cluster_values.values())
        interval = bootstrap_interval(values, seed=seed + channel_index)
        per_seed = [
            statistics.fmean(float(by_key[(seed_value, cluster, channel)]["gap"]) for cluster in clusters)
            for seed_value in seeds
        ]
        length_deltas = [
            abs(int(by_key[(seed_value, cluster, channel)]["length_delta"]))
            for seed_value in seeds for cluster in clusters
        ]
        gaps = [float(by_key[(seed_value, cluster, channel)]["gap"]) for seed_value in seeds for cluster in clusters]
        channels[channel] = {
            "clusters": len(clusters),
            "mean_gap": statistics.fmean(values),
            "median_gap": statistics.median(values),
            "fraction_gap_gt_0": sum(value > 0 for value in values) / len(values),
            "fraction_gap_gt_0_ci95": wilson_interval(
                sum(value > 0 for value in values), len(values)
            ),
            "bootstrap_95ci": interval,
            "seed_mean_gap_values": per_seed,
            "seed_gap_mean": statistics.fmean(per_seed),
            "seed_gap_std": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
            "max_abs_length_delta": max(length_deltas),
            "gap_vs_abs_length_delta_pearson": pearson(gaps, length_deltas),
            "zero_bridge_mean_gap": 0.0,
        }
        for seed_value, value in zip(seeds, per_seed):
            seed_summary[str(seed_value)][channel] = value

    point_checks = {
        channel: (
            channels[channel]["mean_gap"] >= min_gap
            and channels[channel]["fraction_gap_gt_0"] >= min_fraction
        )
        for channel in CHANNELS
    }
    ci_checks = {channel: channels[channel]["bootstrap_95ci"][0] > 0 for channel in CHANNELS}
    point_pass = all(point_checks.values())
    ci_pass = all(ci_checks.values())
    status = "outer_semantic_go" if point_pass and ci_pass else "outer_semantic_inconclusive"
    return {
        "status": status,
        "engineering_gate_pass": True,
        "clusters": len(clusters),
        "seeds": seeds,
        "statistical_unit": "image_cluster",
        "protocol": "A4_channel_isolated_v4",
        "channel_summary": channels,
        "seed_summary": dict(seed_summary),
        "point_checks": point_checks,
        "bootstrap_ci_lower_bound_above_zero": ci_checks,
        "point_gate_pass": point_pass,
        "all_channel_ci_lower_bounds_above_zero": ci_pass,
        "thresholds": {
            "min_channel_gap": min_gap,
            "min_fraction_gap_gt_0": min_fraction,
        },
        "zero_bridge": {
            "included": True,
            "expected_gap": 0.0,
            "interpretation": "control only; no communication residual is injected",
        },
        "interpretation": (
            "A4 outer confirmation is a semantic mechanism gate, not a humorous-caption "
            "quality evaluation. Inconclusive means no method-level conclusion and no caption "
            "unlock; it is not evidence that latent communication is ineffective."
        ),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--max-clusters", type=int,
        help="Real-trace smoke subset only; omitting it enforces all 40 sealed clusters.",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config["loss"].get("semantic_objective") != "channel_isolated_v4":
        raise RuntimeError("A4 evaluator requires semantic_objective=channel_isolated_v4")
    if config["loss"].get("channel_visibility") != "target_only":
        raise RuntimeError("A4 evaluator requires channel_visibility=target_only")
    if config["bridge"].get("channel_fusion") != "fixed_equal":
        raise RuntimeError("A4 evaluator requires fixed_equal channel fusion")
    if bool(config["training"].get("semantic_prompt_include_image", True)):
        raise RuntimeError("A4 evaluator requires semantic_prompt_include_image=false")

    completion_path = checkpoint_path.parent / "complete.json"
    run_manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not completion_path.is_file() or json.loads(completion_path.read_text()).get("status") != "complete":
        raise RuntimeError("A4 checkpoint has no validated complete.json")
    if not run_manifest_path.is_file():
        raise RuntimeError("A4 checkpoint is missing run_manifest.json")
    run_manifest = json.loads(run_manifest_path.read_text())
    if run_manifest.get("semantic_objective") != "channel_isolated_v4":
        raise RuntimeError("checkpoint manifest is not an A4 channel-isolated run")
    if run_manifest.get("channel_visibility") != "target_only":
        raise RuntimeError("checkpoint manifest does not record target-only visibility")
    selected_validation = {str(value) for value in run_manifest.get("validation_cluster_ids", [])}

    dataset = ROOT / config["data"]["dataset"]
    trace_index_path = ROOT / config["data"]["trace_index"]
    validation_rows = read_jsonl(dataset / "validation.jsonl")
    outer_ids = sealed_outer_clusters(validation_rows, selected_validation, args.max_clusters)
    representatives = first_rows_by_cluster(
        [row for row in validation_rows if str(row["cluster_id"]) in set(outer_ids)]
    )
    if len(representatives) != len(outer_ids):
        raise RuntimeError("A4 outer validation did not produce one row per image cluster")
    traces = load_trace_index(trace_index_path)
    missing = sorted(set(outer_ids) - set(traces))
    if missing:
        raise RuntimeError(f"A4 outer validation is missing traces: {missing[:5]}")

    donor_rows = read_jsonl(dataset / "train.jsonl")
    fitted_clusters = {str(value) for value in run_manifest.get("train_cluster_ids", [])}
    donor_rows = [row for row in donor_rows if str(row["cluster_id"]) not in fitted_clusters]
    if not donor_rows:
        raise RuntimeError("A4 checkpoint training-cluster exclusion removed all donor rows")
    donor_map, donor_diagnostics = length_matched_channel_donors(
        representatives, donor_rows, root=ROOT, trace_index=traces
    )

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
        raise RuntimeError("A4 outer confirmation requires CUDA")
    backend.model.eval()
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    bridge = load_bridge(config, checkpoint_path, backend)
    task = ReceiverCrossAttentionTask(
        backend,
        bridge,
        root=ROOT,
        trace_index=traces,
        loss_config=config["loss"],
        max_target_tokens=int(config["training"]["max_target_tokens"]),
        stage="semantic_reconstruction",
        semantic_prompt_include_image=False,
    )

    details: list[dict[str, Any]] = []
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for index, row in enumerate(representatives, start=1):
            example = prepare_example(row, traces[row["cluster_id"]], seed=seed)
            cluster_id = str(row["cluster_id"])
            for channel in CHANNELS:
                result = evaluate_channel(task, example, channel, donor_map[cluster_id][channel])
                details.append({
                    "seed": seed,
                    "evaluation_index": index,
                    "cluster_id": cluster_id,
                    "row_id": row["row_id"],
                    **result,
                })
                print(json.dumps({"status": "evaluated", "protocol": "A4", **details[-1]}), flush=True)
            torch.cuda.empty_cache()

    summary = summarize(
        details,
        seed=int(config["training"]["seed"]),
        min_gap=float(config["gate"]["min_channel_validation_gap"]),
        min_fraction=float(config["gate"]["min_channel_fraction_gap_gt_0"]),
    )
    summary.update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_completion_sha256": sha256(completion_path),
        "checkpoint_run_manifest_sha256": sha256(run_manifest_path),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "trace_index": str(trace_index_path),
        "trace_index_sha256": sha256(trace_index_path),
        "trace_input_manifest_sha256": sha256(dataset / "trace_inputs.jsonl"),
        "frozen_adapter_manifest_sha256": sha256(ROOT / "manifests/frozen_7b_adapters.json"),
        "local_model_manifest_sha256": sha256(ROOT / "manifests/local_qwen2_5_vl_7b.json"),
        "homer_prompt_source_sha256": sha256(ROOT / "src/humor_generator_v35/homer/prompts.py"),
        "receiver_prompt_source_sha256": sha256(
            ROOT / "src/humor_generator_v35/training/cross_attention_bridge.py"
        ),
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "git_commit": git_commit(),
        "validation_early_stopping_cluster_ids": sorted(selected_validation),
        "outer_cluster_ids": sorted(outer_ids),
        "outer_cluster_ids_sha256": hashlib.sha256("\n".join(sorted(outer_ids)).encode()).hexdigest(),
        "selection_policy": "all_validation_clusters_minus_A4_early_stopping_24",
        "subset_mode": "smoke" if args.max_clusters is not None else "sealed_outer_confirmation",
        "donor_policy": "per_channel_length_matched_train_split_excluding_A4_fit_clusters",
        "donor_diagnostics": donor_diagnostics,
        "caption_generation": "excluded",
        "preference_learning": "excluded",
        "info_nce_primary_status": "not_used_as_causal_gate; current teacher is auxiliary stationary coordinate",
    })
    args.output.mkdir(parents=True, exist_ok=False)
    details_path = args.output / "cluster_seed_channel_details.jsonl"
    summary_path = args.output / "summary.json"
    details_path.write_text("".join(json.dumps(row) + "\n" for row in details))
    (args.output / "outer_cluster_ids.json").write_text(
        json.dumps({"clusters": sorted(outer_ids), "count": len(outer_ids)}, indent=2) + "\n"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "protocol": "A4_channel_isolated_v4",
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": summary["evaluator_sha256"],
        "details_sha256": sha256(details_path),
        "summary_sha256": sha256(summary_path),
        "outer_cluster_ids_sha256": summary["outer_cluster_ids_sha256"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "config_sha256": summary["config_sha256"],
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "trace_index_sha256": summary["trace_index_sha256"],
        "target_only": True,
        "semantic_prompt_include_image": False,
        "zero_bridge_included": True,
        "donor_policy": summary["donor_policy"],
        "git_commit": summary["git_commit"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
