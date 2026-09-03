#!/usr/bin/env python3
"""Run the pre-registered outer semantic confirmation for the A3 bridge.

This is an evaluation-only entrypoint.  It uses the 40 validation clusters
that were not used for A3 early stopping, keeps the A3 bridge and both 7B
policies frozen, and repeats the length-matched single-channel counterfactual
protocol.  It deliberately reports an inconclusive outcome without failing
the job: a statistical non-pass is not an engineering failure.
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

from humor_generator_v35.data.traces import load_trace, read_jsonl
from humor_generator_v35.latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.cross_attention_bridge import ReceiverCrossAttentionTask
from humor_generator_v35.training.formal_bridge import (
    PreparedExample,
    cluster_balanced_rows,
    load_trace_index,
    prepare_example,
)
from humor_generator_v35.training.losses import sequence_log_probability, token_cross_entropy
from humor_generator_v35.latent.bridges import TypedLatentBridge
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
        selected.setdefault(row["cluster_id"], row)
    return [selected[key] for key in sorted(selected)]


def outer_clusters(
    rows: list[dict[str, Any]], selected_for_early_stopping: set[str], limit: int | None,
) -> list[str]:
    all_clusters = {row["cluster_id"] for row in rows}
    missing_selected = selected_for_early_stopping - all_clusters
    if missing_selected:
        raise RuntimeError(
            "the A3 early-stopping cluster list contains unknown validation clusters: "
            f"{sorted(missing_selected)[:5]}"
        )
    available = sorted(all_clusters)
    if len(available) != 64:
        raise RuntimeError(f"expected 64 total validation clusters, found {len(available)}")
    if len(selected_for_early_stopping) != 24:
        raise RuntimeError(
            "the A3 run manifest must contain exactly 24 early-stopping validation clusters"
        )
    # The formal run manifest names the 24 selected clusters; the remaining
    # validation clusters are sealed for this confirmation.
    result = [cluster for cluster in available if cluster not in selected_for_early_stopping]
    if limit is None:
        if len(result) != 40:
            raise RuntimeError(f"expected 40 outer clusters, found {len(result)}")
        return result
    if limit < 2 or limit > len(result):
        raise ValueError(f"invalid smoke subset size {limit}; expected 2..{len(result)}")
    return result[:limit]


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
        channel_fusion=str(bridge_config.get("channel_fusion", "fixed_equal")),
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "bridge" not in payload:
        raise RuntimeError(f"checkpoint has no bridge state: {checkpoint_path}")
    bridge.load_state_dict(payload["bridge"], strict=True)
    bridge.to(model_device(backend.model)).eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    return bridge


@torch.no_grad()
def evaluate_channel(
    task: ReceiverCrossAttentionTask,
    example: PreparedExample,
    channel: str,
    donor_cluster: str,
) -> dict[str, Any]:
    prepared = task.prepare_semantic_channel(example, channel)
    embeddings, mask, positions, targets, _teacher, matched_states = prepared
    donor_states = task._states(donor_cluster)
    counterfactual_states = dict(matched_states)
    counterfactual_states[channel] = donor_states[channel]

    matched_logits = task._logits(embeddings, mask, positions, targets, matched_states)
    matched_nll = float(token_cross_entropy(matched_logits, targets).float().cpu())
    matched_logp = float(sequence_log_probability(matched_logits, targets).mean().cpu())
    del matched_logits
    counterfactual_logits = task._logits(
        embeddings, mask, positions, targets, counterfactual_states
    )
    counterfactual_logp = float(
        sequence_log_probability(counterfactual_logits, targets).mean().cpu()
    )
    del counterfactual_logits
    target_length = int(matched_states[channel].states.shape[1])
    donor_length = int(donor_states[channel].states.shape[1])
    return {
        "channel": channel,
        "donor_cluster_id": donor_cluster,
        "matched_logp": matched_logp,
        "counterfactual_logp": counterfactual_logp,
        "gap": matched_logp - counterfactual_logp,
        "matched_nll": matched_nll,
        "target_length": target_length,
        "donor_length": donor_length,
        "length_delta": donor_length - target_length,
    }


@torch.no_grad()
def alignment_retrieval(
    task: ReceiverCrossAttentionTask,
    examples: list[PreparedExample],
) -> dict[str, float]:
    """Compute contextual-teacher retrieval once per unique image cluster."""
    pairs_by_channel: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        name: [] for name in CHANNELS
    }
    for example in examples:
        pairs = task.semantic_alignment_pairs(example)
        for channel in CHANNELS:
            pairs_by_channel[channel].append(pairs[channel])
    result: dict[str, float] = {}
    for channel in CHANNELS:
        students = torch.cat([item[0] for item in pairs_by_channel[channel]], dim=0).float()
        teachers = torch.cat([item[1] for item in pairs_by_channel[channel]], dim=0).float()
        left = torch.nn.functional.normalize(students, dim=-1)
        right = torch.nn.functional.normalize(teachers, dim=-1)
        logits = left @ right.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        result[channel] = float(logits.argmax(-1).eq(labels).float().mean().cpu())
    return result


def summarize(
    details: list[dict[str, Any]],
    retrieval_by_seed: dict[str, dict[str, float]],
    *,
    bootstrap_seed: int,
    min_gap: float,
    min_fraction: float,
    min_retrieval: float,
) -> dict[str, Any]:
    by_seed_cluster_channel: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in details:
        by_seed_cluster_channel[(int(row["seed"]), row["cluster_id"], row["channel"])].append(
            float(row["gap"])
        )
    # One value per (seed, image cluster, channel); duplicate rows are an
    # implementation error and cannot be silently averaged.
    if any(len(values) != 1 for values in by_seed_cluster_channel.values()):
        raise RuntimeError("outer details contain duplicate seed/cluster/channel records")
    cluster_seed_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    seed_summary: dict[str, dict[str, dict[str, float]]] = {}
    for channel in CHANNELS:
        for seed in sorted({int(row["seed"]) for row in details}):
            values = [
                by_seed_cluster_channel[(seed, cluster, channel)][0]
                for cluster in sorted({row["cluster_id"] for row in details})
            ]
            for cluster, value in zip(
                sorted({row["cluster_id"] for row in details}), values
            ):
                cluster_seed_values[(cluster, channel)].append(value)
            seed_summary.setdefault(str(seed), {})[channel] = {
                "mean_gap": statistics.fmean(values),
                "fraction_gap_gt_0": sum(value > 0 for value in values) / len(values),
                "fraction_gap_gt_margin": sum(value > 0.2 for value in values) / len(values),
                "mean_nll": statistics.fmean(
                    float(row["matched_nll"])
                    for row in details
                    if int(row["seed"]) == seed and row["channel"] == channel
                ),
            }
    channel_summary: dict[str, Any] = {}
    for index, channel in enumerate(CHANNELS):
        cluster_values = [
            statistics.fmean(cluster_seed_values[(cluster, channel)])
            for cluster in sorted({row["cluster_id"] for row in details})
        ]
        interval = bootstrap_interval(cluster_values, seed=bootstrap_seed + index)
        seed_means = [seed_summary[str(seed)][channel]["mean_gap"] for seed in sorted(seed_summary)]
        channel_summary[channel] = {
            "clusters": len(cluster_values),
            "mean_gap": statistics.fmean(cluster_values),
            "median_gap": statistics.median(cluster_values),
            "fraction_gap_gt_0": sum(value > 0 for value in cluster_values) / len(cluster_values),
            "fraction_gap_gt_0_ci95": wilson_interval(
                sum(value > 0 for value in cluster_values), len(cluster_values)
            ),
            "fraction_gap_gt_0_2": sum(value > 0.2 for value in cluster_values) / len(cluster_values),
            "bootstrap_95ci": interval,
            "seed_mean_gap_values": seed_means,
            "seed_gap_mean": statistics.fmean(seed_means),
            "seed_gap_std": statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
            "mean_nll": statistics.fmean(
                float(row["matched_nll"])
                for row in details
                if row["channel"] == channel
            ),
            "donor_length_delta_max_abs": max(
                abs(int(row["length_delta"]))
                for row in details
                if row["channel"] == channel
            ),
        }
    point_checks = {
        channel: (
            channel_summary[channel]["mean_gap"] >= min_gap
            and channel_summary[channel]["fraction_gap_gt_0"] >= min_fraction
            and statistics.fmean(
                float(retrieval_by_seed[str(seed)][channel])
                for seed in sorted(retrieval_by_seed)
            ) >= min_retrieval
        )
        for channel in CHANNELS
    }
    ci_checks = {
        channel: channel_summary[channel]["bootstrap_95ci"][0] > 0
        for channel in CHANNELS
    }
    point_pass = all(point_checks.values())
    status = "outer_semantic_go" if point_pass and all(ci_checks.values()) else "outer_semantic_inconclusive"
    return {
        "status": status,
        "engineering_gate_pass": True,
        "clusters": len({row["cluster_id"] for row in details}),
        "seeds": sorted({int(row["seed"]) for row in details}),
        "statistical_unit": "image_cluster",
        "channel_summary": channel_summary,
        "retrieval_at_1_by_seed": retrieval_by_seed,
        "point_checks": point_checks,
        "bootstrap_ci_lower_bound_above_zero": ci_checks,
        "point_gate_pass": point_pass,
        "all_channel_ci_lower_bounds_above_zero": all(ci_checks.values()),
        "thresholds": {
            "min_channel_gap": min_gap,
            "min_fraction_gap_gt_0": min_fraction,
            "min_retrieval_at_1": min_retrieval,
        },
        "interpretation": (
            "Outer semantic confirmation is a mechanism gate, not a humorous-caption quality "
            "evaluation. The forward pass is deterministic for a fixed image/trace; repeated "
            "seeds are reported for provenance and should have near-zero seed variance. "
            "Inconclusive means no stable claim and no caption-stage unlock; it is not a method "
            "failure unless a pre-registered hard negative or engineering invariant fails."
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
        help="Only for a real-trace smoke; omitting it enforces all 40 sealed outer clusters.",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config["loss"].get("semantic_objective") != "channel_balanced_v3":
        raise RuntimeError("outer confirmation requires the channel-balanced A3 objective")
    if config["bridge"].get("channel_fusion") != "fixed_equal":
        raise RuntimeError("outer confirmation requires fixed_equal channel fusion")
    completion_path = checkpoint_path.parent / "complete.json"
    run_manifest_path = checkpoint_path.parent / "run_manifest.json"
    gate_path = checkpoint_path.parent / "semantic_gate.json"
    if not completion_path.is_file() or json.loads(completion_path.read_text()).get("status") != "complete":
        raise RuntimeError("A3 checkpoint has no validated complete.json")
    if not run_manifest_path.is_file():
        raise RuntimeError("A3 checkpoint is missing run_manifest.json")
    run_manifest = json.loads(run_manifest_path.read_text())
    if int(run_manifest.get("validation_clusters", -1)) != 24:
        raise RuntimeError("A3 run manifest does not describe the preregistered 24-cluster subset")
    selected_validation = set(run_manifest.get("validation_cluster_ids", []))
    dataset = ROOT / config["data"]["dataset"]
    trace_index_path = ROOT / config["data"]["trace_index"]
    validation_rows = read_jsonl(dataset / "validation.jsonl")
    outer_ids = outer_clusters(validation_rows, selected_validation, args.max_clusters)
    outer_rows = [row for row in validation_rows if row["cluster_id"] in set(outer_ids)]
    representatives = first_rows_by_cluster(outer_rows)
    if len(representatives) != len(outer_ids):
        raise RuntimeError("outer validation did not produce one representative row per cluster")
    traces = load_trace_index(trace_index_path)
    missing = sorted(set(outer_ids) - set(traces))
    if missing:
        raise RuntimeError(f"outer validation is missing Planner traces: {missing[:5]}")
    donor_rows = read_jsonl(dataset / "train.jsonl")
    fitted_clusters = set(run_manifest.get("train_cluster_ids", []))
    donor_rows = [row for row in donor_rows if row["cluster_id"] not in fitted_clusters]
    if not donor_rows:
        raise RuntimeError("A3 fitted-cluster exclusion removed all outer donor rows")
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
    if model_device(backend.model).type != "cuda":
        raise RuntimeError("outer semantic confirmation requires CUDA")
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
    )
    examples = [prepare_example(row, traces[row["cluster_id"]], seed=0) for row in representatives]
    task.cache_contextual_teachers(outer_ids)

    details: list[dict[str, Any]] = []
    retrieval_by_seed: dict[str, dict[str, float]] = {}
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        retrieval_by_seed[str(seed)] = alignment_retrieval(task, examples)
        for index, example in enumerate(examples, start=1):
            cluster_id = example.row["cluster_id"]
            for channel in CHANNELS:
                result = evaluate_channel(task, example, channel, donor_map[cluster_id][channel])
                detail = {
                    "seed": seed,
                    "evaluation_index": index,
                    "cluster_id": cluster_id,
                    "row_id": example.row["row_id"],
                    **result,
                }
                details.append(detail)
                print(json.dumps({"status": "evaluated", **detail}), flush=True)
            if model_device(backend.model).type == "cuda":
                torch.cuda.empty_cache()

    summary = summarize(
        details,
        retrieval_by_seed,
        bootstrap_seed=int(config["training"]["seed"]),
        min_gap=float(config["gate"]["min_channel_validation_gap"]),
        min_fraction=float(config["gate"]["min_channel_fraction_gap_gt_0"]),
        min_retrieval=float(config["gate"]["min_channel_retrieval_at_1"]),
    )
    summary.update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_completion_sha256": sha256(completion_path),
        "checkpoint_run_manifest_sha256": sha256(run_manifest_path),
        "checkpoint_semantic_gate_sha256": sha256(gate_path) if gate_path.is_file() else None,
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
        "git_commit": git_commit(),
        "validation_early_stopping_cluster_ids": sorted(selected_validation),
        "outer_cluster_ids": sorted(outer_ids),
        "outer_cluster_ids_sha256": hashlib.sha256(
            "\n".join(sorted(outer_ids)).encode()
        ).hexdigest(),
        "selection_policy": "all_validation_clusters_minus_A3_early_stopping_24",
        "subset_mode": "smoke" if args.max_clusters is not None else "sealed_outer_confirmation",
        "donor_policy": "length_matched_train_split_excluding_A3_fit_clusters_per_channel",
        "donor_diagnostics": donor_diagnostics,
        "excluded": ["caption_generation", "caption_judge", "DPO", "preference_learning"],
    })
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cluster_seed_channel_details.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in details)
    )
    (args.output / "outer_cluster_ids.json").write_text(
        json.dumps({"clusters": sorted(outer_ids), "count": len(outer_ids)}, indent=2) + "\n"
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "details_sha256": sha256(args.output / "cluster_seed_channel_details.jsonl"),
        "summary_sha256": sha256(args.output / "summary.json"),
        "outer_cluster_ids_sha256": summary["outer_cluster_ids_sha256"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "config_sha256": summary["config_sha256"],
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "trace_index_sha256": summary["trace_index_sha256"],
        "git_commit": summary["git_commit"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
