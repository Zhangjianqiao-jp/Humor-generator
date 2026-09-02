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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import load_trace, plan_from_record, read_jsonl
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


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must have equal length >= 2")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = sum((a - left_mean) ** 2 for a in left)
    right_scale = sum((b - right_mean) ** 2 for b in right)
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale) ** 0.5


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    """Two-sided Wilson interval for a descriptive cluster-level proportion."""
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * ((proportion * (1 - proportion) / total) + z**2 / (4 * total**2)) ** 0.5
        / denominator
    )
    return [max(0.0, centre - half_width), min(1.0, centre + half_width)]


def _cluster_descriptions(
    rows: list[dict[str, Any]], clusters: list[str],
) -> dict[str, str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["cluster_id"] in clusters:
            grouped[row["cluster_id"]].add(str(row["standard_description"]))
    missing = sorted(set(clusters) - set(grouped))
    if missing:
        raise ValueError(f"missing descriptions for clusters: {missing[:5]}")
    return {cluster: " ".join(sorted(grouped[cluster])) for cluster in clusters}


def _trace_lengths(
    root: Path,
    trace_index: dict[str, dict[str, Any]],
    clusters: list[str],
) -> dict[str, dict[str, int]]:
    """Read only shape metadata, validating each donor trace hash."""
    result: dict[str, dict[str, int]] = {}
    for cluster in clusters:
        record = trace_index[cluster]
        loaded = load_trace(
            root / record["trace_path"], expected_sha256=record["trace_sha256"]
        )
        result[cluster] = {
            channel: int(loaded[channel].states.shape[1]) for channel in CHANNELS
        }
    return result


def _conflict_signatures(
    trace_index: dict[str, dict[str, Any]], clusters: list[str],
) -> dict[str, tuple[str, ...]]:
    return {
        cluster: tuple(
            sorted(
                item.render().casefold()
                for item in plan_from_record(trace_index[cluster]["plan"]).conflicts
            )
        )
        for cluster in clusters
    }


def length_matched_channel_donors(
    target_rows: list[dict[str, Any]],
    donor_rows: list[dict[str, Any]],
    *,
    root: Path,
    trace_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Choose per-channel hard negatives without changing channel length.

    The legacy v1 bridge applies one softmax over the concatenated memory.  A
    donor with a different number of tokens changes the denominator even when
    its semantic content is unrelated.  The primary re-evaluation therefore
    selects a donor from the non-test ``train`` split, first minimizing the
    absolute token-length difference for the swapped channel, then preferring a
    same-source donor and maximizing description TF-IDF similarity.  Most
    target/channel pairs have exact-length matches; residual differences are
    recorded rather than hidden.
    """
    target_clusters = sorted({row["cluster_id"] for row in target_rows})
    donor_clusters = sorted({row["cluster_id"] for row in donor_rows})
    if not target_clusters or not donor_clusters:
        raise ValueError("channel donor selection requires non-empty target and donor clusters")
    if set(target_clusters) & set(donor_clusters):
        raise ValueError("target and donor clusters overlap")
    for cluster in donor_clusters:
        if trace_index[cluster].get("split") != "train":
            raise ValueError(f"channel donor is not from train split: {cluster}")

    all_clusters = target_clusters + donor_clusters
    descriptions = _cluster_descriptions(target_rows + donor_rows, all_clusters)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([descriptions[cluster] for cluster in all_clusters])
    target_matrix = matrix[: len(target_clusters)]
    donor_matrix = matrix[len(target_clusters) :]
    similarity = linear_kernel(target_matrix, donor_matrix)
    signatures = _conflict_signatures(trace_index, all_clusters)
    lengths = _trace_lengths(root, trace_index, all_clusters)
    source_sets: dict[str, set[str]] = {}
    for cluster in all_clusters:
        source_sets[cluster] = {
            str(row.get("dataset", ""))
            for row in target_rows + donor_rows
            if row["cluster_id"] == cluster
        }

    result: dict[str, dict[str, str]] = {}
    per_channel: dict[str, dict[str, float]] = {
        channel: {"exact_length": 0.0, "length_delta_sum": 0.0}
        for channel in CHANNELS
    }
    for left, target in enumerate(target_clusters):
        candidates = [
            right for right, donor in enumerate(donor_clusters)
            if signatures[donor] != signatures[target]
        ]
        if not candidates:
            raise RuntimeError(f"no different-conflict train donor for {target}")
        result[target] = {}
        for channel in CHANNELS:
            target_length = lengths[target][channel]
            min_delta = min(
                abs(lengths[donor_clusters[right]][channel] - target_length)
                for right in candidates
            )
            length_pool = [
                right for right in candidates
                if abs(lengths[donor_clusters[right]][channel] - target_length) == min_delta
            ]
            # Length is the primary confounder for v1's single softmax.  Only
            # after fixing the minimum delta do we prefer a same-source donor.
            shared_length_pool = [
                right for right in length_pool
                if source_sets[target] & source_sets[donor_clusters[right]]
            ]
            ranking_pool = shared_length_pool or length_pool
            right = max(
                ranking_pool,
                key=lambda value: (float(similarity[left, value]), donor_clusters[value]),
            )
            donor = donor_clusters[right]
            result[target][channel] = donor
            per_channel[channel]["exact_length"] += float(min_delta == 0)
            per_channel[channel]["length_delta_sum"] += float(min_delta)
    diagnostics = {
        "policy": (
            "train_split_conflict_mismatched_description_nearest_length_matched_per_channel_"
            "excluding_checkpoint_fit_clusters"
        ),
        "target_clusters": float(len(target_clusters)),
        "donor_pool_clusters": float(len(donor_clusters)),
        "same_source_preference": True,
        "channels": {
            channel: {
                "exact_length_fraction": values["exact_length"] / len(target_clusters),
                "mean_absolute_length_delta": values["length_delta_sum"] / len(target_clusters),
            }
            for channel, values in per_channel.items()
        },
    }
    return result, diagnostics


def checkpoint_training_clusters(checkpoint_path: Path) -> set[str]:
    """Return clusters used by the historical bridge, when its manifest exists."""
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        return set()
    manifest = json.loads(manifest_path.read_text())
    return {str(cluster) for cluster in manifest.get("train_cluster_ids", [])}


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
    channel_donors: dict[str, str],
    full_plan_donor: str,
) -> dict[str, Any]:
    """Measure full-target log-probability under one-channel swaps.

    The image, prompt, target and the other two memory channels remain fixed.
    This makes the comparison valid for old full-plan checkpoints while still
    exposing which channel, if any, affects the Receiver's native output path.
    """
    example = prepare_example(row, task.trace_index[row["cluster_id"]], seed=0)
    prepared = task.prepare(example)
    embeddings, mask, positions, targets, _teacher_cpu, matched_states = prepared
    donor_clusters = set(channel_donors.values()) | {full_plan_donor}
    donor_states_by_cluster = {
        cluster: task._states(cluster) for cluster in sorted(donor_clusters)
    }

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
        "matched_logp": matched_logp,
        **matched_diag,
    }
    swapped_states = dict(matched_states)
    all_swapped = dict(donor_states_by_cluster[full_plan_donor])
    all_logp, _ = score(all_swapped)
    row_out["full_plan_swap_logp"] = all_logp
    row_out["full_plan_swap_gap"] = matched_logp - all_logp
    for channel in CHANNELS:
        donor_cluster = channel_donors[channel]
        donor_states = donor_states_by_cluster[donor_cluster]
        swapped_states[channel] = donor_states[channel]
        counterfactual_logp, _ = score(swapped_states)
        row_out[f"counterfactual_logp_{channel}"] = counterfactual_logp
        row_out[f"gap_{channel}"] = matched_logp - counterfactual_logp
        row_out[f"donor_cluster_id_{channel}"] = donor_cluster
        row_out[f"target_length_{channel}"] = int(matched_states[channel].states.shape[1])
        row_out[f"donor_length_{channel}"] = int(donor_states[channel].states.shape[1])
        row_out[f"length_delta_{channel}"] = (
            row_out[f"donor_length_{channel}"] - row_out[f"target_length_{channel}"]
        )
        row_out[f"donor_semantics_sha256_{channel}"] = hashlib.sha256(
            donor_states[channel].semantics.encode()
        ).hexdigest()
        swapped_states[channel] = matched_states[channel]
    row_out["full_plan_donor_cluster_id"] = full_plan_donor
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
            "fraction_gap_gt_0_ci95": wilson_interval(
                sum(value > 0 for value in values), len(values)
            ),
            "fraction_gap_gt_0_02": sum(value > 0.02 for value in values) / len(values),
            "fraction_gap_gt_0_02_ci95": wilson_interval(
                sum(value > 0.02 for value in values), len(values)
            ),
            "bootstrap_95ci": intervals[channel],
            "max_abs_length_delta": max(
                abs(int(row[f"length_delta_{channel}"])) for row in rows
            ),
            "gap_vs_abs_length_delta_pearson": pearson_correlation(
                values,
                [abs(int(row[f"length_delta_{channel}"])) for row in rows],
            ),
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
    model_manifest_path = ROOT / "manifests/local_qwen2_5_vl_7b.json"
    adapter_manifest_path = ROOT / "manifests/frozen_7b_adapters.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    adapter_manifest = json.loads(adapter_manifest_path.read_text())
    if (
        config["model"]["name"] != model_manifest["model_id"]
        or config["model"]["revision"] != model_manifest["revision"]
    ):
        raise RuntimeError("config model identity does not match pinned local model manifest")
    configured_adapter = config["model"].get("adapter")
    expected_adapter = adapter_manifest["adapters"]["generator_sft"]["local_path"]
    if configured_adapter != expected_adapter:
        raise RuntimeError(
            f"config adapter {configured_adapter!r} does not match frozen generator adapter "
            f"{expected_adapter!r}"
        )
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
    donor_rows_all = read_jsonl(dataset / "train.jsonl")
    fitted_clusters = checkpoint_training_clusters(checkpoint_path)
    checkpoint_run_manifest = checkpoint_path.parent / "run_manifest.json"
    donor_rows = [
        row for row in donor_rows_all if row["cluster_id"] not in fitted_clusters
    ]
    if not donor_rows:
        raise RuntimeError("checkpoint training-cluster exclusion removed every donor")
    traces = load_trace_index(trace_path)
    cluster_ids = {row["cluster_id"] for row in representatives}
    missing = sorted(cluster_ids - set(traces))
    if missing:
        raise RuntimeError(f"missing traces for clusters: {missing[:5]}")
    full_plan_donors, full_plan_donor_diagnostics = hard_negative_cluster_map(
        representatives, traces
    )
    channel_donors, channel_donor_diagnostics = length_matched_channel_donors(
        representatives,
        donor_rows,
        root=ROOT,
        trace_index=traces,
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
        result = evaluate_cluster(
            task,
            row,
            channel_donors[row["cluster_id"]],
            full_plan_donors[row["cluster_id"]],
        )
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
        "selection_seed": int(config["training"]["seed"]),
        "bootstrap_seed": int(config["training"]["seed"]),
        "donor_pool_excluded_checkpoint_train_clusters": sorted(fitted_clusters),
        "donor_pool_excluded_count": len(fitted_clusters),
        "checkpoint_run_manifest_sha256": (
            sha256(checkpoint_run_manifest) if checkpoint_run_manifest.is_file() else None
        ),
        "trace_input_manifest_sha256": sha256(dataset / "trace_inputs.jsonl"),
        "homer_prompt_source_sha256": sha256(ROOT / "src/humor_generator_v35/homer/prompts.py"),
        "receiver_prompt_source_sha256": sha256(
            ROOT / "src/humor_generator_v35/training/cross_attention_bridge.py"
        ),
        "frozen_adapter_manifest_sha256": sha256(ROOT / "manifests/frozen_7b_adapters.json"),
        "local_model_manifest_sha256": sha256(ROOT / "manifests/local_qwen2_5_vl_7b.json"),
        "donor_policy": channel_donor_diagnostics["policy"],
        "channel_donor_diagnostics": channel_donor_diagnostics,
        "full_plan_donor_policy": "description_tfidf_nearest_same-source_different-conflict_within_pilot_clusters",
        "full_plan_donor_diagnostics": full_plan_donor_diagnostics,
        "evaluation_protocol": (
            "full-target_same-image_single-channel_counterfactual; "
            "primary_channel_swaps_length-matched_train_donors"
        ),
        "excluded": ["sealed_test", "caption_generation", "retraining"],
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "method": method,
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "config_sha256": summary["config_sha256"],
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "trace_input_manifest_sha256": summary["trace_input_manifest_sha256"],
        "frozen_adapter_manifest_sha256": summary["frozen_adapter_manifest_sha256"],
        "local_model_manifest_sha256": summary["local_model_manifest_sha256"],
        "checkpoint_run_manifest_sha256": summary["checkpoint_run_manifest_sha256"],
        "homer_prompt_source_sha256": summary["homer_prompt_source_sha256"],
        "receiver_prompt_source_sha256": summary["receiver_prompt_source_sha256"],
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
