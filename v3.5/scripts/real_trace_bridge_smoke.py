#!/usr/bin/env python3
"""One-step bridge smoke using two real, validated frozen-Planner traces."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import load_trace, read_jsonl
from humor_generator_v35.latent.bridges import (
    LearnedLatentBridge,
    TypedLatentBridge,
    mean_embedding_norm,
)
from humor_generator_v35.latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.formal_bridge import (
    FrozenReceiverBridgeTask,
    cluster_balanced_rows,
    hard_negative_cluster_map,
    load_trace_index,
    prepare_example,
)
from humor_generator_v35.training.cross_attention_bridge import ReceiverCrossAttentionTask
from humor_generator_v35.training.memory_safe import configure_frozen_receiver
from train_bridge import fixed_hash_sample_clusters


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    traces = load_trace_index(args.trace_index)
    rows = [
        row for row in read_jsonl(ROOT / config["data"]["dataset"] / "train.jsonl")
        if row["cluster_id"] in traces
    ]
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    rows = fixed_hash_sample_clusters(
        rows,
        config["training"].get("max_train_clusters"),
        seed=seed,
        split="train",
    )
    if len({row["cluster_id"] for row in rows}) < 2:
        raise RuntimeError("real trace smoke requires two clusters")
    # Cover both independent memory-risk axes in the fixed pilot subset while
    # preserving the engineering-smoke contract of at most two examples: raw
    # image size and full (untruncated) Planner memory.
    selected = cluster_balanced_rows(rows, epoch=0, seed=seed)
    def raw_pixels(row: dict) -> int:
        with Image.open(row["image"]) as image:
            return image.width * image.height
    stress = max(selected, key=lambda row: (raw_pixels(row), row["cluster_id"]))
    def latent_tokens(row: dict) -> int:
        record = traces[row["cluster_id"]]
        loaded = load_trace(ROOT / record["trace_path"], expected_sha256=record["trace_sha256"])
        return sum(int(item.states.shape[1]) for item in loaded.values())
    memory_stress = max(selected, key=lambda row: (latent_tokens(row), row["cluster_id"]))
    smoke_rows = []
    for row in (stress, memory_stress):
        if row["cluster_id"] not in {item["cluster_id"] for item in smoke_rows}:
            smoke_rows.append(row)
    negative_map, negative_diagnostics = hard_negative_cluster_map(rows, traces)

    adapter = config["model"].get("adapter")
    backend = QwenBackend.load(
        config["model"]["name"], revision=config["model"]["revision"],
        adapter=None if adapter is None else ROOT / adapter, load_in_4bit=True,
        min_visual_tokens=int(config["model"]["min_visual_tokens"]),
        max_visual_tokens=int(config["model"]["max_visual_tokens"]),
    )
    device = model_device(backend.model)
    backend.model.eval()
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    width = int(backend.model.get_input_embeddings().weight.shape[1])
    baseline = config["experiment"]["baseline"]
    if baseline == "receiver_cross_attention":
        bridge = ReceiverDrivenCrossAttentionBridge(
            width,
            width,
            layer_indices=[int(item) for item in config["bridge"]["layer_indices"]],
            bottleneck_dim=int(config["bridge"]["bottleneck_dim"]),
            heads=int(config["bridge"]["heads"]),
            gate_init=float(config["bridge"]["gate_init"]),
        ).to(device)
    else:
        common_bridge_args = {
            "bottleneck_dim": int(config["bridge"]["bottleneck_dim"]),
            "heads": int(config["bridge"]["heads"]),
            "target_norm": mean_embedding_norm(backend.model.get_input_embeddings().weight),
        }
        bridge = (
            TypedLatentBridge(
                width,
                width,
                slots=int(config["bridge"]["slots_per_channel"]),
                **common_bridge_args,
            )
            if baseline == "typed_learned_latent"
            else LearnedLatentBridge(
                width,
                width,
                slots=int(config["bridge"]["total_slots"]),
                **common_bridge_args,
            )
        ).to(device)
    freeze_report = configure_frozen_receiver(backend.model, bridge)
    task = (
        ReceiverCrossAttentionTask(
            backend, bridge, root=ROOT, trace_index=traces,
            loss_config=config["loss"],
            max_target_tokens=int(config["training"]["max_target_tokens"]),
            stage=str(config["training"]["stage"]),
        )
        if baseline == "receiver_cross_attention"
        else FrozenReceiverBridgeTask(
            backend, bridge, root=ROOT, trace_index=traces,
            loss_config=config["loss"],
            max_caption_tokens=int(config["training"]["max_caption_tokens"]),
        )
    )
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    before = {
        name: parameter.detach().float().clone()
        for name, parameter in bridge.named_parameters() if parameter.requires_grad
    }
    sample_reports = []
    metrics = None
    for row in smoke_rows:
        torch.cuda.reset_peak_memory_stats(device)
        example = prepare_example(row, traces[row["cluster_id"]], seed=seed)
        negative_cluster = negative_map[row["cluster_id"]]
        sample_metrics = task.backward_example(
            example, negative_cluster, loss_scale=1.0 / len(smoke_rows)
        )
        with Image.open(row["image"]) as image:
            size = [image.width, image.height]
        sample_reports.append({
            "cluster": row["cluster_id"],
            "row_id": row["row_id"],
            "image_size": size,
            "raw_image_pixels": size[0] * size[1],
            "full_latent_tokens": latent_tokens(row),
            "hard_negative_cluster": negative_cluster,
            "encode_diagnostics": backend.last_encode_diagnostics,
            "loss": sample_metrics,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        })
        metrics = sample_metrics
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0))
    optimizer.step()
    update_norm = float(sum(
        (parameter.detach().float() - before[name]).square().sum()
        for name, parameter in bridge.named_parameters() if parameter.requires_grad
    ).sqrt())
    if not all(torch.isfinite(torch.tensor(value)) for value in [*metrics.values(), gradient_norm, update_norm]):
        raise RuntimeError("non-finite real-trace smoke diagnostics")
    if update_norm <= 0:
        raise RuntimeError("bridge did not update")
    report = {
        "status": "pass",
        "scientific_training": False,
        "real_planner_traces": True,
        "clusters": [row["cluster_id"] for row in smoke_rows],
        "selection": "max_raw_pixels_plus_max_full_latent_tokens_at_most_two_examples",
        "communication_interface": (
            "receiver_driven_full_state_cross_attention_no_soft_prefix"
            if baseline == "receiver_cross_attention" else "input_soft_prefix"
        ),
        "samples": sample_reports,
        "hard_negative_diagnostics": negative_diagnostics,
        "policy_trainable_parameters": freeze_report.policy_trainable,
        "bridge_trainable_parameters": freeze_report.bridge_trainable,
        "gradient_checkpointing": freeze_report.gradient_checkpointing,
        "use_cache": freeze_report.use_cache,
        "min_visual_tokens": int(config["model"]["min_visual_tokens"]),
        "max_visual_tokens": int(config["model"]["max_visual_tokens"]),
        "gradient_norm": gradient_norm,
        "update_norm": update_norm,
        "last_loss": metrics,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
