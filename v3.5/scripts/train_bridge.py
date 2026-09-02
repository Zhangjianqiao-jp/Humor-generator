#!/usr/bin/env python3
"""Formal learned/typed bridge training; both 7B policies remain frozen."""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import read_jsonl
from humor_generator_v35.latent.bridges import LearnedLatentBridge, TypedLatentBridge, mean_embedding_norm
from humor_generator_v35.latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.formal_bridge import (
    FrozenReceiverBridgeTask,
    cluster_balanced_rows,
    hard_negative_cluster_map,
    load_trace_index,
    mean_metrics,
    prepare_example,
)
from humor_generator_v35.training.cross_attention_bridge import ReceiverCrossAttentionTask
from humor_generator_v35.training.memory_safe import configure_frozen_receiver


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def fixed_hash_sample_clusters(
    rows: list[dict], limit: int | None, *, seed: int, split: str,
) -> list[dict]:
    """Select an order-invariant, seeded cluster subset without ID-order bias."""
    if limit is None:
        return rows
    if limit < 2:
        raise ValueError("a bridge subset needs at least two image clusters")
    clusters = sorted({row["cluster_id"] for row in rows})
    if limit > len(clusters):
        raise ValueError(f"requested {limit} clusters from a {len(clusters)}-cluster {split} split")
    selected = sorted(
        clusters,
        key=lambda cluster: hashlib.sha256(
            f"v35-pilot:{seed}:{split}:{cluster}".encode()
        ).digest(),
    )[:limit]
    allowed = set(selected)
    return [row for row in rows if row["cluster_id"] in allowed]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-bridge", type=Path,
        help="Initialize bridge weights only (for semantic-reconstruction -> caption stage).",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if args.resume and args.init_bridge:
        raise ValueError("--resume and --init-bridge are mutually exclusive")
    if config["experiment"]["baseline"] not in {
        "learned_latent", "typed_learned_latent", "receiver_cross_attention"
    }:
        raise ValueError("unsupported trainable bridge baseline")
    if not config["model"].get("frozen", False):
        raise ValueError("formal bridge experiments require a frozen receiver")

    torch.manual_seed(int(config["training"]["seed"]))
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
        raise RuntimeError("formal bridge training requires CUDA")
    backend.model.eval()
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)

    width = int(backend.model.get_input_embeddings().weight.shape[1])
    target_norm = mean_embedding_norm(backend.model.get_input_embeddings().weight)
    common_bridge_args = {
        "bottleneck_dim": int(config["bridge"]["bottleneck_dim"]),
        "heads": int(config["bridge"]["heads"]),
        "target_norm": target_norm,
    }
    baseline = config["experiment"]["baseline"]
    if baseline == "receiver_cross_attention":
        bridge = ReceiverDrivenCrossAttentionBridge(
            width, width,
            layer_indices=[int(value) for value in config["bridge"]["layer_indices"]],
            bottleneck_dim=int(config["bridge"]["bottleneck_dim"]),
            heads=int(config["bridge"]["heads"]),
            gate_init=float(config["bridge"].get("gate_init", 0.1)),
        ).to(device)
    else:
        bridge = (
            TypedLatentBridge(
                width, width,
                slots=int(config["bridge"]["slots_per_channel"]),
                **common_bridge_args,
            )
            if baseline == "typed_learned_latent"
            else LearnedLatentBridge(
                width, width,
                slots=int(config["bridge"]["total_slots"]),
                **common_bridge_args,
            )
        ).to(device)
    freeze_report = configure_frozen_receiver(backend.model, bridge)
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=True)
        bridge.load_state_dict(resume_state["bridge"])
    elif args.init_bridge:
        initial = torch.load(args.init_bridge, map_location="cpu", weights_only=True)
        if "bridge" not in initial:
            raise RuntimeError("initial checkpoint has no bridge state")
        bridge.load_state_dict(initial["bridge"])
    policy_trainable = freeze_report.policy_trainable
    bridge_trainable = freeze_report.bridge_trainable
    if policy_trainable != 0 or bridge_trainable == 0:
        raise RuntimeError("freeze/trainable-parameter contract failed")

    dataset = ROOT / config["data"]["dataset"]
    trace_path = ROOT / config["data"]["trace_index"]
    train_rows = read_jsonl(dataset / "train.jsonl")
    validation_rows = read_jsonl(dataset / "validation.jsonl")
    seed = int(config["training"]["seed"])
    train_rows = fixed_hash_sample_clusters(
        train_rows, config["training"].get("max_train_clusters"), seed=seed, split="train"
    )
    validation_rows = fixed_hash_sample_clusters(
        validation_rows, config["training"].get("max_validation_clusters"),
        seed=seed, split="validation",
    )
    traces = load_trace_index(trace_path)
    required = {row["cluster_id"] for row in train_rows + validation_rows}
    missing = sorted(required - set(traces))
    if missing:
        raise RuntimeError(f"Planner traces missing for {len(missing)} clusters; first={missing[:5]}")

    train_shuffle, train_negative_diagnostics = hard_negative_cluster_map(train_rows, traces)
    validation_shuffle, validation_negative_diagnostics = hard_negative_cluster_map(
        validation_rows, traces
    )
    if baseline == "receiver_cross_attention":
        task = ReceiverCrossAttentionTask(
            backend, bridge, root=ROOT, trace_index=traces,
            loss_config=config["loss"],
            max_target_tokens=int(config["training"]["max_target_tokens"]),
            stage=str(config["training"]["stage"]),
        )
    else:
        task = FrozenReceiverBridgeTask(
            backend,
            bridge,
            root=ROOT,
            trace_index=traces,
            loss_config=config["loss"],
            max_caption_tokens=int(config["training"]["max_caption_tokens"]),
        )
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    if resume_state is not None:
        if "optimizer" not in resume_state:
            raise RuntimeError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(resume_state["optimizer"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.yaml").write_text(config_path.read_text())
    run_manifest = {
        "schema_version": 1,
        "git_commit": git_commit(),
        "config_sha256": sha256(config_path),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "trace_index_sha256": sha256(trace_path),
        "policy_trainable_parameters": policy_trainable,
        "bridge_trainable_parameters": bridge_trainable,
        "receiver_embedding_mean_norm": target_norm,
        "device": str(device),
        "gradient_checkpointing": freeze_report.gradient_checkpointing,
        "use_cache": freeze_report.use_cache,
        "min_visual_tokens": int(config["model"]["min_visual_tokens"]),
        "max_visual_tokens": int(config["model"]["max_visual_tokens"]),
        "negative_policy": "description_tfidf_nearest_same-source_different-conflict",
        "train_negative_diagnostics": train_negative_diagnostics,
        "validation_negative_diagnostics": validation_negative_diagnostics,
        "train_clusters": len({row["cluster_id"] for row in train_rows}),
        "validation_clusters": len({row["cluster_id"] for row in validation_rows}),
        "train_cluster_ids": sorted({row["cluster_id"] for row in train_rows}),
        "validation_cluster_ids": sorted({row["cluster_id"] for row in validation_rows}),
        "subset_selection": "sha256(v35-pilot:seed:split:cluster_id)",
        "training_stage": config["training"].get("stage", "caption"),
        "communication_interface": (
            "receiver_driven_full_state_cross_attention_no_soft_prefix"
            if baseline == "receiver_cross_attention" else "input_soft_prefix"
        ),
        "initial_bridge_checkpoint": None if args.init_bridge is None else str(args.init_bridge.resolve()),
        "train_cluster_ids_sha256": hashlib.sha256(
            "\n".join(sorted({row["cluster_id"] for row in train_rows})).encode()
        ).hexdigest(),
        "validation_cluster_ids_sha256": hashlib.sha256(
            "\n".join(sorted({row["cluster_id"] for row in validation_rows})).encode()
        ).hexdigest(),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    best = float(resume_state.get("best_validation_total", float("inf"))) if resume_state else float("inf")
    stale = int(resume_state.get("stale_epochs", 0)) if resume_state else 0
    global_step = int(resume_state.get("global_step", 0)) if resume_state else 0
    start_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    accumulation = int(config["training"]["gradient_accumulation"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    log_path = args.output / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    total_epochs = int(config["training"]["epochs"])
    if start_epoch >= total_epochs:
        raise RuntimeError("resume checkpoint already reached configured epochs")
    for epoch in range(start_epoch, total_epochs):
        bridge.train()
        rows = cluster_balanced_rows(train_rows, epoch=epoch, seed=seed)
        train_metrics = []
        for index, row in enumerate(rows):
            window_start = (index // accumulation) * accumulation
            window_size = min(accumulation, len(rows) - window_start)
            example = prepare_example(row, traces[row["cluster_id"]], seed=seed + epoch)
            current = {
                "status": "preparing_example",
                "epoch": epoch + 1,
                "examples_in_epoch": index + 1,
                "examples_per_epoch": len(rows),
                "cluster_id": row["cluster_id"],
                "row_id": row["row_id"],
                "image": row["image"],
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            (args.output / "current_example.json").write_text(
                json.dumps(current, indent=2) + "\n"
            )
            print(json.dumps(current), flush=True)
            metrics = task.backward_example(
                example,
                train_shuffle[row["cluster_id"]],
                loss_scale=1.0 / window_size,
            )
            train_metrics.append(metrics)
            current.update({
                "status": "example_complete",
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "metrics": metrics,
            })
            (args.output / "current_example.json").write_text(
                json.dumps(current, indent=2) + "\n"
            )
            if (index + 1) % int(config["training"].get("log_every_examples", 10)) == 0:
                progress = {
                    "status": "training",
                    "epoch": epoch + 1,
                    "examples_in_epoch": index + 1,
                    "examples_per_epoch": len(rows),
                    "global_step": global_step,
                    "running_train": mean_metrics(train_metrics[-10:]),
                }
                (args.output / "progress.json").write_text(
                    json.dumps(progress, indent=2) + "\n"
                )
                print(json.dumps(progress), flush=True)
            if (index + 1) % accumulation == 0 or index + 1 == len(rows):
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    bridge.parameters(), float(config["training"]["max_grad_norm"])
                ))
                if not torch.isfinite(torch.tensor(gradient_norm)):
                    raise RuntimeError("non-finite bridge gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        bridge.eval()
        validation_by_cluster: dict[str, list[dict[str, float]]] = {}
        for row in validation_rows:
            example = prepare_example(row, traces[row["cluster_id"]], seed=seed)
            validation_by_cluster.setdefault(row["cluster_id"], []).append(task.evaluate_example(
                example, validation_shuffle[row["cluster_id"]]
            ))
        validation_metrics = [
            mean_metrics(validation_by_cluster[cluster])
            for cluster in sorted(validation_by_cluster)
        ]
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": mean_metrics(train_metrics),
            "validation": mean_metrics(validation_metrics),
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        score = record["validation"]["total"]
        improved = score < best - float(config["training"]["early_stopping_min_delta"])
        if improved:
            best = score
            stale = 0
        else:
            stale += 1
        checkpoint = {
            "bridge": {key: value.detach().cpu() for key, value in bridge.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "validation": record["validation"],
            "best_validation_total": best,
            "stale_epochs": stale,
        }
        torch.save(checkpoint, args.output / f"checkpoint_epoch_{epoch + 1}.pt")
        if improved:
            torch.save(checkpoint, args.output / "best_bridge.pt")
        elif stale >= int(config["training"]["early_stopping_patience"]):
            break
    (args.output / "complete.json").write_text(json.dumps({
        "status": "complete",
        "best_validation_total": best,
        "epochs_completed": epoch + 1,
        "global_step": global_step,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
