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

from humor_generator_v3.data.traces import read_jsonl
from humor_generator_v3.latent.bridges import LearnedLatentBridge, TypedLatentBridge, mean_embedding_norm
from humor_generator_v3.qwen_backend import QwenBackend, model_device
from humor_generator_v3.training.formal_bridge import (
    FrozenReceiverBridgeTask,
    cluster_balanced_rows,
    load_trace_index,
    mean_metrics,
    prepare_example,
    shuffled_cluster_map,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config["experiment"]["baseline"] not in {"learned_latent", "typed_learned_latent"}:
        raise ValueError("train_bridge only trains learned_latent or typed_learned_latent")
    if not config["model"].get("frozen", False):
        raise ValueError("formal bridge experiments require a frozen receiver")

    torch.manual_seed(int(config["training"]["seed"]))
    adapter = config["model"].get("adapter")
    backend = QwenBackend.load(
        config["model"]["name"],
        revision=config["model"]["revision"],
        adapter=None if adapter is None else ROOT / adapter,
        load_in_4bit=True,
    )
    device = model_device(backend.model)
    if device.type != "cuda":
        raise RuntimeError("formal bridge training requires CUDA")
    backend.model.eval()
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)

    width = int(backend.model.get_input_embeddings().weight.shape[1])
    target_norm = mean_embedding_norm(backend.model.get_input_embeddings().weight)
    bridge_args = {
        "bottleneck_dim": int(config["bridge"]["bottleneck_dim"]),
        "slots": int(config["bridge"]["slots_per_channel"]),
        "heads": int(config["bridge"]["heads"]),
        "target_norm": target_norm,
    }
    bridge = (
        TypedLatentBridge(width, width, **bridge_args)
        if config["experiment"]["baseline"] == "typed_learned_latent"
        else LearnedLatentBridge(width, width, **bridge_args)
    ).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        bridge.load_state_dict(state["bridge"])
    policy_trainable = sum(p.numel() for p in backend.model.parameters() if p.requires_grad)
    bridge_trainable = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    if policy_trainable != 0 or bridge_trainable == 0:
        raise RuntimeError("freeze/trainable-parameter contract failed")

    dataset = ROOT / config["data"]["dataset"]
    trace_path = ROOT / config["data"]["trace_index"]
    train_rows = read_jsonl(dataset / "train.jsonl")
    validation_rows = read_jsonl(dataset / "validation.jsonl")
    traces = load_trace_index(trace_path)
    required = {row["cluster_id"] for row in train_rows + validation_rows}
    missing = sorted(required - set(traces))
    if missing:
        raise RuntimeError(f"Planner traces missing for {len(missing)} clusters; first={missing[:5]}")

    seed = int(config["training"]["seed"])
    train_shuffle = shuffled_cluster_map(train_rows, seed=seed + 11)
    validation_shuffle = shuffled_cluster_map(validation_rows, seed=seed + 17)
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
    }
    (args.output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    best = float("inf")
    stale = 0
    global_step = 0
    accumulation = int(config["training"]["gradient_accumulation"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    log_path = args.output / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(config["training"]["epochs"])):
        bridge.train()
        rows = cluster_balanced_rows(train_rows, epoch=epoch, seed=seed)
        train_metrics = []
        for index, row in enumerate(rows):
            example = prepare_example(row, traces[row["cluster_id"]], seed=seed + epoch)
            metrics = task.backward_example(
                example,
                train_shuffle[row["cluster_id"]],
                loss_scale=1.0 / accumulation,
            )
            train_metrics.append(metrics)
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
        validation_metrics = []
        validation_selected = cluster_balanced_rows(validation_rows, epoch=0, seed=seed)
        for row in validation_selected:
            example = prepare_example(row, traces[row["cluster_id"]], seed=seed)
            validation_metrics.append(task.evaluate_example(
                example, validation_shuffle[row["cluster_id"]]
            ))
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train": mean_metrics(train_metrics),
            "validation": mean_metrics(validation_metrics),
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        checkpoint = {
            "bridge": {key: value.detach().cpu() for key, value in bridge.state_dict().items()},
            "epoch": epoch + 1,
            "global_step": global_step,
            "validation": record["validation"],
        }
        torch.save(checkpoint, args.output / f"checkpoint_epoch_{epoch + 1}.pt")
        score = record["validation"]["total"]
        if score < best - float(config["training"]["early_stopping_min_delta"]):
            best = score
            stale = 0
            torch.save(checkpoint, args.output / "best_bridge.pt")
        else:
            stale += 1
            if stale >= int(config["training"]["early_stopping_patience"]):
                break
    (args.output / "complete.json").write_text(json.dumps({
        "status": "complete",
        "best_validation_total": best,
        "epochs_completed": epoch + 1,
        "global_step": global_step,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
