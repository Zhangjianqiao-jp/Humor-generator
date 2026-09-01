#!/usr/bin/env python
"""Memory-efficient, image-conditioned offline DPO for a multimodal captioner.

Reference log probabilities are computed beforehand with the frozen SFT
adapter. This keeps a single policy model resident during DPO training and avoids
an approximate or reference-free objective.
"""
from __future__ import annotations

import math
import json
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler, set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_loader import load_qwen_vl_with_lora
from src.preference.cluster_metrics import summarize_image_clusters
from src.preference.losses import preference_loss
from src.training.dpo_dataset import (
    DPOCollator,
    PreferenceDataset,
    model_inputs_from_batch,
    preference_sampling_dataset,
    sequence_logps,
)


def device_for(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def disable_use_cache(model: Any) -> None:
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "use_cache"):
            module.config.use_cache = False


def preference_metrics(
    model: Any,
    batch: dict[str, Any],
    device: torch.device,
    objective_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float], list[dict[str, Any]]]:
    outputs = model(**model_inputs_from_batch(batch, device))
    logps, token_counts = sequence_logps(outputs.logits, batch["labels"].to(device))
    n = int(batch["num_pairs"])
    chosen, rejected = logps[:n], logps[n:]
    ref_chosen = batch.get("reference_chosen_logps")
    ref_rejected = batch.get("reference_rejected_logps")
    result = preference_loss(
        objective=str(objective_cfg.get("name", "dpo")),
        chosen_logp=chosen,
        rejected_logp=rejected,
        chosen_tokens=token_counts[:n],
        rejected_tokens=token_counts[n:],
        beta=float(objective_cfg.get("beta", 0.1)),
        reference_chosen_logp=ref_chosen.to(device) if ref_chosen is not None else None,
        reference_rejected_logp=ref_rejected.to(device) if ref_rejected is not None else None,
        simpo_gamma=float(objective_cfg.get("gamma", 0.5)),
        anchor_weight=float(objective_cfg.get("anchor_weight", 0.1)),
        anchor_length_normalize=bool(objective_cfg.get("anchor_length_normalize", True)),
    )
    with torch.no_grad():
        policy_accuracy = (chosen > rejected).float().mean()
        reward_accuracy = (result.preference_logits > 0).float().mean()
    summary = {
        "policy_accuracy": float(policy_accuracy.detach().cpu()),
        "reward_accuracy": float(reward_accuracy.detach().cpu()),
        "chosen_logp": float(chosen.mean().detach().cpu()),
        "rejected_logp": float(rejected.mean().detach().cpu()),
        "chosen_logp_per_token": float((chosen / token_counts[:n].clamp_min(1)).mean().detach().cpu()),
        "rejected_logp_per_token": float((rejected / token_counts[n:].clamp_min(1)).mean().detach().cpu()),
        "chosen_reward": float(result.chosen_reward.mean().detach().cpu()),
        "rejected_reward": float(result.rejected_reward.mean().detach().cpu()),
        "reward_margin": float((result.chosen_reward - result.rejected_reward).mean().detach().cpu()),
    }
    pair_rows = []
    for index in range(n):
        pair_rows.append(
            {
                "pair_id": str(batch["pair_ids"][index]),
                "image_id": str(batch["image_ids"][index]),
                "loss": float(result.pair_losses[index].detach().cpu()),
                "policy_accuracy": float((chosen[index] > rejected[index]).detach().cpu()),
                "reward_accuracy": float((result.preference_logits[index] > 0).detach().cpu()),
                "chosen_logp": float(chosen[index].detach().cpu()),
                "rejected_logp": float(rejected[index].detach().cpu()),
                "chosen_logp_per_token": float(
                    (chosen[index] / token_counts[index].clamp_min(1)).detach().cpu()
                ),
                "rejected_logp_per_token": float(
                    (rejected[index] / token_counts[n + index].clamp_min(1)).detach().cpu()
                ),
                "chosen_reward": float(result.chosen_reward[index].detach().cpu()),
                "rejected_reward": float(result.rejected_reward[index].detach().cpu()),
                "reward_margin": float(
                    (result.chosen_reward[index] - result.rejected_reward[index]).detach().cpu()
                ),
            }
        )
    return result.loss, summary, pair_rows


CLUSTER_METRICS = (
    "loss",
    "policy_accuracy",
    "reward_accuracy",
    "chosen_logp",
    "rejected_logp",
    "chosen_logp_per_token",
    "rejected_logp_per_token",
    "chosen_reward",
    "rejected_reward",
    "reward_margin",
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def evaluate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    objective_cfg: dict[str, Any],
    *,
    artifacts_dir: Path | None = None,
    step: int | None = None,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260828,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {"loss": 0.0}
    pair_rows: list[dict[str, Any]] = []
    count = 0
    for batch in loader:
        loss, metrics, details = preference_metrics(model, batch, device, objective_cfg)
        n = int(batch["num_pairs"])
        totals["loss"] += float(loss.detach().cpu()) * n
        for key, value in metrics.items():
            totals.setdefault(key, 0.0)
            totals[key] += value * n
        count += n
        pair_rows.extend(details)
    if was_training:
        model.train()
    if count == 0:
        raise RuntimeError("DPO validation loader is empty.")
    values = {f"eval_{key}": value / count for key, value in totals.items()}
    cluster_report = summarize_image_clusters(
        pair_rows,
        CLUSTER_METRICS,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed + int(step or 0),
    )
    for metric, summary in cluster_report["metrics"].items():
        values[f"eval_image_mean_{metric}"] = float(summary["image_mean"])
        values[f"eval_image_median_{metric}"] = float(summary["image_median"])
        low, high = summary["image_cluster_bootstrap_95ci"]
        values[f"eval_image_ci_low_{metric}"] = float(low)
        values[f"eval_image_ci_high_{metric}"] = float(high)
    if artifacts_dir is not None and step is not None:
        stem = f"validation_step_{step:06d}"
        write_jsonl(artifacts_dir / f"{stem}_pairs.jsonl", pair_rows)
        atomic_write_json(
            artifacts_dir / f"{stem}_image_cluster_summary.json",
            {"step": step, **cluster_report},
        )
    return values


def save_adapter(model: Any, processor: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    print(f"[checkpoint] saved LoRA adapter: {path}")


def append_metric(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def configured_eval_milestones(training: dict[str, Any], update_steps: int) -> set[int]:
    configured = training.get("eval_milestones") or []
    milestones = {int(step) for step in configured if 0 < int(step) <= update_steps}
    milestones.add(update_steps)
    return milestones


def is_metric_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"early_stopping.mode must be 'min' or 'max', got {mode!r}")


def main() -> None:
    parser = ArgumentParser(description="Train a compact-conditioned Qwen2.5-VL captioner with offline DPO.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_dpo_newyorker_compact_3b.yaml"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--debug-one-step", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    set_seed(int(training.get("seed", 42)))
    output = config["output"]
    output_dir = Path(output["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    model_cfg = config["model"]
    model, processor = load_qwen_vl_with_lora(
        model_name=model_cfg["model_name"],
        lora_rank=int(model_cfg["lora"]["rank"]),
        lora_alpha=int(model_cfg["lora"]["alpha"]),
        lora_dropout=float(model_cfg["lora"]["dropout"]),
        target_modules=list(model_cfg["lora"]["target_modules"]),
        bias=str(model_cfg["lora"].get("bias", "none")),
        device_map=str(model_cfg.get("device_map", "auto")),
        torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        adapter_path=Path(model_cfg["sft_adapter_dir"]),
        is_trainable=True,
        image_min_pixels=config["data"].get("image_min_pixels"),
        image_max_pixels=config["data"].get("image_max_pixels"),
        load_in_4bit=bool(model_cfg.get("quantization", {}).get("load_in_4bit", False)),
        bnb_4bit_quant_type=str(model_cfg.get("quantization", {}).get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(
            model_cfg.get("quantization", {}).get("bnb_4bit_use_double_quant", True)
        ),
        new_adapter_name=(
            str(model_cfg.get("preference_adapter_name", "preference"))
            if str(model_cfg.get("adapter_strategy", "continue_sft")) == "separate_preference"
            else None
        ),
    )
    if bool(training.get("gradient_checkpointing", True)):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        disable_use_cache(model)
    device = device_for(model)
    data = config["data"]
    collator = DPOCollator(
        processor,
        max_seq_len=int(data.get("max_seq_len", 768)),
        image_min_pixels=data.get("image_min_pixels"),
        image_max_pixels=data.get("image_max_pixels"),
    )
    train_base = PreferenceDataset(Path(data["train_path"]), max_samples=args.max_train_samples)
    val_base = PreferenceDataset(Path(data["val_path"]), max_samples=args.max_val_samples)
    train_sampling = str(data.get("train_sampling", "image_balanced"))
    val_sampling = str(data.get("validation_sampling", "image_balanced"))
    train_dataset = preference_sampling_dataset(
        train_base,
        train_sampling,
        seed=int(training.get("seed", 42)),
        randomize=True,
    )
    val_dataset = preference_sampling_dataset(
        val_base,
        val_sampling,
        seed=int(training.get("seed", 42)),
        randomize=False,
    )
    batch_size = int(training.get("batch_size", 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    print(
        f"[data] sampling: train={train_sampling} {len(train_dataset)}/{len(train_base)} pairs; "
        f"validation={val_sampling} {len(val_dataset)}/{len(val_base)} pairs"
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    epochs = int(training["num_epochs"])
    update_steps = math.ceil(len(train_loader) / accumulation) * epochs
    scheduler = get_scheduler(
        name=str(training.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(update_steps * float(training.get("warmup_ratio", 0.0))),
        num_training_steps=update_steps,
    )
    objective_cfg = dict(config.get("objective") or {})
    objective_cfg.setdefault("name", "dpo")
    objective_cfg.setdefault("beta", float(training.get("beta", 0.1)))
    metrics_path = output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    eval_steps = int(training.get("eval_steps", 50))
    eval_milestones = configured_eval_milestones(training, update_steps)
    validation_dir = output_dir / "validation"
    checkpoint_root = output_dir / "checkpoints"
    bootstrap_samples = int(training.get("cluster_bootstrap_samples", 5000))
    early_cfg = dict(training.get("early_stopping") or {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_metric = str(early_cfg.get("metric", "eval_image_mean_loss"))
    early_mode = str(early_cfg.get("mode", "min"))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_patience = int(early_cfg.get("patience", 2))
    if early_patience < 1:
        raise ValueError("early_stopping.patience must be at least 1")
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    global_step = 0
    best_value = float("inf") if early_mode == "min" else -float("inf")
    best_step = 0
    stale_evaluations = 0
    stopped_early = False
    last_values: dict[str, float] | None = None
    optimizer.zero_grad(set_to_none=True)
    atomic_write_json(
        output_dir / "run_status.json",
        {
            "state": "initializing",
            "step": 0,
            "total_steps": update_steps,
            "eval_milestones": sorted(eval_milestones),
            "early_stopping": early_cfg,
        },
    )
    if not args.debug_one_step:
        baseline = evaluate(
            model,
            val_loader,
            device,
            objective_cfg,
            artifacts_dir=validation_dir,
            step=0,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=int(training.get("seed", 42)),
        )
        if early_metric not in baseline:
            raise KeyError(f"Configured early-stopping metric is unavailable: {early_metric}")
        best_value = baseline[early_metric]
        print("[eval] baseline " + " ".join(f"{key}={value:.5f}" for key, value in baseline.items()))
        append_metric(
            metrics_path,
            {"split": "validation_baseline", "step": 0, "objective": objective_cfg["name"], **baseline},
        )
        last_values = baseline
        atomic_write_json(
            output_dir / "run_status.json",
            {
                "state": "running",
                "step": 0,
                "total_steps": update_steps,
                "best_step": best_step,
                "best_metric": early_metric,
                "best_value": best_value,
                "eval_milestones": sorted(eval_milestones),
                "early_stopping": early_cfg,
            },
        )
    for epoch in range(epochs):
        model.train()
        for micro_step, batch in enumerate(train_loader, start=1):
            loss, metrics, _ = preference_metrics(model, batch, device, objective_cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite DPO loss at epoch={epoch + 1}, micro_step={micro_step}: {loss}")
            (loss / accumulation).backward()
            should_step = micro_step % accumulation == 0 or micro_step == len(train_loader)
            if not should_step:
                continue
            clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % int(training.get("logging_steps", 10)) == 0 or args.debug_one_step:
                print(
                    f"[train] objective={objective_cfg['name']} step={global_step}/{update_steps} "
                    f"loss={float(loss.detach().cpu()):.5f} "
                    f"policy_acc={metrics['policy_accuracy']:.3f} reward_acc={metrics['reward_accuracy']:.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
                append_metric(
                    metrics_path,
                    {"split": "train", "step": global_step, "objective": objective_cfg["name"], "loss": float(loss.detach().cpu()), **metrics},
                )
                atomic_write_json(
                    output_dir / "run_status.json",
                    {
                        "state": "running",
                        "step": global_step,
                        "total_steps": update_steps,
                        "best_step": best_step,
                        "best_metric": early_metric,
                        "best_value": best_value,
                        "stale_evaluations": stale_evaluations,
                        "eval_milestones": sorted(eval_milestones),
                        "early_stopping": early_cfg,
                    },
                )
            if args.debug_one_step:
                print("[debug-one-step] completed one optimizer step; no adapter checkpoint saved.")
                return
            should_evaluate = global_step in eval_milestones or (
                eval_steps > 0 and global_step % eval_steps == 0
            )
            if should_evaluate:
                values = evaluate(
                    model,
                    val_loader,
                    device,
                    objective_cfg,
                    artifacts_dir=validation_dir,
                    step=global_step,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=int(training.get("seed", 42)),
                )
                print("[eval] " + " ".join(f"{key}={value:.5f}" for key, value in values.items()))
                append_metric(metrics_path, {"split": "validation", "step": global_step, "objective": objective_cfg["name"], **values})
                save_adapter(model, processor, checkpoint_root / f"step_{global_step:06d}")
                save_adapter(model, processor, Path(output["latest_adapter_dir"]))
                if is_metric_improvement(
                    values[early_metric], best_value, early_mode, early_min_delta
                ):
                    best_value = values[early_metric]
                    best_step = global_step
                    stale_evaluations = 0
                    save_adapter(model, processor, Path(output["best_adapter_dir"]))
                else:
                    stale_evaluations += 1
                last_values = values
                atomic_write_json(
                    output_dir / "run_status.json",
                    {
                        "state": "running",
                        "step": global_step,
                        "total_steps": update_steps,
                        "best_step": best_step,
                        "best_metric": early_metric,
                        "best_value": best_value,
                        "stale_evaluations": stale_evaluations,
                        "last_validation": values,
                        "eval_milestones": sorted(eval_milestones),
                        "early_stopping": early_cfg,
                    },
                )
                if early_enabled and stale_evaluations >= early_patience:
                    stopped_early = True
                    print(
                        f"[early-stop] metric={early_metric} best={best_value:.6f} "
                        f"best_step={best_step} patience={early_patience}"
                    )
                    break
        if stopped_early:
            break
        print(f"[epoch] completed {epoch + 1}/{epochs}")
    if last_values is None or global_step not in eval_milestones:
        last_values = evaluate(
            model,
            val_loader,
            device,
            objective_cfg,
            artifacts_dir=validation_dir,
            step=global_step,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=int(training.get("seed", 42)),
        )
    print("[eval] final " + " ".join(f"{key}={value:.5f}" for key, value in last_values.items()))
    append_metric(metrics_path, {"split": "validation_final", "step": global_step, "objective": objective_cfg["name"], **last_values})
    save_adapter(model, processor, Path(output["final_adapter_dir"]))
    atomic_write_json(
        output_dir / "run_status.json",
        {
            "state": "early_stopped" if stopped_early else "complete",
            "step": global_step,
            "total_steps": update_steps,
            "best_step": best_step,
            "best_metric": early_metric,
            "best_value": best_value,
            "stale_evaluations": stale_evaluations,
            "last_validation": last_values,
            "eval_milestones": sorted(eval_milestones),
            "early_stopping": early_cfg,
        },
    )


if __name__ == "__main__":
    main()
