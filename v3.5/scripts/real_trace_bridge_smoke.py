#!/usr/bin/env python3
"""One-step bridge smoke using two real, validated frozen-Planner traces."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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
    length_matched_channel_donors,
    load_trace_index,
    prepare_example,
    _prompt_and_full,
)
from humor_generator_v35.training.cross_attention_bridge import ReceiverCrossAttentionTask
from humor_generator_v35.training.losses import symmetric_info_nce, variance_floor_loss
from humor_generator_v35.training.memory_safe import configure_frozen_receiver
from humor_generator_v35.training.public_bridge import (
    PUBLIC_BRIDGE_PROMPT_TRACK,
    load_context_index,
    validate_row_context,
    public_text_caption_messages,
    target_caption_text,
)
from humor_generator_v35.homer.prompts import text_part
from humor_generator_v35.training.selected_plan import (
    SELECTED_PLAN_POLICY, LEGACY_MEMORY_POLICY, PARTIAL_CAPTION_POLICY, selected_plan_channels,
)
from scripts.train_bridge import fixed_hash_sample_clusters


def memory_token_count(row, *, memory_source, backend, context_index, traces):
    """Measure the actual configured memory, with the production chat template."""
    if memory_source == SELECTED_PLAN_POLICY:
        context = context_index[str(row["cluster_id"])]
        return sum(
            int(backend.encode(
                [{"role": "user", "content": [text_part(text)]}],
                add_generation_prompt=False,
            )["input_ids"].shape[1])
            for text in selected_plan_channels(context).values()
        )
    if memory_source != LEGACY_MEMORY_POLICY:
        raise ValueError(f"unsupported smoke memory source: {memory_source}")
    record = traces[row["cluster_id"]]
    loaded = load_trace(ROOT / record["trace_path"], expected_sha256=record["trace_sha256"])
    return sum(int(item.states.shape[1]) for item in loaded.values())


def select_stress_rows(rows, raw_pixels, latent_tokens):
    if len({row["cluster_id"] for row in rows}) < 2:
        raise ValueError("smoke requires two independent clusters")
    image_stress = max(rows, key=lambda row: (raw_pixels(row), row["cluster_id"]))
    memory_stress = max(rows, key=lambda row: (latent_tokens(row), row["cluster_id"]))
    result = [image_stress]
    if memory_stress["cluster_id"] != image_stress["cluster_id"]:
        result.append(memory_stress)
    else:
        result.append(next(row for row in rows if row["cluster_id"] != image_stress["cluster_id"]))
    return result


def require_finite_metrics(metrics, label):
    for name, value in metrics.items():
        if not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite {label}: {name}={value}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    # Resolve the protocol before selecting donor policy.  The semantic smoke
    # branch needs to know whether it is exercising the receiver cross-
    # attention path; keeping this assignment next to config parsing prevents
    # an uninitialised-baseline failure before any model forward.
    baseline = config["experiment"]["baseline"]
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
    # image size and full memory from the configured source.
    selected = cluster_balanced_rows(rows, epoch=0, seed=seed)
    def raw_pixels(row: dict) -> int:
        with Image.open(row["image"]) as image:
            return image.width * image.height
    semantic_channel_training = (
        baseline == "receiver_cross_attention"
        and str(config["training"].get("stage")) == "semantic_reconstruction"
        and str(config["loss"].get("semantic_objective"))
        in {"channel_balanced_v3", "channel_isolated_v4", "channel_isolated_v5"}
    )
    if semantic_channel_training:
        negative_map, negative_diagnostics = length_matched_channel_donors(
            rows, rows, root=ROOT, trace_index=traces,
            allow_target_donor_overlap=True,
        )
    else:
        negative_map, negative_diagnostics = hard_negative_cluster_map(rows, traces)

    # The current public-code route must exercise the same sealed HOMER
    # summary/retrieval/selection context as the formal trainer.  Falling back
    # to the legacy zero-prefix/text-plan prompt here would make a seemingly
    # successful smoke a protocol mismatch, so load and validate the context
    # before touching the receiver weights.
    prompt_mode = str(config.get("protocol", {}).get("bridge_prompt_track", "legacy"))
    context_index = None
    context_index_path = None
    if prompt_mode == PUBLIC_BRIDGE_PROMPT_TRACK:
        context_value = config.get("data", {}).get("context_index")
        if not isinstance(context_value, str) or not context_value:
            raise RuntimeError("current public bridge smoke requires data.context_index")
        context_index_path = ROOT / context_value
        context_index = load_context_index(context_index_path)
        for row in selected:
            context = context_index.get(str(row["cluster_id"]))
            if context is None:
                raise RuntimeError(f"missing current HOMER context: {row['cluster_id']}")
            validate_row_context(row, context)
    elif prompt_mode != "legacy":
        raise RuntimeError(f"unsupported bridge prompt track: {prompt_mode}")

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
    memory_source = config.get("protocol", {}).get("memory_source", LEGACY_MEMORY_POLICY)
    if memory_source == SELECTED_PLAN_POLICY and context_index is None:
        raise RuntimeError("selected-plan smoke requires validated contexts")
    memory_lengths = {
        row["cluster_id"]: memory_token_count(
            row, memory_source=memory_source, backend=backend,
            context_index=context_index, traces=traces,
        ) for row in selected
    }
    def latent_tokens(row):
        return memory_lengths[row["cluster_id"]]
    prompt_lengths = {}
    if prompt_mode == PUBLIC_BRIDGE_PROMPT_TRACK:
        # The public caption route sends a text description, not image pixels.
        # Measure the teacher-forced text prompt including the actual target.
        for row in selected:
            target = target_caption_text(row)
            partial = config.get("protocol", {}).get("training_target_policy") == PARTIAL_CAPTION_POLICY
            if partial:
                target = "##Caption: " + target
            _, full, _ = _prompt_and_full(
                backend, public_text_caption_messages(context_index[str(row["cluster_id"])]),
                target, partial_answer=partial,
            )
            prompt_lengths[row["cluster_id"]] = int(full["input_ids"].shape[1])
        smoke_rows = select_stress_rows(selected, lambda row: prompt_lengths[row["cluster_id"]], latent_tokens)
    else:
        smoke_rows = select_stress_rows(selected, raw_pixels, latent_tokens)
    width = int(backend.model.get_input_embeddings().weight.shape[1])
    if baseline == "receiver_cross_attention":
        bridge = ReceiverDrivenCrossAttentionBridge(
            width,
            width,
            layer_indices=[int(item) for item in config["bridge"]["layer_indices"]],
            bottleneck_dim=int(config["bridge"]["bottleneck_dim"]),
            heads=int(config["bridge"]["heads"]),
            gate_init=float(config["bridge"]["gate_init"]),
            channel_fusion=str(config["bridge"].get("channel_fusion", "learned")),
            projection_mode=str(config["bridge"].get("projection_mode", "shared")),
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
            semantic_prompt_include_image=bool(
                config["training"].get("semantic_prompt_include_image", True)
            ),
            prompt_mode=prompt_mode,
            context_index=context_index,
            memory_source=config.get("protocol", {}).get("memory_source", "early_planner_trace_v1"),
            target_policy=config.get("protocol", {}).get("training_target_policy", "source_caption_only_no_fabricated_explanation"),
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
    alignment_students: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("conflict", "local", "global")
    }
    alignment_teachers: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("conflict", "local", "global")
    }
    metrics = None
    for row in smoke_rows:
        torch.cuda.reset_peak_memory_stats(device)
        example = prepare_example(row, traces[row["cluster_id"]], seed=seed)
        negative_cluster = negative_map[row["cluster_id"]]
        if semantic_channel_training:
            sample_metrics = task.backward_example(
                example, shuffled_clusters=negative_cluster,
                loss_scale=1.0 / len(smoke_rows)
            )
        else:
            sample_metrics = task.backward_example(
                example, negative_cluster, loss_scale=1.0 / len(smoke_rows)
            )
        if (
            isinstance(task, ReceiverCrossAttentionTask)
            and float(config["loss"].get("info_nce", 0.0)) > 0
        ):
            pairs = task.semantic_alignment_pairs(example)
            for channel, (student, teacher) in pairs.items():
                alignment_students[channel].append(student)
                alignment_teachers[channel].append(teacher)
        require_finite_metrics(sample_metrics, f"sample {row['cluster_id']}")
        validation_metrics = None
        if memory_source == SELECTED_PLAN_POLICY:
            validation_metrics = task.evaluate_example(example, negative_cluster)
            require_finite_metrics(validation_metrics, f"validation {row['cluster_id']}")
            if "nll_improvement_over_no_memory" not in validation_metrics:
                raise RuntimeError("selected-plan smoke did not exercise no-memory validation")
        with Image.open(row["image"]) as image:
            size = [image.width, image.height]
        sample_reports.append({
            "cluster": row["cluster_id"],
            "row_id": row["row_id"],
            "image_size": size,
            "raw_image_pixels": size[0] * size[1],
            "full_latent_tokens": latent_tokens(row),
            "text_teacher_full_tokens": prompt_lengths.get(row["cluster_id"]),
            "hard_negative_cluster": negative_cluster,
            "encode_diagnostics": backend.last_encode_diagnostics,
            "loss": sample_metrics,
            "validation_at_initial_weights": validation_metrics,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        })
        metrics = sample_metrics
    contrastive_metrics: dict[str, float] = {}
    if any(alignment_students.values()):
        nce_by_channel = {}
        retrieval_by_channel = {}
        variance_by_channel = {}
        for channel in alignment_students:
            students = torch.cat(alignment_students[channel], dim=0)
            teachers = torch.cat(alignment_teachers[channel], dim=0)
            nce_by_channel[channel], retrieval_by_channel[channel] = symmetric_info_nce(
                students, teachers,
                temperature=float(config["loss"].get("info_nce_temperature", 0.07)),
            )
            variance_by_channel[channel] = variance_floor_loss(students)
        info_nce = torch.stack(list(nce_by_channel.values())).mean()
        retrieval_at_1 = torch.stack(list(retrieval_by_channel.values())).mean()
        anti_collapse = torch.stack(list(variance_by_channel.values())).mean()
        (
            float(config["loss"]["info_nce"]) * info_nce
            + float(config["loss"].get("variance_floor", 0.0)) * anti_collapse
        ).backward()
        contrastive_metrics = {
            "info_nce": float(info_nce.detach().cpu()),
            "info_nce_retrieval_at_1": float(retrieval_at_1.detach().cpu()),
            "variance_floor": float(anti_collapse.detach().cpu()),
        }
        for channel in nce_by_channel:
            contrastive_metrics[f"info_nce_{channel}"] = float(
                nce_by_channel[channel].detach().cpu()
            )
            contrastive_metrics[f"retrieval_at_1_{channel}"] = float(
                retrieval_by_channel[channel].detach().cpu()
            )
    require_finite_metrics(contrastive_metrics, "contrastive loss")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        bridge.parameters(), float(config["training"].get("max_grad_norm", 1.0)),
        error_if_nonfinite=True,
    ))
    if gradient_norm <= 0:
        raise RuntimeError("bridge received no gradient")
    if any(parameter.grad is not None for parameter in backend.model.parameters()):
        raise RuntimeError("frozen receiver accumulated gradients")
    optimizer.step()
    update_norm = float(sum(
        (parameter.detach().float() - before[name]).square().sum()
        for name, parameter in bridge.named_parameters() if parameter.requires_grad
    ).sqrt())
    require_finite_metrics({"gradient_norm": gradient_norm, "update_norm": update_norm}, "optimizer")
    if update_norm <= 0:
        raise RuntimeError("bridge did not update")
    report = {
        "status": "pass",
        "scientific_training": False,
        "real_planner_traces": memory_source == LEGACY_MEMORY_POLICY,
        "memory_source": memory_source,
        "training_target_policy": task.target_policy if isinstance(task, ReceiverCrossAttentionTask) else None,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "clusters": [row["cluster_id"] for row in smoke_rows],
        "receiver_input_modality": "text_description" if prompt_mode == PUBLIC_BRIDGE_PROMPT_TRACK else "image_and_text",
        "selection": (
            "max_text_teacher_full_tokens_plus_max_full_latent_tokens_at_most_two_examples"
            if prompt_mode == PUBLIC_BRIDGE_PROMPT_TRACK
            else "max_raw_pixels_plus_max_full_latent_tokens_at_most_two_examples"
        ),
        "communication_interface": (
            "receiver_driven_full_state_cross_attention_no_soft_prefix"
            if baseline == "receiver_cross_attention" else "input_soft_prefix"
        ),
        "bridge_prompt_track": prompt_mode,
        "context_index": (
            None if context_index_path is None else str(context_index_path.relative_to(ROOT))
        ),
        "context_index_records": None if context_index is None else len(context_index),
        "context_index_sha256": (
            None if context_index_path is None
            else hashlib.sha256(context_index_path.read_bytes()).hexdigest()
        ),
        "channel_visibility": config["loss"].get("channel_visibility", "all"),
        "semantic_prompt_include_image": config["training"].get(
            "semantic_prompt_include_image", True
        ),
        "counterfactual_reconstruction_weight": config["loss"].get(
            "counterfactual_reconstruction", 0.0
        ),
        "samples": sample_reports,
        "hard_negative_diagnostics": negative_diagnostics,
        "channel_length_matching": semantic_channel_training,
        "policy_trainable_parameters": freeze_report.policy_trainable,
        "bridge_trainable_parameters": freeze_report.bridge_trainable,
        "gradient_checkpointing": freeze_report.gradient_checkpointing,
        "use_cache": freeze_report.use_cache,
        "min_visual_tokens": int(config["model"]["min_visual_tokens"]),
        "max_visual_tokens": int(config["model"]["max_visual_tokens"]),
        "gradient_norm": gradient_norm,
        "update_norm": update_norm,
        "last_loss": metrics,
        "contrastive_loss": contrastive_metrics,
        "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in sample_reports),
        "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in sample_reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
