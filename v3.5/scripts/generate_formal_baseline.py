#!/usr/bin/env python3
"""Generate one resumable v3 baseline condition on image-clustered held-out data."""
from __future__ import annotations

from argparse import ArgumentParser
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.data.traces import load_trace, plan_from_record, read_jsonl
from humor_generator_v35.homer.pipeline import HomerTextPipeline
from humor_generator_v35.homer.retrieval import (
    HomerRetrievalAugmenter,
    HomerRetrievalConfig,
    NltkWordNetGraph,
    OfficialQueryFittedTfidfIndex,
)
from humor_generator_v35.latent.bridges import (
    LearnedLatentBridge, TypedLatentBridge, mean_embedding_norm,
    nearest_vocabulary_embeddings,
)
from humor_generator_v35.latent.budget import channel_causal_tail, concatenate_budgeted
from humor_generator_v35.latent.statebridge import StateBridgeAlignment
from humor_generator_v35.qwen_backend import QwenBackend, model_device
from humor_generator_v35.training.formal_bridge import (
    full_plan_text_messages,
    latent_messages,
    load_trace_index,
)


CONDITIONS = {
    "text_homer",
    "full_plan_text",
    "budget_text",
    "token_embedding",
    "statebridge",
    "learned_latent",
    "typed_learned_latent",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_rows_by_cluster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["row_id"]):
        result.setdefault(row["cluster_id"], row)
    return list(result.values())


def load_bridge(config: dict[str, Any], checkpoint: Path, backend: QwenBackend) -> torch.nn.Module:
    width = int(backend.model.get_input_embeddings().weight.shape[1])
    kwargs = {
        "bottleneck_dim": int(config["bridge"]["bottleneck_dim"]),
        "heads": int(config["bridge"]["heads"]),
        "target_norm": mean_embedding_norm(backend.model.get_input_embeddings().weight),
    }
    baseline = config["experiment"]["baseline"]
    bridge = (
        TypedLatentBridge(
            width, width, slots=int(config["bridge"]["slots_per_channel"]), **kwargs
        )
        if baseline == "typed_learned_latent"
        else LearnedLatentBridge(
            width, width, slots=int(config["bridge"]["total_slots"]), **kwargs
        )
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    bridge.load_state_dict(payload["bridge"])
    bridge.to(model_device(backend.model)).eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    return bridge


def latent_slots(
    condition: str,
    trace_record: dict[str, Any],
    backend: QwenBackend,
    *,
    bridge: torch.nn.Module | None,
    slots_per_channel: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    trace_path = ROOT / trace_record["trace_path"]
    states = load_trace(trace_path, expected_sha256=trace_record["trace_sha256"])
    device = model_device(backend.model)
    budgeted = channel_causal_tail(states, slots_per_channel=slots_per_channel)
    if condition == "token_embedding":
        token_ids = concatenate_budgeted(budgeted).token_ids.to(device)
        slots = backend.model.get_input_embeddings()(token_ids)
        return slots, {
            "slots_per_channel": slots_per_channel,
            "latent_slots": int(slots.shape[1]),
            "communication_budget": "causal_tail_per_channel",
        }
    if condition == "statebridge":
        aligner = StateBridgeAlignment(backend.model.get_input_embeddings().weight)
        aligned = {}
        diagnostics = {}
        for name in TypedLatentBridge.channel_order:
            item = budgeted.channels[name]
            aligned[name] = aligner(item.states.to(device), item.token_ids.to(device))
            diagnostics[name] = dict(aligner.last_diagnostics)
        slots = torch.cat([aligned[name] for name in TypedLatentBridge.channel_order], dim=1)
        return slots, {
            "statebridge_original_tokens": budgeted.original_lengths,
            "statebridge_transmitted_tokens": budgeted.transmitted_lengths,
            "slots_per_channel": slots_per_channel,
            "latent_slots": int(slots.shape[1]),
            "statebridge_truncation": "causal_tail_per_channel",
            "statebridge_solver": diagnostics,
        }
    if bridge is None:
        raise RuntimeError(f"{condition} requires a trained bridge checkpoint")
    dtype = next(bridge.parameters()).dtype
    values = {
        name: states[name].states.to(device=device, dtype=dtype)
        for name in ("conflict", "local", "global")
    }
    with torch.inference_mode():
        if isinstance(bridge, TypedLatentBridge):
            slots = bridge(values)["all"]
        else:
            slots = bridge(torch.cat(list(values.values()), dim=1))
    return slots, {"latent_slots": int(slots.shape[1])}


def exact_semantics(trace_record: dict[str, Any]) -> dict[str, str]:
    states = load_trace(
        ROOT / trace_record["trace_path"], expected_sha256=trace_record["trace_sha256"]
    )
    return {name: states[name].semantics for name in TypedLatentBridge.channel_order}


def budget_text_semantics(
    trace_record: dict[str, Any], backend: QwenBackend, *, slots_per_channel: int
) -> tuple[dict[str, str], dict[str, int]]:
    states = load_trace(
        ROOT / trace_record["trace_path"], expected_sha256=trace_record["trace_sha256"]
    )
    budgeted = channel_causal_tail(states, slots_per_channel=slots_per_channel)
    tokenizer = getattr(backend.processor, "tokenizer", backend.processor)
    semantics = {
        name: tokenizer.decode(
            budgeted.channels[name].token_ids.squeeze(0).tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        for name in TypedLatentBridge.channel_order
    }
    if any(not value for value in semantics.values()):
        raise RuntimeError("budget-text decoding produced an empty semantic channel")
    return semantics, budgeted.original_lengths


def build_retriever() -> HomerRetrievalAugmenter:
    manifest = json.loads((ROOT / "manifests/homer_official_assets.json").read_text())
    corpus_path = ROOT / manifest["joke_corpus"]["local_path"]
    if file_sha256(corpus_path) != manifest["joke_corpus"]["sha256"]:
        raise RuntimeError("HOMER joke corpus hash mismatch")
    with corpus_path.open(encoding="utf-8", newline="") as handle:
        jokes = [row["Joke"] for row in csv.DictReader(handle)]
    import nltk
    nltk.data.path.insert(0, str(ROOT / "artifacts/nltk_data"))
    return HomerRetrievalAugmenter(
        OfficialQueryFittedTfidfIndex(jokes),
        NltkWordNetGraph(),
        config=HomerRetrievalConfig(top_k=5, delta=5),
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument(
        "--label",
        help="Output label for checkpoint variants; implementation condition is preserved separately.",
    )
    parser.add_argument("--receiver", choices=["base", "sft"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--split", choices=[
            "validation", "internal_test", "official_hia_unseen_test",
            "official_hia_seen_diagnostic",
        ],
        default="internal_test",
    )
    parser.add_argument(
        "--trace-index", type=Path,
        default=ROOT / "data/cache/planner_traces_homer_strict_v35/index.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260830, 20260831, 20260832])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--slots-per-channel", type=int, default=8)
    parser.add_argument(
        "--exclude-clusters-file", type=Path,
        help="JSON file containing a validation_cluster_ids list to exclude (outer-pilot eval).",
    )
    parser.add_argument(
        "--quantize-bridge-output", action="store_true",
        help="Replace learned continuous slots with nearest receiver token embeddings.",
    )
    args = parser.parse_args()
    if args.condition in {"learned_latent", "typed_learned_latent"} and args.checkpoint is None:
        parser.error("learned conditions require --checkpoint")
    if args.slots_per_channel < 2:
        parser.error("communication baselines need at least two states per channel")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("generation seeds must be unique")
    condition_label = args.label or args.condition
    if not condition_label.replace("_", "").replace("-", "").isalnum():
        parser.error("--label must contain only letters, digits, '_' or '-'")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    expected_receiver = "sft" if config["model"].get("adapter") else "base"
    if expected_receiver != args.receiver:
        raise RuntimeError(f"config receiver is {expected_receiver}, CLI requested {args.receiver}")
    if args.condition in {"learned_latent", "typed_learned_latent"}:
        if config["experiment"]["baseline"] != args.condition:
            raise RuntimeError("bridge config baseline does not match requested condition")

    adapter = config["model"].get("adapter")
    backend = QwenBackend.load(
        config["model"]["name"],
        revision=config["model"]["revision"],
        adapter=None if adapter is None else ROOT / adapter,
        load_in_4bit=True,
    )
    bridge = None if args.checkpoint is None else load_bridge(config, args.checkpoint, backend)
    traces = load_trace_index(args.trace_index)
    rows = first_rows_by_cluster(read_jsonl(ROOT / config["data"]["dataset"] / f"{args.split}.jsonl"))
    excluded_clusters: set[str] = set()
    if args.exclude_clusters_file is not None:
        exclusion = json.loads(args.exclude_clusters_file.read_text())
        values = exclusion.get("validation_cluster_ids")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise RuntimeError("exclusion file must contain validation_cluster_ids: list[str]")
        excluded_clusters = set(values)
        rows = [row for row in rows if row["cluster_id"] not in excluded_clusters]
        if not rows:
            raise RuntimeError("cluster exclusion removed every evaluation row")
    missing = sorted({row["cluster_id"] for row in rows} - set(traces))
    if missing:
        raise RuntimeError(f"missing {len(missing)} sealed-{args.split} traces; first={missing[:5]}")

    retriever = build_retriever() if args.condition == "text_homer" else None
    homer = HomerTextPipeline(backend, retriever=retriever, strict_reproduction=True) if retriever else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, int]] = set()
    if args.output.is_file():
        completed = {
            (row["cluster_id"], int(row["generation_seed"]))
            for row in read_jsonl(args.output)
        }
    started = time.time()
    with args.output.open("a") as handle:
        for row in rows:
            if all((row["cluster_id"], value) in completed for value in args.seeds):
                continue
            trace_record = traces[row["cluster_id"]]
            plan = plan_from_record(trace_record["plan"])
            augmented_plan = retriever.augment(plan) if retriever is not None else None
            for generation_seed in args.seeds:
                if (row["cluster_id"], generation_seed) in completed:
                    continue
                metadata: dict[str, Any] = {}
                if args.condition == "text_homer":
                    result = homer.generate_caption(augmented_plan, seed=generation_seed)  # type: ignore[arg-type,union-attr]
                    caption = result.caption
                    metadata.update({
                        "selected_conflict": result.selected_conflict,
                        "selected_path": list(result.selected_path),
                        "retrieval": "official_query_fitted_tfidf",
                    })
                elif args.condition in {"full_plan_text", "budget_text"}:
                    semantics = exact_semantics(trace_record)
                    if args.condition == "budget_text":
                        semantics, original_lengths = budget_text_semantics(
                            trace_record, backend, slots_per_channel=args.slots_per_channel
                        )
                        metadata.update({
                            "original_tokens": original_lengths,
                            "slots_per_channel": args.slots_per_channel,
                            "communication_budget": "decoded_causal_tail_per_channel",
                        })
                    caption = backend.generate(
                        full_plan_text_messages(row["image"], semantics),
                        temperature=1.0,
                        max_new_tokens=args.max_new_tokens,
                        seed=generation_seed,
                    )
                else:
                    slots, latent_meta = latent_slots(
                        args.condition,
                        trace_record,
                        backend,
                        bridge=bridge,
                        slots_per_channel=args.slots_per_channel,
                    )
                    if args.quantize_bridge_output:
                        if args.condition not in {"learned_latent", "typed_learned_latent"}:
                            raise RuntimeError("bridge-output quantization requires a learned bridge")
                        slots, token_ids = nearest_vocabulary_embeddings(
                            slots, backend.model.get_input_embeddings().weight
                        )
                        latent_meta.update({
                            "bridge_output_quantized": True,
                            "quantized_token_ids": token_ids.squeeze(0).tolist(),
                        })
                    caption = backend.generate_with_latent_prefix(
                        latent_messages(row["image"]),
                        slots,
                        temperature=1.0,
                        max_new_tokens=args.max_new_tokens,
                        seed=generation_seed,
                    )
                    metadata.update(latent_meta)
                caption = caption.strip()
                if caption.casefold().startswith("caption:"):
                    caption = caption.split(":", 1)[1].strip()
                empty_output = not caption
                if empty_output:
                    # Never resample an unfavorable seed: that would bias a
                    # fixed-seed comparison.  Preserve the failure for judges.
                    caption = "[EMPTY OUTPUT]"
                record = {
                    "schema_version": 1,
                    "receiver": args.receiver,
                    "condition": condition_label,
                    "implementation_condition": args.condition,
                    "split": args.split,
                    "cluster_id": row["cluster_id"],
                    "contest_number": row["contest_number"],
                    "image": row["image"],
                    "image_sha256": row["image_sha256"],
                    "standard_description": row["standard_description"],
                    "generation_seed": generation_seed,
                    "temperature": 1.0,
                    "max_new_tokens": args.max_new_tokens,
                    "caption": caption,
                    "empty_output": empty_output,
                    **metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed.add((row["cluster_id"], generation_seed))
                print(json.dumps({
                    "completed": len(completed),
                    "total": len(rows) * len(args.seeds),
                    "elapsed_seconds": round(time.time() - started, 1),
                }), flush=True)

    manifest = {
        "schema_version": 1,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "condition": condition_label,
        "implementation_condition": args.condition,
        "receiver": args.receiver,
        "split": args.split,
        "seeds": args.seeds,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0},
        "communication_budget": {
            "slots_per_channel": args.slots_per_channel,
            "channels": list(TypedLatentBridge.channel_order),
            "total_slots": 3 * args.slots_per_channel,
        },
        "config_sha256": file_sha256(config_path),
        "trace_index_sha256": file_sha256(args.trace_index),
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        "quantize_bridge_output": args.quantize_bridge_output,
        "output_sha256": file_sha256(args.output),
        "records": len(completed),
        "excluded_clusters": sorted(excluded_clusters),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
