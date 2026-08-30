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

from humor_generator_v3.data.traces import load_trace, plan_from_record, read_jsonl
from humor_generator_v3.homer.pipeline import HomerTextPipeline
from humor_generator_v3.homer.prompts import caption_messages
from humor_generator_v3.homer.retrieval import (
    HomerRetrievalAugmenter,
    HomerRetrievalConfig,
    NltkWordNetGraph,
    OfficialQueryFittedTfidfIndex,
)
from humor_generator_v3.latent.bridges import LearnedLatentBridge, TypedLatentBridge, mean_embedding_norm
from humor_generator_v3.latent.statebridge import StateBridgeAlignment
from humor_generator_v3.qwen_backend import QwenBackend, model_device
from humor_generator_v3.training.formal_bridge import latent_messages, load_trace_index, prepare_example


CONDITIONS = {"text_homer", "matched_text", "statebridge", "learned_latent", "typed_learned_latent"}


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
        "slots": int(config["bridge"]["slots_per_channel"]),
        "heads": int(config["bridge"]["heads"]),
        "target_norm": mean_embedding_norm(backend.model.get_input_embeddings().weight),
    }
    baseline = config["experiment"]["baseline"]
    bridge = (
        TypedLatentBridge(width, width, **kwargs)
        if baseline == "typed_learned_latent"
        else LearnedLatentBridge(width, width, **kwargs)
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
    statebridge_max_tokens: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    trace_path = ROOT / trace_record["trace_path"]
    states = load_trace(trace_path, expected_sha256=trace_record["trace_sha256"])
    device = model_device(backend.model)
    if condition == "statebridge":
        hidden = torch.cat([states[name].states for name in ("conflict", "local", "global")], dim=1)
        token_ids = torch.cat([states[name].token_ids for name in ("conflict", "local", "global")], dim=1)
        original_tokens = int(token_ids.shape[1])
        # StateBridge uses a bounded communication prefix.  Tail truncation is
        # deterministic and preserves the most recent autoregressive states.
        hidden = hidden[:, -statebridge_max_tokens:]
        token_ids = token_ids[:, -statebridge_max_tokens:]
        aligner = StateBridgeAlignment(backend.model.get_input_embeddings().weight)
        slots = aligner(hidden.to(device), token_ids.to(device))
        return slots, {
            "statebridge_original_tokens": original_tokens,
            "statebridge_transmitted_tokens": int(slots.shape[1]),
            "statebridge_truncation": "causal_tail",
            "statebridge_solver": aligner.last_diagnostics,
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
    parser.add_argument("--receiver", choices=["base", "sft"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--trace-index", type=Path, default=ROOT / "data/cache/planner_traces/index.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, default=[20260830, 20260831, 20260832])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--statebridge-max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.condition in {"learned_latent", "typed_learned_latent"} and args.checkpoint is None:
        parser.error("learned conditions require --checkpoint")
    if args.statebridge_max_tokens < 2:
        parser.error("StateBridge needs at least two aligned tokens")

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
                elif args.condition == "matched_text":
                    example = prepare_example(row, trace_record, seed=generation_seed)
                    caption = backend.generate(
                        caption_messages(example.row["standard_description"], example.conflict, list(example.path)),
                        temperature=1.0,
                        max_new_tokens=args.max_new_tokens,
                        seed=generation_seed,
                    )
                    metadata.update({"selected_conflict": example.conflict, "selected_path": list(example.path)})
                else:
                    slots, latent_meta = latent_slots(
                        args.condition,
                        trace_record,
                        backend,
                        bridge=bridge,
                        statebridge_max_tokens=args.statebridge_max_tokens,
                    )
                    caption = backend.generate_with_latent_prefix(
                        latent_messages(plan.description),
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
                    "condition": args.condition,
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
                    "total": len(rows) * 3,
                    "elapsed_seconds": round(time.time() - started, 1),
                }), flush=True)

    manifest = {
        "schema_version": 1,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "condition": args.condition,
        "receiver": args.receiver,
        "split": args.split,
        "seeds": args.seeds,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0},
        "config_sha256": file_sha256(config_path),
        "trace_index_sha256": file_sha256(args.trace_index),
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        "output_sha256": file_sha256(args.output),
        "records": len(completed),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
