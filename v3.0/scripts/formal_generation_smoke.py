#!/usr/bin/env python3
"""Real-trace inference smoke for text, StateBridge, learned, and typed paths."""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v3.data.traces import read_jsonl
from humor_generator_v3.homer.prompts import caption_messages
from humor_generator_v3.latent.bridges import LearnedLatentBridge, TypedLatentBridge, mean_embedding_norm
from humor_generator_v3.qwen_backend import QwenBackend, model_device
from humor_generator_v3.training.formal_bridge import latent_messages, load_trace_index, prepare_example
from generate_formal_baseline import latent_slots


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
    if not rows:
        raise RuntimeError("generation smoke needs one real Planner trace")
    row = rows[0]
    trace_record = traces[row["cluster_id"]]
    adapter = config["model"].get("adapter")
    backend = QwenBackend.load(
        config["model"]["name"], revision=config["model"]["revision"],
        adapter=None if adapter is None else ROOT / adapter, load_in_4bit=True,
    )
    device = model_device(backend.model)
    width = int(backend.model.get_input_embeddings().weight.shape[1])
    common = {
        "bottleneck_dim": int(config["bridge"]["bottleneck_dim"]),
        "slots": int(config["bridge"]["slots_per_channel"]),
        "heads": int(config["bridge"]["heads"]),
        "target_norm": mean_embedding_norm(backend.model.get_input_embeddings().weight),
    }
    bridges = {
        "learned_latent": LearnedLatentBridge(width, width, **common).to(device).eval(),
        "typed_learned_latent": TypedLatentBridge(width, width, **common).to(device).eval(),
    }
    example = prepare_example(row, trace_record, seed=123)
    outputs = {
        "matched_text": backend.generate(
            caption_messages(row["standard_description"], example.conflict, list(example.path)),
            temperature=1.0, max_new_tokens=24, seed=123,
        )
    }
    latent_meta = {}
    for condition in ("statebridge", "learned_latent", "typed_learned_latent"):
        bridge = bridges.get(condition)
        slots, metadata = latent_slots(
            condition, trace_record, backend, bridge=bridge, statebridge_max_tokens=64
        )
        outputs[condition] = backend.generate_with_latent_prefix(
            latent_messages(row["standard_description"]), slots,
            temperature=1.0, max_new_tokens=24, seed=123,
        )
        latent_meta[condition] = metadata
    empty_conditions = sorted(name for name, value in outputs.items() if not value.strip())
    outputs = {
        name: value.strip() or "[EMPTY OUTPUT]"
        for name, value in outputs.items()
    }
    report = {
        "status": "pass",
        "scientific_evaluation": False,
        "cluster_id": row["cluster_id"],
        "policy_trainable_parameters": sum(
            parameter.numel() for parameter in backend.model.parameters() if parameter.requires_grad
        ),
        "sampling": {"temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0},
        "empty_conditions": empty_conditions,
        "outputs": outputs,
        "latent_metadata": latent_meta,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    if report["policy_trainable_parameters"] != 0:
        raise RuntimeError("frozen receiver contract failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
