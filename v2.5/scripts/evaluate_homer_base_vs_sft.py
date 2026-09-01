#!/usr/bin/env python3
"""Generate 10-candidate Base-vs-SFT pools under the same online HOMER latent trace."""
from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.latent_communication.bridge import TypedHomerLatentBridge
from src.latent_communication.qwen_pipeline import generate_homer_candidates, generate_homer_plan_trace, model_device
from src.models.qwen_vl_dual_adapter import load_shared_qwen_vl_adapters
from src.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-below-requested-minimum", action="store_true", help="Development only; manifest remains marked non-formal")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rows = read_jsonl(Path(cfg["data"]["input_jsonl"]))
    unique = {}
    for row in rows:
        unique.setdefault(str(row["cluster_id"]), row)
    requested = int(cfg["data"]["requested_independent_images"])
    if len(unique) < requested and not args.allow_below_requested_minimum:
        raise RuntimeError(f"formal evaluation requires {requested} independent images; found {len(unique)}")
    rows = list(unique.values())
    if args.limit is not None:
        rows = rows[:args.limit]
    model, processor, process_vision_info = load_shared_qwen_vl_adapters(
        cfg["model"]["model_name"], cfg["planner"]["adapter_dir"],
        next(item["adapter_dir"] for item in cfg["generator_systems"] if item["adapter_dir"]),
        device_map="auto", torch_dtype="bfloat16", load_in_4bit=True,
    )
    payload = torch.load(cfg["bridge"]["checkpoint"], map_location="cpu", weights_only=False)
    bcfg = payload["config"]; embedding = model.get_input_embeddings().weight
    hidden = int(embedding.shape[1]); target_norm = float(embedding[:4096].float().norm(dim=-1).mean())
    bridge = TypedHomerLatentBridge(
        hidden, hidden, bottleneck_dim=int(bcfg["bottleneck_dim"]),
        num_slots=int(bcfg["num_slots_per_channel"]), num_heads=int(bcfg["num_heads"]),
        dropout=float(bcfg["dropout"]), target_norm=target_norm,
    )
    bridge.load_state_dict(payload["state_dict"]); bridge.to(model_device(model), dtype=torch.bfloat16).eval()
    out = Path(cfg["output"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    results, traces = [], []
    for image_index, row in enumerate(rows):
        trace = generate_homer_plan_trace(
            model, processor, process_vision_info, image=row["image"],
            max_new_tokens=384, max_state_tokens=256, image_max_pixels=100352,
        )
        traces.append({"cluster_id": row["cluster_id"], "image": row["image"], "plan": trace.plan.to_dict()})
        for seed in cfg["generation"]["seeds"]:
            resolved = int(seed) + image_index
            for system in cfg["generator_systems"]:
                torch.manual_seed(resolved)
                candidates = generate_homer_candidates(
                    model, processor, process_vision_info, bridge, trace, image=row["image"], mode="latent",
                    num_candidates=int(cfg["generation"]["candidates_per_system"]),
                    max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
                    temperature=float(cfg["generation"]["temperature"]), top_p=float(cfg["generation"]["top_p"]),
                    top_k=cfg["generation"].get("top_k"),
                    generator_adapter=None if system["adapter_dir"] is None else "generator",
                )
                results.append({"cluster_id": row["cluster_id"], "image": row["image"], "seed": int(seed), "system": system["name"], "candidates": candidates})
        print(f"[homer-eval] {image_index+1}/{len(rows)} {row['cluster_id']}")
    write_jsonl(out / "candidate_pools.jsonl", results); write_jsonl(out / "planner_traces.jsonl", traces)
    manifest = {
        "independent_images": len(rows), "requested_independent_images": requested,
        "formal_minimum_met": len(rows) >= requested, "candidates_per_system": cfg["generation"]["candidates_per_system"],
        "seeds": cfg["generation"]["seeds"], "same_online_trace_across_systems": True,
        "statistical_unit": "cluster_id", "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
