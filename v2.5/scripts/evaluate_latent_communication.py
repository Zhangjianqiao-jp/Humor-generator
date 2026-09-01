#!/usr/bin/env python3
"""Generate matched Text/Latent/Hybrid Group-of-3 candidates with an online 7B Planner."""
from __future__ import annotations

import hashlib
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.latent_communication.bridge import LearnedLatentBridge
from src.latent_communication.qwen_pipeline import generate_generator_candidates, generate_planner_trace, model_device
from src.models.qwen_vl_dual_adapter import assert_only_bridge_trainable, load_shared_qwen_vl_adapters
from src.training.sft_dataset import extract_image_path
from src.utils.io import read_jsonl, write_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_images(rows: list[dict]) -> list[dict]:
    result, seen = [], set()
    for row in rows:
        key = str(row.get("image_id") or extract_image_path(row))
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model, processor, process_vision_info = load_shared_qwen_vl_adapters(
        config["model"]["model_name"],
        config["planner"]["adapter_dir"],
        config["generator"]["adapter_dir"],
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
        load_in_4bit=bool(config["model"].get("load_in_4bit", True)),
    )
    bridge_file = Path(config["bridge"]["checkpoint"])
    payload = torch.load(bridge_file, map_location="cpu", weights_only=True)
    bridge_config = payload["config"]
    embedding = model.get_input_embeddings().weight
    hidden_size = int(embedding.shape[1])
    target_norm = float(embedding[: min(4096, embedding.shape[0])].float().norm(dim=-1).mean())
    bridge = LearnedLatentBridge(
        hidden_size,
        hidden_size,
        bottleneck_dim=int(bridge_config["bottleneck_dim"]),
        num_slots=int(bridge_config["num_slots"]),
        num_heads=int(bridge_config["num_heads"]),
        dropout=float(bridge_config["dropout"]),
        target_norm=target_norm,
    )
    bridge.load_state_dict(payload["state_dict"], strict=True)
    bridge.to(model_device(model), dtype=torch.bfloat16).eval()
    assert_only_bridge_trainable(model, bridge)

    rows = unique_images(read_jsonl(Path(config["data"]["input_jsonl"])))
    if args.limit is not None:
        rows = rows[: args.limit]
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    planner_prompt = Path(config["planner"]["prompt_file"]).read_text(encoding="utf-8").strip()
    modes = ["text", "latent", "hybrid"]
    outputs = {(mode, seed): [] for mode in modes for seed in config["generation"]["seeds"]}
    trace_rows = []
    for image_index, row in enumerate(rows):
        image = str(extract_image_path(row))
        image_id = str(row.get("image_id") or Path(image).stem)
        trace = generate_planner_trace(
            model,
            processor,
            process_vision_info,
            image=image,
            planner_prompt=planner_prompt,
            max_new_tokens=int(config["planner"]["max_new_tokens"]),
            max_state_tokens=int(config["planner"]["max_state_tokens"]),
        )
        trace_rows.append(
            {
                "image": image,
                "image_id": image_id,
                "plan_text": trace.text,
                "plan_token_count": int(trace.token_ids.shape[1]),
                "latent_source_tokens": int(trace.hidden_states.shape[1]),
            }
        )
        for seed in config["generation"]["seeds"]:
            for mode in modes:
                resolved_seed = int(seed) + image_index
                torch.manual_seed(resolved_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(resolved_seed)
                candidates = generate_generator_candidates(
                    model,
                    processor,
                    process_vision_info,
                    bridge,
                    trace,
                    image=image,
                    mode=mode,
                    num_candidates=int(config["generation"]["candidates_per_group"]),
                    max_new_tokens=int(config["generation"]["max_new_tokens"]),
                    temperature=float(config["generation"]["temperature"]),
                    top_p=float(config["generation"]["top_p"]),
                    top_k=config["generation"].get("top_k"),
                )
                outputs[(mode, seed)].append(
                    {
                        "image": image,
                        "image_id": image_id,
                        "candidates": candidates,
                        "seed": int(seed),
                        "resolved_image_seed": resolved_seed,
                        "communication_mode": mode,
                    }
                )
        print(f"[latent-eval] {image_index + 1}/{len(rows)} {image_id}")

    for (mode, seed), values in outputs.items():
        write_jsonl(output_dir / f"{mode}_seed{seed}.jsonl", values)
    write_jsonl(output_dir / "online_planner_traces.jsonl", trace_rows)
    manifest = {
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "bridge_checkpoint": str(bridge_file),
        "bridge_checkpoint_sha256": sha256(bridge_file),
        "images": len(rows),
        "modes": modes,
        "planner_called_online": True,
        "same_planner_trace_shared_across_modes": True,
        "test47_read": False,
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
