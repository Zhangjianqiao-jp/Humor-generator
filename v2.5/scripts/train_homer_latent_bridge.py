#!/usr/bin/env python3
"""Train only the typed HOMER conflict/imagination bridge; freeze both 7B policies."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.latent_communication.bridge import TypedHomerLatentBridge, inject_latent_slots, insert_constant_slots
from src.latent_communication.qwen_pipeline import HomerPlannerTrace, PlannerTrace, build_image_message, generate_homer_plan_trace, homer_generator_prompt, model_device
from src.models.qwen_vl_dual_adapter import assert_only_bridge_trainable, load_shared_qwen_vl_adapters
from src.utils.io import read_jsonl


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HomerTraceCache:
    def __init__(self, root: Path, identity: dict[str, Any]) -> None:
        self.root, self.identity = root, identity
        root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".pt")

    def get(self, key: str) -> HomerPlannerTrace | None:
        path = self.path(key)
        if not path.exists():
            return None
        value = torch.load(path, map_location="cpu", weights_only=False)
        return value["trace"] if value.get("identity") == self.identity else None

    def put(self, key: str, trace: HomerPlannerTrace) -> None:
        torch.save({"identity": self.identity, "trace": trace}, self.path(key))


def unique_clusters(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(row)
    rng = random.Random(seed)
    result = [rng.choice(grouped[key]) for key in sorted(grouped)]
    rng.shuffle(result)
    return result


def encode(
    model: Any, processor: Any, process_vision_info: Any, bridge: TypedHomerLatentBridge,
    trace: HomerPlannerTrace, row: dict[str, Any], *, mode: str,
) -> dict[str, torch.Tensor]:
    prompt = homer_generator_prompt(mode, trace)
    messages = build_image_message(row["image"], prompt)
    full = messages + [{"role": "assistant", "content": [{"type": "text", "text": row["caption"]}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    prompt_images, prompt_videos = process_vision_info(messages)
    full_images, full_videos = process_vision_info(full)
    prompt_batch = processor(text=[prompt_text], images=prompt_images, videos=prompt_videos, return_tensors="pt")
    batch = processor(text=[full_text], images=full_images, videos=full_videos, return_tensors="pt").to(model_device(model))
    prompt_len = int(prompt_batch["attention_mask"][0].sum())
    labels = batch["input_ids"].clone()
    labels[:, :prompt_len] = -100
    embeddings = model.get_input_embeddings()(batch["input_ids"])
    dtype = next(bridge.parameters()).dtype
    slots = bridge(
        trace.conflict.hidden_states.to(model_device(model), dtype=dtype),
        trace.imagination_hidden_states.to(model_device(model), dtype=dtype),
    )["latent_slots"]
    placeholder = processor.tokenizer.pad_token_id
    if placeholder is None:
        placeholder = processor.tokenizer.eos_token_id
    inserted = inject_latent_slots(
        batch["input_ids"], embeddings, batch["attention_mask"], slots.to(embeddings.dtype),
        torch.tensor([prompt_len], device=model_device(model)), placeholder_token_id=int(placeholder), labels=labels,
    )
    result = {key: value for key, value in batch.items() if key not in {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}}
    result.update(inserted)
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in batch:
            result[key] = insert_constant_slots(batch[key], torch.tensor([prompt_len], device=model_device(model)), slots.shape[1], value=0)
    return result


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["training"]["seed"])
    random.seed(seed); torch.manual_seed(seed)
    model, processor, process_vision_info = load_shared_qwen_vl_adapters(
        cfg["model"]["model_name"], cfg["planner"]["adapter_dir"], cfg["generator"]["adapter_dir"],
        device_map=cfg["model"].get("device_map", "auto"), torch_dtype=cfg["model"].get("torch_dtype", "bfloat16"),
        load_in_4bit=bool(cfg["model"].get("load_in_4bit", True)),
    )
    embedding = model.get_input_embeddings().weight
    hidden = int(embedding.shape[1])
    target_norm = float(embedding[:4096].float().norm(dim=-1).mean())
    bcfg = cfg["bridge"]
    bridge = TypedHomerLatentBridge(
        hidden, hidden, bottleneck_dim=int(bcfg["bottleneck_dim"]),
        num_slots=int(bcfg["num_slots_per_channel"]), num_heads=int(bcfg["num_heads"]),
        dropout=float(bcfg["dropout"]), target_norm=target_norm,
    ).to(model_device(model), dtype=torch.bfloat16)
    print("[freeze-gate]", assert_only_bridge_trainable(model, bridge))
    train_rows, val_rows = read_jsonl(Path(cfg["data"]["train_path"])), read_jsonl(Path(cfg["data"]["val_path"]))
    all_clusters = {row["cluster_id"] for row in train_rows + val_rows + read_jsonl(Path(cfg["data"]["test_path"]))}
    all_count = len(train_rows) + len(val_rows) + len(read_jsonl(Path(cfg["data"]["test_path"])))
    if not args.smoke and (all_count < int(cfg["data"]["min_formal_dataset_rows"]) or len(all_clusters) < int(cfg["data"]["min_formal_unique_images"])):
        raise RuntimeError(f"formal-data gate failed: rows={all_count}, unique_clusters={len(all_clusters)}")
    if args.smoke:
        train_rows, val_rows = train_rows[:2], val_rows[:1]
    output = args.output_dir or Path(cfg["output"]["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    identity = {"protocol": "homer_staged_v1", "planner_adapter": cfg["planner"]["adapter_dir"], "max_state_tokens": cfg["planner"]["max_state_tokens"]}
    cache = HomerTraceCache(output / "planner_state_cache", identity)
    def trace_for(row: dict[str, Any]) -> HomerPlannerTrace:
        key = str(row["cluster_id"])
        trace = cache.get(key)
        if trace is None:
            trace = generate_homer_plan_trace(
                model, processor, process_vision_info, image=row["image"],
                max_new_tokens=int(cfg["planner"]["max_new_tokens"]),
                max_state_tokens=int(cfg["planner"]["max_state_tokens"]),
                image_max_pixels=int(cfg["data"]["image_max_pixels"]),
            )
            cache.put(key, trace)
        return trace
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    epochs = 1 if args.smoke else int(cfg["training"]["num_epochs"])
    mixes = list(bcfg["curriculum"]["token_to_latent_mix"])
    best, history = math.inf, []
    for epoch in range(epochs):
        bridge.train(); optimizer.zero_grad(set_to_none=True); total = 0.0
        rows = unique_clusters(train_rows, seed + epoch)
        latent_probability = float(mixes[min(epoch * len(mixes) // epochs, len(mixes)-1)])
        for index, row in enumerate(rows):
            trace = trace_for(row); model.set_adapter("generator")
            mode = "latent" if random.random() < latent_probability else "hybrid"
            batch = encode(model, processor, process_vision_info, bridge, trace, row, mode=mode)
            loss = model(**batch, use_cache=False).loss
            (loss / int(cfg["training"]["gradient_accumulation_steps"])).backward(); total += float(loss.detach())
            accum = int(cfg["training"]["gradient_accumulation_steps"])
            if (index + 1) % accum == 0 or index + 1 == len(rows):
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), float(cfg["training"]["max_grad_norm"]))
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        bridge.eval(); losses = []
        with torch.no_grad():
            for row in unique_clusters(val_rows, seed):
                model.set_adapter("generator")
                losses.append(float(model(**encode(model, processor, process_vision_info, bridge, trace_for(row), row, mode="latent"), use_cache=False).loss))
        record = {"epoch": epoch+1, "latent_probability": latent_probability, "train_nll": total/max(len(rows),1), "val_nll": sum(losses)/max(len(losses),1)}
        history.append(record); print(json.dumps(record))
        payload = {"state_dict": bridge.state_dict(), "config": bcfg, "history": history, "identity": identity}
        torch.save(payload, output / "latest.pt")
        if record["val_nll"] < best:
            best = record["val_nll"]; torch.save(payload, output / "best.pt")
    (output / "manifest.json").write_text(json.dumps({"config_sha256": digest(args.config), "bridge_only": True, "policies_frozen": True, "rows": all_count, "unique_clusters": len(all_clusters), "history": history}, indent=2)+"\n")


if __name__ == "__main__":
    main()
