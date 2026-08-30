#!/usr/bin/env python3
"""Train only a latent Planner->Generator bridge while both 7B policies stay frozen."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.latent_communication.bridge import LearnedLatentBridge, inject_latent_slots, insert_constant_slots
from src.latent_communication.qwen_pipeline import (
    PlannerTrace,
    build_image_message,
    generate_planner_trace,
    generator_prompt,
    model_device,
)
from src.models.qwen_vl_dual_adapter import assert_only_bridge_trainable, load_shared_qwen_vl_adapters
from src.training.sft_dataset import extract_caption, extract_image_path
from src.utils.io import read_jsonl


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def unique_epoch_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("image_id") or extract_image_path(row))
        grouped[key].append(row)
    rng = random.Random(seed)
    selected = [rng.choice(grouped[key]) for key in sorted(grouped)]
    rng.shuffle(selected)
    return selected


class PlannerTraceCache:
    def __init__(self, path: Path, identity: dict[str, Any]) -> None:
        self.path = path
        self.identity = identity
        self.memory: dict[str, PlannerTrace] = {}
        path.mkdir(parents=True, exist_ok=True)

    def _path(self, image_id: str) -> Path:
        safe = hashlib.sha256(image_id.encode()).hexdigest()[:20]
        return self.path / f"{safe}.pt"

    def get(self, image_id: str) -> PlannerTrace | None:
        if image_id in self.memory:
            return self.memory[image_id]
        path = self._path(image_id)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("identity") != self.identity:
            return None
        trace = PlannerTrace(
            text=str(payload["text"]),
            token_ids=payload["token_ids"],
            hidden_states=payload["hidden_states"],
        )
        self.memory[image_id] = trace
        return trace

    def put(self, image_id: str, trace: PlannerTrace) -> None:
        self.memory[image_id] = trace
        torch.save(
            {
                "identity": self.identity,
                "text": trace.text,
                "token_ids": trace.token_ids.cpu(),
                "hidden_states": trace.hidden_states.cpu(),
            },
            self._path(image_id),
        )


def encode_caption_example(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    bridge: LearnedLatentBridge,
    trace: PlannerTrace,
    row: dict[str, Any],
    *,
    image_max_pixels: int | None,
) -> dict[str, torch.Tensor]:
    image = str(extract_image_path(row) or "")
    answer = str(extract_caption(row) or "").strip()
    if not image or not answer:
        raise ValueError("Bridge SFT row is missing image or assistant caption")
    prompt = generator_prompt("latent", trace.text)
    messages = build_image_message(image, prompt)
    if image_max_pixels is not None:
        messages[0]["content"][0]["max_pixels"] = int(image_max_pixels)
    full_messages = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_images, prompt_videos = process_vision_info(messages)
    full_images, full_videos = process_vision_info(full_messages)
    prompt_encoded = processor(
        text=[prompt_text], images=prompt_images, videos=prompt_videos, return_tensors="pt"
    )
    encoded = processor(
        text=[full_text], images=full_images, videos=full_videos, return_tensors="pt"
    )
    device = model_device(model)
    encoded = encoded.to(device)
    prompt_encoded = prompt_encoded.to(device)
    prompt_len = int(prompt_encoded["attention_mask"][0].sum().item())
    labels = encoded["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[encoded["attention_mask"] == 0] = -100
    if int(labels.ne(-100).sum()) < 1:
        raise ValueError("Caption answer was fully truncated")

    sender = trace.hidden_states.to(device=device, dtype=next(bridge.parameters()).dtype)
    latent = bridge(sender).latent_slots
    embeddings = model.get_input_embeddings()(encoded["input_ids"])
    placeholder = processor.tokenizer.pad_token_id
    if placeholder is None:
        placeholder = processor.tokenizer.eos_token_id
    inserted = inject_latent_slots(
        encoded["input_ids"],
        embeddings,
        encoded["attention_mask"],
        latent.to(embeddings.dtype),
        torch.tensor([prompt_len], device=device),
        placeholder_token_id=int(placeholder),
        labels=labels,
    )
    result = {
        key: value
        for key, value in encoded.items()
        if key not in {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
    }
    result.update(inserted)
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in encoded:
            result[key] = insert_constant_slots(
                encoded[key],
                torch.tensor([prompt_len], device=device),
                latent.shape[1],
                value=0,
            )
    return result


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Optional isolated output override")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    planner_prompt_path = Path(config["planner"]["prompt_file"])
    planner_prompt = planner_prompt_path.read_text(encoding="utf-8").strip()
    planner_adapter = Path(config["planner"]["adapter_dir"])
    generator_adapter = Path(config["generator"]["adapter_dir"])
    model, processor, process_vision_info = load_shared_qwen_vl_adapters(
        config["model"]["model_name"],
        planner_adapter,
        generator_adapter,
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
        load_in_4bit=bool(config["model"].get("load_in_4bit", True)),
    )
    embedding = model.get_input_embeddings().weight
    target_norm = float(embedding[: min(4096, embedding.shape[0])].float().norm(dim=-1).mean())
    hidden_size = int(embedding.shape[1])
    bridge = LearnedLatentBridge(
        hidden_size,
        hidden_size,
        bottleneck_dim=int(config["bridge"]["bottleneck_dim"]),
        num_slots=int(config["bridge"]["num_slots"]),
        num_heads=int(config["bridge"]["num_heads"]),
        dropout=float(config["bridge"]["dropout"]),
        target_norm=target_norm,
    ).to(model_device(model), dtype=torch.bfloat16)
    trainable = assert_only_bridge_trainable(model, bridge)
    print(f"[freeze-gate] {trainable}")

    output_dir = args.output_dir or Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "planner_adapter": str(planner_adapter),
        "planner_adapter_config_sha256": file_sha256(planner_adapter / "adapter_config.json"),
        "planner_prompt_sha256": text_sha256(planner_prompt),
        "max_new_tokens": int(config["planner"]["max_new_tokens"]),
        "max_state_tokens": int(config["planner"]["max_state_tokens"]),
        "decode": "greedy",
    }
    cache = PlannerTraceCache(output_dir / "planner_state_cache", identity)

    def trace_for(row: dict[str, Any]) -> PlannerTrace:
        image_id = str(row.get("image_id") or extract_image_path(row))
        cached = cache.get(image_id)
        if cached is not None:
            return cached
        trace = generate_planner_trace(
            model,
            processor,
            process_vision_info,
            image=str(extract_image_path(row)),
            planner_prompt=planner_prompt,
            max_new_tokens=int(config["planner"]["max_new_tokens"]),
            max_state_tokens=int(config["planner"]["max_state_tokens"]),
        )
        cache.put(image_id, trace)
        return trace

    train_path = Path(config["data"]["train_path"])
    val_path = Path(config["data"]["val_path"])
    train_rows = read_jsonl(train_path)
    val_rows = unique_epoch_rows(read_jsonl(val_path), seed)
    epochs = 1 if args.smoke else int(config["training"]["num_epochs"])
    if args.smoke:
        train_rows, val_rows = train_rows[:2], val_rows[:1]
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    grad_accum = int(config["training"]["gradient_accumulation_steps"])
    best_val = math.inf
    history: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(epochs):
        bridge.train()
        epoch_rows = (
            unique_epoch_rows(train_rows, seed + epoch)
            if config["data"].get("sample_one_caption_per_image", True) and not args.smoke
            else list(train_rows)
        )
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for index, row in enumerate(epoch_rows):
            trace = trace_for(row)
            model.set_adapter("generator")
            batch = encode_caption_example(
                model,
                processor,
                process_vision_info,
                bridge,
                trace,
                row,
                image_max_pixels=config["data"].get("image_max_pixels"),
            )
            loss = model(**batch, use_cache=False).loss / grad_accum
            loss.backward()
            train_loss += float(loss.detach()) * grad_accum
            if (index + 1) % grad_accum == 0 or index + 1 == len(epoch_rows):
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), float(config["training"]["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        bridge.eval()
        val_losses = []
        for row in val_rows:
            trace = trace_for(row)
            model.set_adapter("generator")
            with torch.no_grad():
                batch = encode_caption_example(
                    model,
                    processor,
                    process_vision_info,
                    bridge,
                    trace,
                    row,
                    image_max_pixels=config["data"].get("image_max_pixels"),
                )
                val_losses.append(float(model(**batch, use_cache=False).loss))
        record = {
            "epoch": epoch + 1,
            "optimizer_steps": global_step,
            "train_loss": train_loss / max(len(epoch_rows), 1),
            "val_loss": sum(val_losses) / max(len(val_losses), 1),
        }
        history.append(record)
        print(json.dumps(record))
        torch.save({"state_dict": bridge.state_dict(), "config": config["bridge"]}, output_dir / "latest.pt")
        if record["val_loss"] < best_val:
            best_val = record["val_loss"]
            torch.save({"state_dict": bridge.state_dict(), "config": config["bridge"]}, output_dir / "best.pt")

    manifest = {
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "train_path": str(train_path),
        "train_sha256": file_sha256(train_path),
        "val_path": str(val_path),
        "val_sha256": file_sha256(val_path),
        "planner_trace_identity": identity,
        "generator_adapter": str(generator_adapter),
        "generator_adapter_config_sha256": file_sha256(generator_adapter / "adapter_config.json"),
        "trainable": trainable,
        "bridge_only": True,
        "policies_frozen": True,
        "smoke": args.smoke,
        "history": history,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
