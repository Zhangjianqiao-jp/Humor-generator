#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_humor_reranker import (  # noqa: E402
    HumorReranker,
    RerankerHead,
    autocast_context,
    load_backbone,
    move_tensors,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_candidate(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu")
    config = state.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    return state, config


def infer_head_shape(state: dict[str, Any]) -> tuple[int, int, int]:
    head_state = state["head_state_dict"]
    linear_weights = [(key, value) for key, value in head_state.items() if key.endswith("weight") and value.ndim == 2]
    if len(linear_weights) < 2:
        raise ValueError("Could not infer reranker head shape from checkpoint.")
    input_dim = int(linear_weights[0][1].shape[1])
    hidden_dim = int(linear_weights[0][1].shape[0])
    # All Linear layers except the final score layer are hidden layers.
    num_head_layers = max(1, len(linear_weights) - 1)
    return input_dim, hidden_dim, num_head_layers


@torch.no_grad()
def score_candidates(
    model: HumorReranker,
    processor: Any,
    image_path: str,
    candidates: list[str],
    device: torch.device,
    max_text_len: int,
    bf16: bool,
    fp16: bool,
    candidate_batch_size: int,
) -> list[float]:
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        image_inputs = processor(images=[image_rgb], return_tensors="pt")
    image_inputs = move_tensors(image_inputs, device)
    with autocast_context(device, bf16=bf16, fp16=fp16):
        image_embed = model.encode_image(image_inputs)

    scores: list[float] = []
    for start in range(0, len(candidates), candidate_batch_size):
        batch_candidates = candidates[start : start + candidate_batch_size]
        text_inputs = processor(
            text=batch_candidates,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        text_inputs = move_tensors(text_inputs, device)
        with autocast_context(device, bf16=bf16, fp16=fp16):
            text_embeds = model.encode_text(text_inputs)
            image_embeds = image_embed.expand(text_embeds.shape[0], -1)
            batch_scores = model.score_from_embeds(image_embeds, text_embeds)
        scores.extend([float(value) for value in batch_scores.detach().cpu().tolist()])
    return scores


def main() -> None:
    parser = ArgumentParser(description="Use a trained humor reranker to sort generated caption candidates.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/humor_reranker_v1/mixed/checkpoint_best.pt"))
    parser.add_argument("--input-jsonl", type=Path, default=Path("outputs/generations/v1_5_candidates_clean.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/reranked/v1_5_candidates_reranked.jsonl"))
    parser.add_argument("--backbone-name", type=str, default=None)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--max-text-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bf16", action=BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action=BooleanOptionalAction, default=None)
    parser.add_argument("--local-files-only", action=BooleanOptionalAction, default=None)
    parser.add_argument("--trust-remote-code", action=BooleanOptionalAction, default=None)
    parser.add_argument("--skip-missing-images", action=BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()

    state, checkpoint_config = load_checkpoint_config(args.checkpoint)
    backbone_name = args.backbone_name or str(checkpoint_config.get("backbone_name") or "openai/clip-vit-base-patch32")
    max_text_len = args.max_text_len or int(checkpoint_config.get("max_text_len") or 64)
    bf16 = bool(checkpoint_config.get("bf16", True)) if args.bf16 is None else bool(args.bf16)
    fp16 = bool(checkpoint_config.get("fp16", False)) if args.fp16 is None else bool(args.fp16)
    trust_remote_code = bool(checkpoint_config.get("trust_remote_code", False)) if args.trust_remote_code is None else bool(args.trust_remote_code)
    local_files_only = bool(checkpoint_config.get("local_files_only", False)) if args.local_files_only is None else bool(args.local_files_only)
    normalize_embeddings = bool(checkpoint_config.get("normalize_embeddings", True))

    input_dim, hidden_dim, num_head_layers = infer_head_shape(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader_args = type(
        "LoaderArgs",
        (),
        {
            "backbone_name": backbone_name,
            "torch_dtype": checkpoint_config.get("torch_dtype", "auto"),
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        },
    )()
    print(f"[rerank] loading backbone={backbone_name}")
    backbone, processor = load_backbone(loader_args)
    backbone.eval().to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)

    head = RerankerHead(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_head_layers, dropout=0.0)
    head.load_state_dict(state["head_state_dict"])
    head.eval().to(device)
    model = HumorReranker(backbone=backbone, head=head, normalize_embeddings=normalize_embeddings).eval().to(device)

    rows = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()

    skipped_missing = 0
    skipped_empty = 0
    for row in tqdm(rows, desc="reranking", dynamic_ncols=True, disable=not args.progress):
        image = str(row.get("image") or "")
        if not Path(image).exists():
            skipped_missing += 1
            if args.skip_missing_images:
                continue
            raise FileNotFoundError(image)
        candidates = [clean_candidate(candidate) for candidate in (row.get("candidates") or []) if clean_candidate(candidate)]
        candidates = candidates[: args.max_candidates]
        if not candidates:
            skipped_empty += 1
            continue
        scores = score_candidates(
            model=model,
            processor=processor,
            image_path=image,
            candidates=candidates,
            device=device,
            max_text_len=max_text_len,
            bf16=bf16,
            fp16=fp16,
            candidate_batch_size=args.candidate_batch_size,
        )
        ranked = sorted(
            [
                {
                    "rank": 0,
                    "candidate_index": index,
                    "caption": candidate,
                    "reranker_score": score,
                }
                for index, (candidate, score) in enumerate(zip(candidates, scores), start=1)
            ],
            key=lambda item: item["reranker_score"],
            reverse=True,
        )
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        best = ranked[0]
        append_jsonl(
            args.output_jsonl,
            {
                "image": row.get("image"),
                "image_id": row.get("image_id"),
                "gold_caption": row.get("gold_caption"),
                "gold_captions": row.get("gold_captions") or [],
                "prompt": row.get("prompt"),
                "selected_caption": best["caption"],
                "selected_score": best["reranker_score"],
                "selected_original_index": best["candidate_index"],
                "ranked_candidates": ranked[: args.top_k],
                "num_candidates": len(candidates),
                "checkpoint": str(args.checkpoint),
            },
        )
    print(f"[rerank] saved to {args.output_jsonl}")
    print(f"[rerank] skipped_missing={skipped_missing} skipped_empty={skipped_empty}")


if __name__ == "__main__":
    main()
