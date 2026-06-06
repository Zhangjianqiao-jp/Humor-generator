#!/usr/bin/env python
from __future__ import annotations

import json
import math
import random
import shutil
import sys
import time
from argparse import ArgumentParser, BooleanOptionalAction
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cfg_get(config: dict[str, Any], dotted: str, default: Any) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def pick(value: Any, config: dict[str, Any], dotted: str, default: Any) -> Any:
    return cfg_get(config, dotted, default) if value is None else value


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None or str(name).lower() in {"", "auto", "none"}:
        return None
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def autocast_context(device: torch.device, bf16: bool, fp16: bool):
    if device.type != "cuda":
        return nullcontext()
    if bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], use_row_loss_weight: bool = True, default_loss_weight: float = 1.0):
        self.rows = rows
        self.use_row_loss_weight = use_row_loss_weight
        self.default_loss_weight = default_loss_weight

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        weight = float(row.get("loss_weight", self.default_loss_weight)) if self.use_row_loss_weight else self.default_loss_weight
        return {
            "image": str(row["image"]),
            "image_id": str(row.get("image_id", "")),
            "positive": str(row["positive"]).strip(),
            "negative": str(row["negative"]).strip(),
            "pos_score": float(row.get("pos_score", 0.0)),
            "neg_score": float(row.get("neg_score", 0.0)),
            "score_gap": float(row.get("score_gap", 0.0)),
            "pair_type": str(row.get("pair_type", "")),
            "loss_weight": weight,
        }


class PairCollator:
    def __init__(self, processor: Any, max_text_len: int):
        self.processor = processor
        self.max_text_len = max_text_len

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = []
        for example in examples:
            with Image.open(example["image"]) as image:
                images.append(image.convert("RGB"))
        positives = [example["positive"] for example in examples]
        negatives = [example["negative"] for example in examples]
        texts = positives + negatives
        image_inputs = self.processor(images=images, return_tensors="pt")
        text_inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return {
            "image_inputs": image_inputs,
            "text_inputs": text_inputs,
            "loss_weight": torch.tensor([example["loss_weight"] for example in examples], dtype=torch.float32),
            "pos_score": torch.tensor([example["pos_score"] for example in examples], dtype=torch.float32),
            "neg_score": torch.tensor([example["neg_score"] for example in examples], dtype=torch.float32),
            "score_gap": torch.tensor([example["score_gap"] for example in examples], dtype=torch.float32),
            "meta": examples,
        }


def move_tensors(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    # HuggingFace BatchFeature/BatchEncoding objects are not always plain dicts,
    # but they expose .to(device). Leaving them on CPU breaks CLIP/SigLIP forward.
    if hasattr(data, "to") and callable(getattr(data, "to")):
        return data.to(device)
    if isinstance(data, Mapping):
        return {key: move_tensors(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_tensors(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(move_tensors(value, device) for value in data)
    return data


def extract_embedding(output: Any, preferred_attr: str) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr in (preferred_attr, "pooler_output", "text_embeds", "image_embeds"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    last_hidden = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden, torch.Tensor):
        return last_hidden.mean(dim=1)
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"Could not extract tensor embedding from output type={type(output)!r}")


class RerankerHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class HumorReranker(nn.Module):
    def __init__(self, backbone: nn.Module, head: RerankerHead, normalize_embeddings: bool = True):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.normalize_embeddings = normalize_embeddings

    def encode_image(self, image_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.backbone, "get_image_features"):
            embeds = extract_embedding(self.backbone.get_image_features(**image_inputs), "image_embeds")
        else:
            embeds = extract_embedding(self.backbone(**image_inputs), "image_embeds")
        return F.normalize(embeds.float(), dim=-1) if self.normalize_embeddings else embeds.float()

    def encode_text(self, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.backbone, "get_text_features"):
            embeds = extract_embedding(self.backbone.get_text_features(**text_inputs), "text_embeds")
        else:
            embeds = extract_embedding(self.backbone(**text_inputs), "text_embeds")
        return F.normalize(embeds.float(), dim=-1) if self.normalize_embeddings else embeds.float()

    @staticmethod
    def build_features(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        if image_embeds.shape[-1] == text_embeds.shape[-1]:
            cosine = (image_embeds * text_embeds).sum(dim=-1, keepdim=True)
            return torch.cat(
                [image_embeds, text_embeds, image_embeds * text_embeds, (image_embeds - text_embeds).abs(), cosine],
                dim=-1,
            )
        return torch.cat([image_embeds, text_embeds], dim=-1)

    def score_from_embeds(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        return self.head(self.build_features(image_embeds, text_embeds))


def encode_batch(
    model: HumorReranker,
    batch: dict[str, Any],
    device: torch.device,
    freeze_backbone: bool,
    bf16: bool,
    fp16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_inputs = move_tensors(batch["image_inputs"], device)
    text_inputs = move_tensors(batch["text_inputs"], device)
    batch_size = batch["loss_weight"].shape[0]

    grad_context = torch.no_grad() if freeze_backbone else nullcontext()
    with grad_context:
        with autocast_context(device, bf16=bf16, fp16=fp16):
            image_embeds = model.encode_image(image_inputs)
            text_embeds = model.encode_text(text_inputs)
    text_pos, text_neg = text_embeds[:batch_size], text_embeds[batch_size:]
    return image_embeds, text_pos, text_neg, batch["loss_weight"].to(device)


def compute_loss(
    model: HumorReranker,
    image_embeds: torch.Tensor,
    text_pos: torch.Tensor,
    text_neg: torch.Tensor,
    loss_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    score_pos = model.score_from_embeds(image_embeds, text_pos)
    score_neg = model.score_from_embeds(image_embeds, text_neg)
    margin = score_pos - score_neg
    per_example_loss = F.softplus(-margin)
    loss = (per_example_loss * loss_weight).sum() / loss_weight.clamp_min(1e-6).sum()
    accuracy = (margin > 0).float().mean()
    return {
        "loss": loss,
        "accuracy": accuracy,
        "margin": margin.mean(),
        "score_pos": score_pos.mean(),
        "score_neg": score_neg.mean(),
    }


def split_by_image(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_ids = sorted({str(row.get("image_id") or row.get("image")) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    val_count = max(1, int(round(len(image_ids) * val_ratio))) if val_ratio > 0 else 0
    val_ids = set(image_ids[:val_count])
    train_rows = [row for row in rows if str(row.get("image_id") or row.get("image")) not in val_ids]
    val_rows = [row for row in rows if str(row.get("image_id") or row.get("image")) in val_ids]
    return train_rows, val_rows


def choose_rows(
    stage: str,
    pair_jsonl: Path,
    weak_pair_jsonl: Path | None,
    weak_mix_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if stage == "strong":
        return read_jsonl(pair_jsonl)
    if stage == "weak":
        if weak_pair_jsonl is None:
            raise ValueError("--weak-pair-jsonl is required for stage=weak")
        return read_jsonl(weak_pair_jsonl)
    if stage == "mixed":
        if weak_pair_jsonl is None:
            raise ValueError("--weak-pair-jsonl is required for stage=mixed")
        strong = read_jsonl(pair_jsonl)
        weak = read_jsonl(weak_pair_jsonl)
        rng = random.Random(seed)
        rng.shuffle(weak)
        weak_mix_ratio = max(0.0, min(0.95, weak_mix_ratio))
        weak_target = int(len(strong) * weak_mix_ratio / max(1e-6, 1.0 - weak_mix_ratio))
        rows = strong + weak[: min(len(weak), weak_target)]
        rng.shuffle(rows)
        return rows
    raise ValueError(f"Unsupported stage: {stage}")


def sample_rows(rows: list[dict[str, Any]], max_rows: int | None, seed: int) -> list[dict[str, Any]]:
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    sampled = rows[:]
    rng.shuffle(sampled)
    return sampled[:max_rows]


def filter_existing_images(rows: list[dict[str, Any]], skip_missing_images: bool) -> list[dict[str, Any]]:
    exists_cache: dict[str, bool] = {}
    kept: list[dict[str, Any]] = []
    missing_rows = 0
    missing_images: set[str] = set()
    for row in rows:
        image = str(row.get("image", ""))
        exists = exists_cache.get(image)
        if exists is None:
            exists = Path(image).exists()
            exists_cache[image] = exists
        if exists:
            kept.append(row)
        else:
            missing_rows += 1
            missing_images.add(image)
    if missing_rows:
        sample = sorted(missing_images)[:8]
        message = (
            f"[data] missing images: rows={missing_rows}/{len(rows)} "
            f"unique_images={len(missing_images)} sample={sample}"
        )
        if not skip_missing_images:
            raise FileNotFoundError(message)
        print(message)
        print(f"[data] skipped missing-image rows; kept={len(kept)}")
    return kept


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    output_dir: Path,
    name: str,
    state: dict[str, Any],
    backbone: nn.Module,
    processor: Any,
    save_backbone: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    latest = output_dir / "checkpoint_last.pt"
    if path != latest:
        shutil.copy2(path, latest)
    if save_backbone:
        backbone_dir = output_dir / "backbone"
        backbone.save_pretrained(backbone_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir / "processor")


@torch.no_grad()
def evaluate(
    model: HumorReranker,
    loader: DataLoader,
    device: torch.device,
    freeze_backbone: bool,
    bf16: bool,
    fp16: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_margin = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        image_embeds, text_pos, text_neg, loss_weight = encode_batch(
            model, batch, device, freeze_backbone=freeze_backbone, bf16=bf16, fp16=fp16
        )
        metrics = compute_loss(model, image_embeds, text_pos, text_neg, loss_weight)
        count = int(batch["loss_weight"].shape[0])
        total_loss += float(metrics["loss"].detach().cpu()) * count
        total_acc += float(metrics["accuracy"].detach().cpu()) * count
        total_margin += float(metrics["margin"].detach().cpu()) * count
        total_pos += float(metrics["score_pos"].detach().cpu()) * count
        total_neg += float(metrics["score_neg"].detach().cpu()) * count
        total_count += count
    if total_count == 0:
        return {"loss": 0.0, "accuracy": 0.0, "margin": 0.0, "score_pos": 0.0, "score_neg": 0.0}
    return {
        "loss": total_loss / total_count,
        "accuracy": total_acc / total_count,
        "margin": total_margin / total_count,
        "score_pos": total_pos / total_count,
        "score_neg": total_neg / total_count,
    }


def load_backbone(args: Any) -> tuple[Any, Any]:
    dtype = torch_dtype_from_name(args.torch_dtype)
    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    backbone = AutoModel.from_pretrained(args.backbone_name, **kwargs)
    processor = AutoProcessor.from_pretrained(
        args.backbone_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    return backbone, processor


def infer_feature_dim(
    backbone: nn.Module,
    processor: Any,
    probe_batch: dict[str, Any],
    device: torch.device,
    freeze_backbone: bool,
    bf16: bool,
    fp16: bool,
) -> int:
    dummy_head = RerankerHead(input_dim=1, hidden_dim=1, num_layers=1, dropout=0.0).to(device)
    model = HumorReranker(backbone=backbone, head=dummy_head).to(device)
    image_embeds, text_pos, _, _ = encode_batch(model, probe_batch, device, freeze_backbone, bf16, fp16)
    features = model.build_features(image_embeds, text_pos)
    return int(features.shape[-1])


def print_batch_debug(batch: dict[str, Any]) -> None:
    print("=" * 80)
    for index, item in enumerate(batch["meta"][:3]):
        print(f"[debug {index}] image: {item['image']}")
        print(f"[debug {index}] positive: {item['positive']}")
        print(f"[debug {index}] negative: {item['negative']}")
        print(f"[debug {index}] pos_score={item['pos_score']} neg_score={item['neg_score']} gap={item['score_gap']} type={item['pair_type']}")
    print("=" * 80)


def main() -> None:
    parser = ArgumentParser(description="Train a contrastive pairwise humor caption reranker.")
    parser.add_argument("--config", type=Path, default=Path("configs/humor_reranker.yaml"))
    parser.add_argument("--stage", choices=["strong", "mixed", "weak"], default=None)
    parser.add_argument("--pair-jsonl", type=Path, default=None)
    parser.add_argument("--weak-pair-jsonl", type=Path, default=None)
    parser.add_argument("--weak-mix-ratio", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--backbone-name", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--trust-remote-code", action=BooleanOptionalAction, default=None)
    parser.add_argument("--local-files-only", action=BooleanOptionalAction, default=None)
    parser.add_argument("--freeze-backbone", action=BooleanOptionalAction, default=None)
    parser.add_argument("--save-backbone", action=BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing", action=BooleanOptionalAction, default=None)
    parser.add_argument("--normalize-embeddings", action=BooleanOptionalAction, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-head-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-text-len", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--bf16", action=BooleanOptionalAction, default=None)
    parser.add_argument("--fp16", action=BooleanOptionalAction, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--skip-missing-images", action=BooleanOptionalAction, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-row-loss-weight", action=BooleanOptionalAction, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Load only reranker head weights and start a fresh optimizer/scheduler.")
    parser.add_argument("--progress", action=BooleanOptionalAction, default=None)
    parser.add_argument("--debug-batch", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    args.stage = pick(args.stage, config, "data.stage", "strong")
    args.pair_jsonl = Path(pick(args.pair_jsonl, config, "data.pair_jsonl", Path("data/processed/reranker_score_pools_strict/strong_pairs.jsonl")))
    weak_pair_value = pick(args.weak_pair_jsonl, config, "data.weak_pair_jsonl", Path("data/processed/reranker_score_pools_strict/weak_pairs.jsonl"))
    args.weak_pair_jsonl = None if weak_pair_value in {None, ""} else Path(weak_pair_value)
    args.weak_mix_ratio = float(pick(args.weak_mix_ratio, config, "data.weak_mix_ratio", 0.30))
    args.output_dir = Path(pick(args.output_dir, config, "output.output_dir", Path("outputs/humor_reranker_v1")))

    args.backbone_name = str(pick(args.backbone_name, config, "model.backbone_name", "openai/clip-vit-base-patch32"))
    args.torch_dtype = pick(args.torch_dtype, config, "model.torch_dtype", "auto")
    args.trust_remote_code = bool(pick(args.trust_remote_code, config, "model.trust_remote_code", False))
    args.local_files_only = bool(pick(args.local_files_only, config, "model.local_files_only", False))
    args.freeze_backbone = bool(pick(args.freeze_backbone, config, "model.freeze_backbone", True))
    args.save_backbone = bool(pick(args.save_backbone, config, "model.save_backbone", False))
    args.gradient_checkpointing = bool(pick(args.gradient_checkpointing, config, "model.gradient_checkpointing", False))
    args.normalize_embeddings = bool(pick(args.normalize_embeddings, config, "model.normalize_embeddings", True))
    args.hidden_dim = int(pick(args.hidden_dim, config, "model.hidden_dim", 1024))
    args.num_head_layers = int(pick(args.num_head_layers, config, "model.num_head_layers", 2))
    args.dropout = float(pick(args.dropout, config, "model.dropout", 0.10))
    args.max_text_len = int(pick(args.max_text_len, config, "data.max_text_len", 64))

    args.batch_size = int(pick(args.batch_size, config, "training.batch_size", 128))
    args.gradient_accumulation_steps = int(pick(args.gradient_accumulation_steps, config, "training.gradient_accumulation_steps", 1))
    args.num_epochs = float(pick(args.num_epochs, config, "training.num_epochs", 3))
    args.learning_rate = float(pick(args.learning_rate, config, "training.learning_rate", 1.0e-4))
    args.weight_decay = float(pick(args.weight_decay, config, "training.weight_decay", 0.01))
    args.warmup_ratio = float(pick(args.warmup_ratio, config, "training.warmup_ratio", 0.03))
    args.max_grad_norm = float(pick(args.max_grad_norm, config, "training.max_grad_norm", 1.0))
    args.bf16 = bool(pick(args.bf16, config, "training.bf16", True))
    args.fp16 = bool(pick(args.fp16, config, "training.fp16", False))
    args.num_workers = int(pick(args.num_workers, config, "training.num_workers", 4))
    args.skip_missing_images = bool(pick(args.skip_missing_images, config, "training.skip_missing_images", True))
    args.val_ratio = float(pick(args.val_ratio, config, "training.val_ratio", 0.02))
    args.max_train_samples = pick(args.max_train_samples, config, "training.max_train_samples", None)
    args.max_val_samples = pick(args.max_val_samples, config, "training.max_val_samples", None)
    args.eval_max_batches = pick(args.eval_max_batches, config, "training.eval_max_batches", None)
    args.logging_steps = int(pick(args.logging_steps, config, "training.logging_steps", 20))
    args.eval_steps = int(pick(args.eval_steps, config, "training.eval_steps", 500))
    args.save_steps = int(pick(args.save_steps, config, "training.save_steps", 1000))
    args.seed = int(pick(args.seed, config, "training.seed", 42))
    args.use_row_loss_weight = bool(pick(args.use_row_loss_weight, config, "training.use_row_loss_weight", True))
    args.progress = bool(pick(args.progress, config, "training.progress", True))

    if isinstance(args.max_train_samples, str) and args.max_train_samples.lower() == "none":
        args.max_train_samples = None
    if isinstance(args.max_val_samples, str) and args.max_val_samples.lower() == "none":
        args.max_val_samples = None
    if isinstance(args.eval_max_batches, str) and args.eval_max_batches.lower() == "none":
        args.eval_max_batches = None
    args.max_train_samples = None if args.max_train_samples is None else int(args.max_train_samples)
    args.max_val_samples = None if args.max_val_samples is None else int(args.max_val_samples)
    args.eval_max_batches = None if args.eval_max_batches is None else int(args.eval_max_batches)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[reranker] stage={args.stage} pair_jsonl={args.pair_jsonl} weak_pair_jsonl={args.weak_pair_jsonl}")
    rows = choose_rows(args.stage, args.pair_jsonl, args.weak_pair_jsonl, args.weak_mix_ratio, args.seed)
    rows = filter_existing_images(rows, skip_missing_images=args.skip_missing_images)
    train_rows, val_rows = split_by_image(rows, args.val_ratio, args.seed)
    train_rows = sample_rows(train_rows, args.max_train_samples, args.seed + 1)
    val_rows = sample_rows(val_rows, args.max_val_samples, args.seed + 2)
    print(f"[data] rows={len(rows)} train={len(train_rows)} val={len(val_rows)} val_ratio={args.val_ratio}")
    if not train_rows or not val_rows:
        raise ValueError("Train/val split is empty. Lower val_ratio or check pair files.")

    print(f"[model] loading backbone={args.backbone_name} freeze_backbone={args.freeze_backbone}")
    backbone, processor = load_backbone(args)
    if args.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if args.freeze_backbone:
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad_(False)
    else:
        backbone.train()
    backbone.to(device)

    collator = PairCollator(processor=processor, max_text_len=args.max_text_len)
    train_loader = DataLoader(
        PairDataset(train_rows, use_row_loss_weight=args.use_row_loss_weight),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        PairDataset(val_rows, use_row_loss_weight=args.use_row_loss_weight),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        drop_last=False,
    )

    probe_batch = next(iter(train_loader))
    if args.debug_batch:
        print_batch_debug(probe_batch)
    feature_dim = infer_feature_dim(
        backbone, processor, probe_batch, device, args.freeze_backbone, args.bf16, args.fp16
    )
    print(f"[model] inferred feature_dim={feature_dim}")
    head = RerankerHead(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_head_layers,
        dropout=args.dropout,
    ).to(device)
    model = HumorReranker(backbone=backbone, head=head, normalize_embeddings=args.normalize_embeddings).to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for param in trainable_params)
    total_count = sum(param.numel() for param in model.parameters())
    print(f"[model] trainable parameters: {trainable_count:,}/{total_count:,} ({100 * trainable_count / max(1, total_count):.4f}%)")

    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        model.head.load_state_dict(state["head_state_dict"])
        print(f"[init] loaded head weights from {args.init_checkpoint}; optimizer/scheduler will start fresh")

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    total_update_steps = max(1, int(math.ceil(update_steps_per_epoch * args.num_epochs)))
    scheduler = make_scheduler(optimizer, total_update_steps, args.warmup_ratio)
    start_step = 0
    best_val_acc = -1.0

    if args.resume_checkpoint is not None:
        state = torch.load(args.resume_checkpoint, map_location="cpu")
        model.head.load_state_dict(state["head_state_dict"])
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_step = int(state.get("global_step", 0))
        best_val_acc = float(state.get("best_val_acc", -1.0))
        print(f"[resume] loaded {args.resume_checkpoint} global_step={start_step} best_val_acc={best_val_acc:.4f}")

    resolved_config = vars(args).copy()
    for key, value in list(resolved_config.items()):
        if isinstance(value, Path):
            resolved_config[key] = str(value)
    write_json(args.output_dir / "config_resolved.json", resolved_config)

    if args.debug_batch:
        val_metrics = evaluate(model, val_loader, device, args.freeze_backbone, args.bf16, args.fp16, max_batches=1)
        print(f"[debug] one eval batch: {val_metrics}")
        return

    global_step = start_step
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_acc = 0.0
    running_margin = 0.0
    running_count = 0
    start_time = time.time()

    max_train_batches = int(math.ceil(len(train_loader) * args.num_epochs))
    completed_batches = 0
    progress_bar = tqdm(
        total=total_update_steps,
        initial=global_step,
        desc=f"reranker {args.stage}",
        dynamic_ncols=True,
        disable=not args.progress,
    )
    for epoch_index in range(int(math.ceil(args.num_epochs))):
        if epoch_index >= args.num_epochs:
            break
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        for batch_index, batch in enumerate(train_loader):
            fractional_epoch = epoch_index + batch_index / max(1, len(train_loader))
            if fractional_epoch >= args.num_epochs:
                break
            completed_batches += 1
            image_embeds, text_pos, text_neg, loss_weight = encode_batch(
                model, batch, device, args.freeze_backbone, args.bf16, args.fp16
            )
            with autocast_context(device, bf16=args.bf16, fp16=args.fp16):
                metrics = compute_loss(model, image_embeds, text_pos, text_neg, loss_weight)
                loss = metrics["loss"] / max(1, args.gradient_accumulation_steps)
            if not torch.isfinite(loss):
                print(f"[error] non-finite loss at step={global_step} batch={batch_index}: {loss}")
                print_batch_debug(batch)
                raise SystemExit(1)
            loss.backward()

            batch_count = int(batch["loss_weight"].shape[0])
            running_loss += float(metrics["loss"].detach().cpu()) * batch_count
            running_acc += float(metrics["accuracy"].detach().cpu()) * batch_count
            running_margin += float(metrics["margin"].detach().cpu()) * batch_count
            running_count += batch_count

            if completed_batches % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                else:
                    grad_norm = torch.tensor(0.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress_bar.update(1)

                window_loss = running_loss / max(1, running_count)
                window_acc = running_acc / max(1, running_count)
                window_margin = running_margin / max(1, running_count)
                lr = scheduler.get_last_lr()[0]
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                progress_bar.set_postfix(
                    loss=f"{window_loss:.4f}",
                    acc=f"{window_acc:.3f}",
                    margin=f"{window_margin:.3f}",
                    lr=f"{lr:.1e}",
                    mem=f"{gpu_mem:.1f}G",
                )

                if global_step % args.logging_steps == 0:
                    elapsed = max(1e-6, time.time() - start_time)
                    lr = scheduler.get_last_lr()[0]
                    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                    tqdm.write(str(
                        {
                            "step": global_step,
                            "epoch": f"{fractional_epoch:.3f}",
                            "loss": f"{running_loss / max(1, running_count):.4f}",
                            "pair_acc": f"{running_acc / max(1, running_count):.4f}",
                            "margin": f"{running_margin / max(1, running_count):.4f}",
                            "lr": f"{lr:.3e}",
                            "grad_norm": f"{float(grad_norm):.4f}",
                            "pairs_per_sec": f"{running_count / elapsed:.1f}",
                            "gpu_mem_gb": f"{gpu_mem:.2f}",
                        }
                    ))
                    running_loss = running_acc = running_margin = 0.0
                    running_count = 0
                    start_time = time.time()

                if global_step % args.eval_steps == 0:
                    val_metrics = evaluate(
                        model, val_loader, device, args.freeze_backbone, args.bf16, args.fp16, max_batches=args.eval_max_batches
                    )
                    tqdm.write(str({f"eval/{key}": f"{value:.4f}" for key, value in val_metrics.items()} | {"step": global_step}))
                    if val_metrics["accuracy"] > best_val_acc:
                        best_val_acc = val_metrics["accuracy"]
                        save_checkpoint(
                            args.output_dir,
                            "checkpoint_best.pt",
                            {
                                "global_step": global_step,
                                "best_val_acc": best_val_acc,
                                "head_state_dict": model.head.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "config": resolved_config,
                            },
                            backbone=model.backbone,
                            processor=processor,
                            save_backbone=args.save_backbone,
                        )
                        tqdm.write(f"[checkpoint] saved best accuracy={best_val_acc:.4f}")

                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        args.output_dir,
                        f"checkpoint_step_{global_step}.pt",
                        {
                            "global_step": global_step,
                            "best_val_acc": best_val_acc,
                            "head_state_dict": model.head.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "config": resolved_config,
                        },
                        backbone=model.backbone,
                        processor=processor,
                        save_backbone=args.save_backbone,
                    )
                    tqdm.write(f"[checkpoint] saved step={global_step}")

    progress_bar.close()
    val_metrics = evaluate(model, val_loader, device, args.freeze_backbone, args.bf16, args.fp16, max_batches=args.eval_max_batches)
    save_checkpoint(
        args.output_dir,
        "checkpoint_final.pt",
        {
            "global_step": global_step,
            "best_val_acc": max(best_val_acc, val_metrics["accuracy"]),
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": resolved_config,
            "final_eval": val_metrics,
        },
        backbone=model.backbone,
        processor=processor,
        save_backbone=args.save_backbone,
    )
    print({f"final_eval/{key}": f"{value:.4f}" for key, value in val_metrics.items()} | {"step": global_step})
    print(f"[done] saved final checkpoint to {args.output_dir / 'checkpoint_final.pt'}")


if __name__ == "__main__":
    main()
