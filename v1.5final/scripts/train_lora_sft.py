#!/usr/bin/env python
from __future__ import annotations

import math
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.qwen_vl_lora_loader import load_qwen_vl_with_lora
from src.training.sft_dataset import DEFAULT_SFT_PROMPT, HumorSFTDataset, clean_generated_caption
from src.utils.io import write_jsonl

NON_MODEL_INPUT_KEYS = {
    "metadata",
    "supervised_token_counts",
    "prompt_lengths",
    "truncated_flags",
}


class HumorSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        metadata = inputs.get("metadata")
        supervised_counts = inputs.get("supervised_token_counts")
        if supervised_counts is not None:
            supervised_tokens = int(supervised_counts.sum().detach().cpu().item())
            self._last_supervised_tokens = supervised_tokens
            if supervised_tokens <= 0:
                raise RuntimeError(f"Batch has no supervised tokens. metadata={metadata}")

        model_inputs = {k: v for k, v in inputs.items() if k not in NON_MODEL_INPUT_KEYS}
        outputs = model(**model_inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if not torch.isfinite(loss):
            print(f"[error] Non-finite loss at global_step={self.state.global_step}: {loss}")
            if metadata is not None:
                print(f"[error] batch metadata: {metadata}")
            raise FloatingPointError(f"Non-finite training loss at step {self.state.global_step}: {loss}")
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        logs = dict(logs)
        if hasattr(self, "_last_supervised_tokens"):
            logs["supervised_tokens"] = float(self._last_supervised_tokens)
        if torch.cuda.is_available():
            logs["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        return super().log(logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and math.isfinite(eval_loss):
            metrics["eval_ppl"] = math.exp(min(eval_loss, 20.0))
            self.log({"eval_ppl": metrics["eval_ppl"]})
        return metrics


class CadenceGuardCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0:
            return control
        eval_steps = int(args.eval_steps or 0)
        save_steps = int(args.save_steps or 0)
        if eval_steps > 0 and state.global_step % eval_steps != 0:
            control.should_evaluate = False
        if save_steps > 0 and state.global_step % save_steps != 0:
            control.should_save = False
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        state.eval_steps = args.eval_steps
        state.save_steps = args.save_steps
        state.logging_steps = args.logging_steps
        print(
            "[cadence] using current config cadence: "
            f"logging_steps={args.logging_steps}, eval_steps={args.eval_steps}, save_steps={args.save_steps}"
        )
        return control


class AdapterCheckpointCallback(TrainerCallback):
    def __init__(self, processor: Any, output_dir: Path, latest_dir: Path, best_dir: Path) -> None:
        self.processor = processor
        self.output_dir = output_dir
        self.latest_dir = latest_dir
        self.best_dir = best_dir
        self.best_eval_loss = float("inf")

    def _save_adapter(self, model: Any, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        self.processor.save_pretrained(path)
        print(f"[checkpoint] saved LoRA adapter: {path}")

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._save_adapter(model, self.latest_dir)
        return control

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if model is None or not metrics:
            return control
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and math.isfinite(eval_loss) and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self._save_adapter(model, self.best_dir)
            print(f"[checkpoint] new best_val_loss={eval_loss:.6f} at step={state.global_step}")
        return control


class FixedGenerationCallback(TrainerCallback):
    def __init__(
        self,
        val_dataset: HumorSFTDataset,
        processor: Any,
        output_dir: Path,
        generation_config: dict[str, Any],
        num_samples: int = 5,
    ) -> None:
        self.val_dataset = val_dataset
        self.processor = processor
        self.generation_dir = output_dir / "eval_generations"
        self.generation_config = generation_config
        self.num_samples = min(num_samples, len(val_dataset))
        self.generated_steps: set[int] = set()

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step in self.generated_steps:
            return control
        self.generated_steps.add(state.global_step)
        self._generate(model, state.global_step)
        return control

    def _generate(self, model: Any, step: int) -> None:
        was_training = model.training
        model.eval()
        rows = []
        for row in self.val_dataset.rows[: self.num_samples]:
            generated = generate_caption(
                model=model,
                processor=self.processor,
                messages=self.val_dataset.build_prompt_messages(row),
                prompt=self.val_dataset.prompt_for_row(row),
                generation_config=self.generation_config,
            )[0]
            rows.append(
                {
                    "step": step,
                    "image": row["image"],
                    "gold_caption": row["caption"],
                    "generated_caption": generated,
                    "prompt": self.val_dataset.prompt_for_row(row),
                    "score": row.get("meta", {}).get("score"),
                }
            )
        output_path = self.generation_dir / f"step_{step:04d}.jsonl"
        write_jsonl(output_path, rows)
        print(f"[generation] saved fixed validation generations: {output_path}")
        if was_training:
            model.train()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("data", {})
    config["data"].setdefault("normalize_prompt", True)
    config["data"].setdefault("sft_prompt", DEFAULT_SFT_PROMPT)
    config["data"].setdefault("image_root", None)
    config["data"].setdefault("max_caption_chars", 240)
    config["data"].setdefault("min_supervised_tokens", 3)
    config.setdefault("evaluation", {})
    config["evaluation"].setdefault("fixed_generation_samples", 5)
    config["evaluation"].setdefault("max_eval_samples", None)
    config.setdefault("generation", {})
    config["generation"].setdefault("max_new_tokens", 48)
    config["generation"].setdefault("temperature", 0.8)
    config["generation"].setdefault("top_p", 0.9)
    config["generation"].setdefault("do_sample", True)
    config["generation"].setdefault("num_candidates", 10)
    config["generation"].setdefault("repetition_penalty", 1.05)
    return config


def apply_cli_overrides(config: dict[str, Any], unknown_args: list[str]) -> None:
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if not arg.startswith("--override-"):
            raise ValueError(f"Unknown argument: {arg}")
        if i + 1 >= len(unknown_args):
            raise ValueError(f"Missing value for {arg}")
        dotted_key = arg[len("--override-") :]
        value = yaml.safe_load(unknown_args[i + 1])
        target = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
        print(f"[config] override {dotted_key}={value!r}")
        i += 2


def resolve_optional_path(value: str | None) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(value)


def build_dataset(
    config: dict[str, Any],
    split: str,
    processor: Any | None,
    max_samples: int | None = None,
    validate_images: bool = True,
) -> HumorSFTDataset:
    data = config["data"]
    output_dir = Path(config["output"]["output_dir"])
    report_dir = output_dir / "missing_images"
    report_dir.mkdir(parents=True, exist_ok=True)
    path_key = f"{split}_path"
    report_path = report_dir / f"{split}_missing.jsonl"
    return HumorSFTDataset(
        path=Path(data[path_key]),
        processor=processor,
        max_seq_len=int(data.get("max_seq_len", 512)),
        image_root=resolve_optional_path(data.get("image_root")),
        max_caption_chars=int(data.get("max_caption_chars", 240)),
        skip_missing_images=bool(data.get("skip_missing_images", False)),
        normalize_prompt=bool(data.get("normalize_prompt", True)),
        sft_prompt=str(data.get("sft_prompt", DEFAULT_SFT_PROMPT)),
        min_supervised_tokens=int(data.get("min_supervised_tokens", 3)),
        missing_image_report_path=report_path,
        max_samples=max_samples,
        validate_images=validate_images,
    )


def get_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if key in NON_MODEL_INPUT_KEYS:
            continue
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def generate_caption(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    prompt: str,
    generation_config: dict[str, Any],
) -> list[str]:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(get_model_device(model))
    num_return_sequences = int(generation_config.get("num_return_sequences", 1))
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=bool(generation_config.get("do_sample", True)),
            temperature=float(generation_config.get("temperature", 0.8)),
            top_p=float(generation_config.get("top_p", 0.9)),
            max_new_tokens=int(generation_config.get("max_new_tokens", 48)),
            repetition_penalty=float(generation_config.get("repetition_penalty", 1.05)),
            num_return_sequences=num_return_sequences,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [clean_generated_caption(text, prompt=prompt) for text in decoded]


def save_run_config(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def tensorboard_report_to() -> list[str]:
    try:
        import tensorboard  # noqa: F401

        return ["tensorboard"]
    except Exception:
        print("[logging] tensorboard is not installed; using stdout logs only.")
        return []



def disable_use_cache(model: Any) -> None:
    seen = set()
    stack = [model]
    if hasattr(model, "modules"):
        stack.extend(list(model.modules()))
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        generation_config = getattr(candidate, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            generation_config.use_cache = False
        stack.extend([
            getattr(candidate, "base_model", None),
            getattr(candidate, "model", None),
            getattr(candidate, "language_model", None),
        ])


def prepare_model_and_processor(config: dict[str, Any]) -> tuple[Any, Any]:
    model, processor = load_qwen_vl_with_lora(
        model_name=config["model"]["model_name"],
        lora_rank=config["model"]["lora"]["rank"],
        lora_alpha=config["model"]["lora"]["alpha"],
        lora_dropout=config["model"]["lora"]["dropout"],
        target_modules=config["model"]["lora"]["target_modules"],
        bias=config["model"]["lora"].get("bias", "none"),
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "auto"),
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    if config["training"].get("gradient_checkpointing", False):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        disable_use_cache(model)
        print("[model] gradient checkpointing enabled; use_cache disabled on PEFT/base configs")
    return model, processor


def run_debug_data(config: dict[str, Any], num_debug_samples: int, max_train_samples: int | None, max_val_samples: int | None) -> None:
    train_dataset = build_dataset(config, "train", processor=None, max_samples=max_train_samples)
    val_dataset = build_dataset(config, "val", processor=None, max_samples=max_val_samples)
    print(f"[debug-data] train={len(train_dataset)}/{train_dataset.original_count}")
    print(f"[debug-data] val={len(val_dataset)}/{val_dataset.original_count}")
    print("[debug-data] first train samples")
    train_dataset.print_debug_samples(num_debug_samples)
    print("[debug-data] first validation samples")
    val_dataset.print_debug_samples(num_debug_samples)


def run_debug_collator(config: dict[str, Any], num_debug_samples: int) -> None:
    set_seed(config["training"].get("seed", 42))
    model, processor = prepare_model_and_processor(config)
    dataset = build_dataset(config, "train", processor=processor, max_samples=num_debug_samples)
    examples = [dataset[i] for i in range(min(num_debug_samples, len(dataset)))]
    batch = dataset.collate_fn(examples)
    dataset.print_debug_batch(batch, examples, n=num_debug_samples)
    model.eval()
    model_inputs = move_batch_to_device(batch, get_model_device(model))
    with torch.no_grad():
        outputs = model(**model_inputs)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    print(f"[debug-collator] one forward loss={float(loss.detach().cpu()):.6f}")
    print(f"[debug-collator] supervised tokens per sample={batch['supervised_token_counts'].tolist()}")


def training_args_from_config(config: dict[str, Any], debug_one_step: bool = False) -> TrainingArguments:
    training = config["training"]
    output = config["output"]
    kwargs = {
        "output_dir": output["output_dir"],
        "per_device_train_batch_size": training["batch_size"],
        "per_device_eval_batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "num_train_epochs": training["num_epochs"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "max_grad_norm": training["max_grad_norm"],
        "logging_steps": training["logging_steps"],
        "eval_steps": training["eval_steps"],
        "save_steps": training["save_steps"],
        "save_total_limit": training.get("save_total_limit", 3),
        "eval_strategy": "no" if debug_one_step else "steps",
        "save_strategy": "no" if debug_one_step else "steps",
        "bf16": bool(training.get("bf16", False)) and torch.cuda.is_available(),
        "fp16": bool(training.get("fp16", False)) and torch.cuda.is_available(),
        "optim": training.get("optim", "adamw_torch"),
        "remove_unused_columns": False,
        "report_to": tensorboard_report_to(),
        "logging_dir": output.get("tensorboard_dir", str(Path(output["output_dir"]) / "tensorboard")),
        "skip_memory_metrics": False,
    }
    if debug_one_step:
        kwargs["max_steps"] = 1
        kwargs["gradient_accumulation_steps"] = 1
        kwargs["logging_steps"] = 1
    if "warmup_steps" in training:
        kwargs["warmup_steps"] = training["warmup_steps"]
    else:
        kwargs["warmup_ratio"] = training.get("warmup_ratio", 0.03)
    return TrainingArguments(**kwargs)


def train(
    config_path: Path,
    resume_from_checkpoint: str | None = None,
    debug_one_step: bool = False,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> None:
    config = load_config(config_path)
    set_seed(config["training"].get("seed", 42))
    if resume_from_checkpoint:
        print(
            "[warning] Resuming from a checkpoint was explicitly requested. "
            "For clean-prompt repair, start from the base model unless you truly want to continue a run."
        )

    output_dir = Path(config["output"]["output_dir"])
    save_run_config(config, config_path, output_dir)
    model, processor = prepare_model_and_processor(config)

    if debug_one_step:
        max_train_samples = max_train_samples or max(1, int(config["training"].get("batch_size", 1)))
        max_val_samples = max_val_samples or 1
    elif max_val_samples is None:
        max_val_samples = config.get("evaluation", {}).get("max_eval_samples")

    train_dataset = build_dataset(config, "train", processor=processor, max_samples=max_train_samples)
    val_dataset = build_dataset(config, "val", processor=processor, max_samples=max_val_samples)
    print(
        "[data] dataset sizes after validation: "
        f"train={len(train_dataset)}/{train_dataset.original_count}, "
        f"val={len(val_dataset)}/{val_dataset.original_count}"
    )
    train_dataset.print_debug_samples(3)

    args = training_args_from_config(config, debug_one_step=debug_one_step)
    callbacks = [CadenceGuardCallback()]
    if not debug_one_step:
        callbacks.extend(
            [
                AdapterCheckpointCallback(
                    processor=processor,
                    output_dir=output_dir,
                    latest_dir=Path(config["output"].get("latest_adapter_dir", output_dir / "latest")),
                    best_dir=Path(config["output"].get("best_adapter_dir", output_dir / "best_val_loss")),
                ),
                FixedGenerationCallback(
                    val_dataset=val_dataset,
                    processor=processor,
                    output_dir=output_dir,
                    generation_config=config.get("generation", {}),
                    num_samples=int(config.get("evaluation", {}).get("fixed_generation_samples", 5)),
                ),
            ]
        )

    trainer = HumorSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if debug_one_step:
        print("[debug-one-step] completed one optimizer step; no adapter checkpoint saved.")
        return

    final_metrics = trainer.evaluate()
    print(f"[eval] final metrics: {final_metrics}")
    final_dir = Path(config["output"].get("final_adapter_dir", output_dir / "final_lora"))
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[checkpoint] saved final LoRA adapter: {final_dir}")
    print(f"[data] collator truncated sample encounters: train={train_dataset.truncated_sample_count}, val={val_dataset.truncated_sample_count}")


def main() -> None:
    parser = ArgumentParser(description="Train or debug V1.5 LoRA-SFT humor generator.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft.yaml"))
    parser.add_argument("--resume_from_checkpoint", "--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--debug-data", action="store_true")
    parser.add_argument("--debug-collator", action="store_true")
    parser.add_argument("--debug-one-step", action="store_true")
    parser.add_argument("--num-debug-samples", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-epochs", type=float, default=None)
    args, unknown = parser.parse_known_args()

    config = load_config(args.config)
    apply_cli_overrides(config, unknown)
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
        print(f"[config] override training.num_epochs={args.num_epochs}")

    if args.debug_data:
        run_debug_data(config, args.num_debug_samples, args.max_train_samples, args.max_val_samples)
        return
    if args.debug_collator:
        run_debug_collator(config, args.num_debug_samples)
        return

    tmp_config_path = args.config
    if config != load_config(args.config):
        tmp_config_path = Path(config["output"]["output_dir"]) / "config_cli_resolved.yaml"
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    train(
        tmp_config_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        debug_one_step=args.debug_one_step,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()
