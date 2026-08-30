"""Qwen2.5-VL backend used only by explicitly labelled v3 engineering runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .latent.state_capture import (
    AlignedMessageStates,
    DecodeStateCapture,
    ReplayAlignmentReport,
    SequenceStateCapture,
    assert_causal_replay_alignment,
    teacher_forced_prediction_states,
)


def model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")


def find_last_decoder_layer(model: Any) -> Any:
    queue, seen = [model], set()
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        layers = getattr(current, "layers", None)
        if layers is not None and len(layers):
            return layers[-1]
        for name in ("base_model", "model", "language_model"):
            child = getattr(current, name, None)
            if child is not None:
                queue.append(child)
    raise RuntimeError(f"cannot locate decoder layers below {type(model).__name__}")


def find_final_decoder_norm(model: Any) -> Any:
    queue, seen = [model], set()
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        layers = getattr(current, "layers", None)
        norm = getattr(current, "norm", None)
        if layers is not None and len(layers) and norm is not None:
            return norm
        for name in ("base_model", "model", "language_model"):
            child = getattr(current, name, None)
            if child is not None:
                queue.append(child)
    raise RuntimeError(f"cannot locate final decoder norm below {type(model).__name__}")


@torch.no_grad()
def token_prediction_accuracy(model: Any, states: AlignedMessageStates) -> float:
    device = model_device(model)
    norm = find_final_decoder_norm(model)
    output = model.get_output_embeddings()
    hidden = states.states.to(device=device, dtype=next(norm.parameters()).dtype)
    predicted = output(norm(hidden)).argmax(dim=-1).detach().cpu()
    return float(predicted.eq(states.token_ids).float().mean())


@dataclass(frozen=True)
class GenerationAlignmentReport:
    replay: ReplayAlignmentReport
    processed_score_token_accuracy: float
    raw_head_token_accuracy: float


class QwenBackend:
    def __init__(self, model: Any, processor: Any, process_vision_info: Any) -> None:
        self.model = model
        self.processor = processor
        self.process_vision_info = process_vision_info

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        revision: str | None = None,
        adapter: str | Path | None = None,
        load_in_4bit: bool = True,
    ) -> "QwenBackend":
        try:
            from peft import PeftModel
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError("Qwen/PEFT/qwen-vl-utils dependencies are unavailable") from exc
        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if revision:
            kwargs["revision"] = revision
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        if adapter:
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        return cls(model, processor, process_vision_info)

    def encode(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> dict[str, torch.Tensor]:
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        kwargs: dict[str, Any] = {"text": [rendered], "padding": True, "return_tensors": "pt"}
        if image_inputs is not None:
            kwargs["images"] = image_inputs
        if video_inputs is not None:
            kwargs["videos"] = video_inputs
        return self.processor(**kwargs).to(model_device(self.model))

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> str:
        inputs = self.encode(messages, add_generation_prompt=True)
        torch.manual_seed(seed)
        generation_kwargs: dict[str, Any] = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.05,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def generate_and_verify_states(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        seed: int,
    ) -> tuple[str, AlignedMessageStates, GenerationAlignmentReport]:
        """Fail rather than trim when generation hooks and causal replay disagree."""
        inputs = self.encode(messages, add_generation_prompt=True)
        prompt_length = inputs["input_ids"].shape[1]
        layer = find_last_decoder_layer(self.model)
        decode_capture = DecodeStateCapture(output_device="cpu")
        handle = layer.register_forward_hook(decode_capture)
        try:
            torch.manual_seed(seed)
            with torch.inference_mode():
                generated_output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        finally:
            handle.remove()
        generated = generated_output.sequences
        token_ids = generated[:, prompt_length:]
        hook_states = decode_capture.align(token_ids)
        if len(generated_output.scores) != token_ids.shape[1]:
            raise AssertionError(
                "generation score/token mismatch: "
                f"steps={len(generated_output.scores)} tokens={token_ids.shape[1]}"
            )
        processed_predictions = torch.stack(
            [score.argmax(dim=-1) for score in generated_output.scores], dim=1
        ).detach().cpu()
        processed_accuracy = float(processed_predictions.eq(token_ids.cpu()).float().mean())
        if processed_accuracy != 1.0:
            raise AssertionError(
                f"generation scores predict only {processed_accuracy:.3%} of emitted greedy tokens"
            )

        replay_ids = torch.cat([inputs["input_ids"], token_ids.to(inputs["input_ids"].device)], dim=1)
        replay_mask = torch.ones_like(replay_ids)
        sequence_capture = SequenceStateCapture()
        replay_handle = layer.register_forward_hook(sequence_capture)
        try:
            with torch.inference_mode():
                self.model(
                    input_ids=replay_ids,
                    attention_mask=replay_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    logits_to_keep=1,
                )
        finally:
            replay_handle.remove()
        teacher_states = teacher_forced_prediction_states(
            sequence_capture.require(),
            prompt_length=prompt_length,
            generated_token_ids=token_ids,
        )
        hook_accuracy = token_prediction_accuracy(self.model, hook_states)
        replay = assert_causal_replay_alignment(hook_states, teacher_states)
        text = self.processor.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return text, hook_states, GenerationAlignmentReport(
            replay=replay,
            processed_score_token_accuracy=processed_accuracy,
            raw_head_token_accuracy=hook_accuracy,
        )
