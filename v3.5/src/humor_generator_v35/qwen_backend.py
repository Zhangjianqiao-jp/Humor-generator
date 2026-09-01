"""Qwen2.5-VL backend used only by explicitly labelled v3 engineering runs."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
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
    teacher_forced_token_states,
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


def find_multimodal_core(model: Any) -> Any:
    """Locate the Qwen-VL core that owns placeholder and MRoPE helpers."""
    queue, seen = [model], set()
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if all(hasattr(current, name) for name in ("get_placeholder_mask", "get_rope_index")):
            return current
        for name in ("base_model", "model", "language_model"):
            child = getattr(current, name, None)
            if child is not None:
                queue.append(child)
    raise RuntimeError(f"cannot locate multimodal Qwen core below {type(model).__name__}")


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
    emitted_token_mean_logprob: float
    sampling_mode: str
    communication_state_definition: str


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

    @classmethod
    def load_with_adapters(
        cls,
        model_name: str,
        adapters: dict[str, str | Path],
        *,
        revision: str | None = None,
        load_in_4bit: bool = True,
    ) -> "QwenBackend":
        """Load one immutable base and multiple frozen PEFT adapters.

        This avoids holding duplicate 7B backbones while keeping Planner and
        Generator adapter identities explicit.  Adapter switching is never
        used inside an autograd graph.
        """
        if not adapters:
            raise ValueError("at least one named adapter is required")
        first_name, first_path = next(iter(adapters.items()))
        backend = cls.load(
            model_name, revision=revision, adapter=None, load_in_4bit=load_in_4bit
        )
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("PEFT is unavailable") from exc
        model = PeftModel.from_pretrained(
            backend.model, str(first_path), adapter_name=first_name, is_trainable=False
        )
        for name, path in list(adapters.items())[1:]:
            model.load_adapter(str(path), adapter_name=name, is_trainable=False)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        backend.model = model
        backend.set_adapter(first_name)
        return backend

    def set_adapter(self, name: str) -> None:
        if not hasattr(self.model, "set_adapter"):
            raise RuntimeError("backend has no named PEFT adapters")
        self.model.set_adapter(name)
        self.model.eval()

    @contextmanager
    def base_only(self):
        """Temporarily disable PEFT adapters for the base-receiver condition."""
        disable = getattr(self.model, "disable_adapter", None)
        if disable is None:
            yield
            return
        with disable():
            yield

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

    @torch.no_grad()
    def multimodal_embeddings_and_positions(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize visual embeddings and exact pre-insertion 3D positions.

        Passing raw token embeddings as ``inputs_embeds`` bypasses neither the
        vision tower nor MRoPE safely.  This helper performs the same visual
        placeholder replacement as Qwen2.5-VL, then computes position ids from
        the untouched discrete sequence before any continuous slots are added.
        """
        if "input_ids" not in inputs or "attention_mask" not in inputs:
            raise ValueError("encoded inputs require input_ids and attention_mask")
        input_ids = inputs["input_ids"]
        embeddings = self.model.get_input_embeddings()(input_ids)
        core = find_multimodal_core(self.model)

        image_features = None
        if inputs.get("pixel_values") is not None:
            image_output = self.model.get_image_features(
                inputs["pixel_values"], inputs.get("image_grid_thw")
            )
            image_features = torch.cat(image_output.pooler_output, dim=0).to(
                device=embeddings.device, dtype=embeddings.dtype
            )
        video_features = None
        if inputs.get("pixel_values_videos") is not None:
            video_output = self.model.get_video_features(
                inputs["pixel_values_videos"], inputs.get("video_grid_thw")
            )
            video_features = torch.cat(video_output.pooler_output, dim=0).to(
                device=embeddings.device, dtype=embeddings.dtype
            )
        image_mask, video_mask = core.get_placeholder_mask(
            input_ids,
            inputs_embeds=embeddings,
            image_features=image_features,
            video_features=video_features,
        )
        if image_features is not None:
            embeddings = embeddings.masked_scatter(image_mask, image_features)
        if video_features is not None:
            embeddings = embeddings.masked_scatter(video_mask, video_features)

        mm_types = inputs.get("mm_token_type_ids")
        if (image_features is not None or video_features is not None) and mm_types is None:
            raise RuntimeError("multimodal input is missing mm_token_type_ids required for MRoPE")
        if mm_types is None:
            base = torch.arange(input_ids.shape[1], device=input_ids.device)
            positions = base.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        else:
            positions, _ = core.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_types,
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                second_per_grid_ts=inputs.get("second_per_grid_ts"),
                attention_mask=inputs["attention_mask"],
            )
        if positions.shape[-1] != embeddings.shape[1]:
            raise RuntimeError("Qwen MRoPE positions do not align with multimodal embeddings")
        return embeddings.detach(), positions.detach()

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
            # The pinned Qwen checkpoint ships repetition_penalty=1.05.  Set
            # the neutral value explicitly so text and latent conditions do
            # not inherit an undisclosed checkpoint-side sampling choice.
            "repetition_penalty": 1.0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 1.0
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def generate_with_latent_prefix(
        self,
        messages: list[dict[str, Any]],
        slots: torch.Tensor,
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> str:
        """Append continuous communication slots after the receiver prompt."""
        if slots.ndim != 3 or slots.shape[0] != 1 or slots.shape[1] < 1:
            raise ValueError("slots must be non-empty [1,S,D]")
        inputs = self.encode(messages, add_generation_prompt=True)
        embeddings, position_ids = self.multimodal_embeddings_and_positions(inputs)
        slots = slots.to(device=embeddings.device, dtype=embeddings.dtype)
        combined = torch.cat([embeddings, slots], dim=1)
        mask = torch.cat([
            inputs["attention_mask"],
            torch.ones(
                (1, slots.shape[1]),
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device,
            ),
        ], dim=1)
        start = position_ids[:, :, -1:] + 1
        offsets = torch.arange(slots.shape[1], device=position_ids.device).view(1, 1, -1)
        combined_positions = torch.cat([position_ids, start + offsets], dim=-1)
        torch.manual_seed(seed)
        kwargs: dict[str, Any] = {
            "inputs_embeds": combined,
            "attention_mask": mask,
            "position_ids": combined_positions,
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.0,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = 1.0
        with torch.inference_mode():
            generated = self.model.generate(**kwargs)
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def generate_and_verify_states(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        seed: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple[str, AlignedMessageStates, GenerationAlignmentReport]:
        """Fail rather than trim when generation hooks and causal replay disagree."""
        inputs = self.encode(messages, add_generation_prompt=True)
        prompt_length = inputs["input_ids"].shape[1]
        layer = find_last_decoder_layer(self.model)
        decode_capture = DecodeStateCapture(output_device="cpu")
        handle = layer.register_forward_hook(decode_capture)
        try:
            torch.manual_seed(seed)
            generation_kwargs: dict[str, Any] = {
                "do_sample": temperature > 0,
                "max_new_tokens": max_new_tokens,
                "return_dict_in_generate": True,
                "output_scores": True,
                "repetition_penalty": 1.0,
            }
            if temperature > 0:
                generation_kwargs.update({"temperature": temperature, "top_p": top_p})
            with torch.inference_mode():
                generated_output = self.model.generate(
                    **inputs,
                    **generation_kwargs,
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
        if temperature <= 0 and processed_accuracy != 1.0:
            raise AssertionError(
                f"generation scores predict only {processed_accuracy:.3%} of emitted greedy tokens"
            )
        emitted_logprobs = torch.stack([
            score.log_softmax(dim=-1).gather(
                1, token_ids[:, index : index + 1].to(score.device)
            ).squeeze(1)
            for index, score in enumerate(generated_output.scores)
        ], dim=1)

        text = self.processor.batch_decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        # Re-run the processor on the complete conversation.  For multimodal
        # Qwen inputs, manually appending token IDs leaves vision position
        # metadata inconsistent with the new sequence length.
        replay_inputs = self.encode(
            [*messages, {"role": "assistant", "content": [{"type": "text", "text": text}]}],
            add_generation_prompt=False,
        )
        replay_ids = replay_inputs["input_ids"]
        replay_targets = replay_ids[:, prompt_length : prompt_length + token_ids.shape[1]]
        if replay_targets.shape != token_ids.shape or not torch.equal(
            replay_targets.detach().cpu(), token_ids.detach().cpu()
        ):
            raise AssertionError(
                "decoded assistant text does not replay to the exact emitted token sequence"
            )
        sequence_capture = SequenceStateCapture()
        replay_handle = layer.register_forward_hook(sequence_capture)
        try:
            with torch.inference_mode():
                self.model(
                    **replay_inputs,
                    use_cache=False,
                    output_hidden_states=False,
                    logits_to_keep=1,
                )
        finally:
            replay_handle.remove()
        replay_hidden = sequence_capture.require()
        teacher_states = teacher_forced_prediction_states(
            replay_hidden,
            prompt_length=prompt_length,
            generated_token_ids=token_ids,
        )
        communication_states = teacher_forced_token_states(
            replay_hidden,
            prompt_length=prompt_length,
            generated_token_ids=token_ids,
        )
        hook_accuracy = token_prediction_accuracy(self.model, hook_states)
        replay = assert_causal_replay_alignment(hook_states, teacher_states)
        return text, communication_states, GenerationAlignmentReport(
            replay=replay,
            processed_score_token_accuracy=processed_accuracy,
            raw_head_token_accuracy=hook_accuracy,
            emitted_token_mean_logprob=float(emitted_logprobs.mean().detach().cpu()),
            sampling_mode="sample" if temperature > 0 else "greedy",
            communication_state_definition="teacher_forced_post_token",
        )

    @torch.no_grad()
    def replay_assistant_states(
        self,
        messages: list[dict[str, Any]],
        text: str,
    ) -> tuple[AlignedMessageStates, dict[str, Any]]:
        """Capture post-token states for fixed text under the original prompt.

        This is used only after validator-feedback serialization repair.  The
        repaired answer is generated in a separate recovery turn, then replayed
        under the unchanged HOMER prompt so its communication states remain
        conditioned on the same input as ordinary traces.  It is intentionally
        reported as teacher-forced replay, not generation/replay alignment.
        """
        prompt_inputs = self.encode(messages, add_generation_prompt=True)
        prompt_length = prompt_inputs["input_ids"].shape[1]
        replay_inputs = self.encode(
            [*messages, {"role": "assistant", "content": [{"type": "text", "text": text}]}],
            add_generation_prompt=False,
        )
        token_ids = replay_inputs["input_ids"][:, prompt_length:]
        if token_ids.shape[1] < 1:
            raise AssertionError("repaired assistant answer replayed to zero tokens")
        decoded = self.processor.batch_decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        if decoded != text.strip():
            raise AssertionError("repaired assistant text does not round-trip exactly")
        layer = find_last_decoder_layer(self.model)
        capture = SequenceStateCapture()
        handle = layer.register_forward_hook(capture)
        try:
            self.model(
                **replay_inputs,
                use_cache=False,
                output_hidden_states=False,
                logits_to_keep=1,
            )
        finally:
            handle.remove()
        states = teacher_forced_token_states(
            capture.require(),
            prompt_length=prompt_length,
            generated_token_ids=token_ids,
        )
        return states, {
            "state_capture_mode": "teacher_forced_replay_after_validator_repair",
            "exact_text_roundtrip": True,
            "assistant_token_count": int(token_ids.shape[1]),
            "communication_state_definition": "teacher_forced_post_token",
        }
