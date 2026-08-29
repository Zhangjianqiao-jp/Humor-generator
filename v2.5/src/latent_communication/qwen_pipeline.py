"""Qwen2.5-VL online Planner -> latent/text -> Generator plumbing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from src.latent_communication.bridge import LearnedLatentBridge, TypedHomerLatentBridge, inject_latent_slots, insert_constant_slots
from src.latent_communication.homer import (
    CONFLICT_SYSTEM_PROMPT,
    DESCRIPTION_PROMPT,
    IMAGINATION_GLOBAL_SYSTEM_PROMPT,
    IMAGINATION_LOCAL_SYSTEM_PROMPT,
    HomerPlan,
    conflict_user_prompt,
    global_imagination_user_prompt,
    latent_generator_context,
    local_imagination_user_prompt,
    text_generator_context,
)
from src.latent_communication.state_capture import GeneratedTokenStateCapture, find_last_decoder_layer
from src.training.sft_dataset import clean_generated_caption

CommunicationMode = Literal["text", "latent", "hybrid"]


@dataclass
class PlannerTrace:
    text: str
    token_ids: torch.Tensor
    hidden_states: torch.Tensor


@dataclass
class HomerPlannerTrace:
    plan: HomerPlan
    grounding: PlannerTrace
    conflict: PlannerTrace
    imagination_local: PlannerTrace
    imagination_global: PlannerTrace

    @property
    def imagination_hidden_states(self) -> torch.Tensor:
        return torch.cat(
            [self.imagination_local.hidden_states, self.imagination_global.hidden_states], dim=1
        )


def model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")


def build_image_message(
    image: str | Path, prompt: str, *, max_pixels: int | None = None
) -> list[dict[str, Any]]:
    image_part: dict[str, Any] = {"type": "image", "image": str(image)}
    if max_pixels is not None:
        image_part["max_pixels"] = int(max_pixels)
    return [
        {
            "role": "user",
            "content": [
                image_part,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_planner_trace(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    *,
    image: str | Path,
    planner_prompt: str,
    max_new_tokens: int = 384,
    max_state_tokens: int = 64,
) -> PlannerTrace:
    """Generate one plan online and capture its final-block decode states."""
    model.set_adapter("planner")
    messages = build_image_message(image, planner_prompt)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model_device(model))
    capture = GeneratedTokenStateCapture(output_device="cpu")
    handle = find_last_decoder_layer(model).register_forward_hook(capture)
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.05,
            )
    finally:
        handle.remove()
    prompt_len = inputs["input_ids"].shape[1]
    token_ids = generated[:, prompt_len:].detach().cpu()
    if token_ids.shape[1] < 1:
        raise RuntimeError(f"Planner generated no tokens for {image}")
    # Hook output includes the prefill call and decode calls.  Retaining the
    # final generated-token count follows StateBridge's memory-efficient hook.
    states = capture.stacked(keep_last=min(token_ids.shape[1], max_state_tokens))
    token_ids = token_ids[:, -states.shape[1] :]
    text = processor.batch_decode(
        generated[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    if not text:
        raise RuntimeError(f"Planner decoded an empty plan for {image}")
    return PlannerTrace(text=text, token_ids=token_ids, hidden_states=states)


def _text_message(prompt: str, *, system: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if system:
        rows.append({"role": "system", "content": [{"type": "text", "text": system}]})
    rows.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    return rows


def _generate_trace_from_messages(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    max_state_tokens: int,
) -> PlannerTrace:
    model.set_adapter("planner")
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    kwargs: dict[str, Any] = {"text": [rendered], "padding": True, "return_tensors": "pt"}
    if image_inputs is not None:
        kwargs["images"] = image_inputs
    if video_inputs is not None:
        kwargs["videos"] = video_inputs
    inputs = processor(**kwargs).to(model_device(model))
    capture = GeneratedTokenStateCapture(output_device="cpu")
    handle = find_last_decoder_layer(model).register_forward_hook(capture)
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens, repetition_penalty=1.05
            )
    finally:
        handle.remove()
    prompt_len = inputs["input_ids"].shape[1]
    ids = generated[:, prompt_len:].detach().cpu()
    if ids.shape[1] == 0:
        raise RuntimeError("Planner stage generated no tokens")
    states = capture.stacked(keep_last=min(ids.shape[1], max_state_tokens))
    text = processor.batch_decode(
        generated[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return PlannerTrace(text=text, token_ids=ids[:, -states.shape[1] :], hidden_states=states)


def generate_homer_plan_trace(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    *,
    image: str | Path,
    max_new_tokens: int = 384,
    max_state_tokens: int = 256,
    image_max_pixels: int | None = 100352,
    culture_context: str | None = None,
) -> HomerPlannerTrace:
    """Run HOMER's dependent description/conflict/local/global stages online."""
    grounding = _generate_trace_from_messages(
        model, processor, process_vision_info,
        build_image_message(image, DESCRIPTION_PROMPT, max_pixels=image_max_pixels),
        max_new_tokens=max_new_tokens, max_state_tokens=max_state_tokens,
    )
    conflict = _generate_trace_from_messages(
        model, processor, process_vision_info,
        _text_message(conflict_user_prompt(grounding.text), system=CONFLICT_SYSTEM_PROMPT),
        max_new_tokens=max_new_tokens, max_state_tokens=max_state_tokens,
    )
    local = _generate_trace_from_messages(
        model, processor, process_vision_info,
        _text_message(
            local_imagination_user_prompt(grounding.text, conflict.text),
            system=IMAGINATION_LOCAL_SYSTEM_PROMPT,
        ), max_new_tokens=max_new_tokens, max_state_tokens=max_state_tokens,
    )
    global_messages = [
        {"role": "system", "content": [{"type": "text", "text": IMAGINATION_GLOBAL_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": str(image), **({"max_pixels": int(image_max_pixels)} if image_max_pixels is not None else {})},
            {"type": "text", "text": global_imagination_user_prompt(conflict.text)},
        ]},
    ]
    global_trace = _generate_trace_from_messages(
        model, processor, process_vision_info, global_messages,
        max_new_tokens=max_new_tokens, max_state_tokens=max_state_tokens,
    )
    plan = HomerPlan(
        grounding=grounding.text,
        conflict=conflict.text,
        associative_imagination_local=local.text,
        associative_imagination_global=global_trace.text,
        culture_context=culture_context,
    )
    return HomerPlannerTrace(plan, grounding, conflict, local, global_trace)


def homer_generator_prompt(mode: CommunicationMode, trace: HomerPlannerTrace, *, include_culture: bool = False) -> str:
    if mode == "text":
        return text_generator_context(trace.plan, include_culture=include_culture)
    if mode in ("latent", "hybrid"):
        base = latent_generator_context(trace.plan, include_culture=include_culture)
        if mode == "hybrid":
            base += f"\n\nConflict scripts:\n{trace.plan.conflict}\n\nFree-association chains:\n{trace.plan.associative_imagination}"
        return base
    raise ValueError(f"Unknown communication mode: {mode}")


def generate_homer_candidates(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    bridge: TypedHomerLatentBridge,
    trace: HomerPlannerTrace,
    *,
    image: str | Path,
    mode: CommunicationMode,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
    include_culture: bool = False,
) -> list[str]:
    model.set_adapter("generator")
    prompt = homer_generator_prompt(mode, trace, include_culture=include_culture)
    messages = build_image_message(image, prompt)
    inputs = _encode_messages(processor, process_vision_info, messages).to(model_device(model))
    original_len = inputs["input_ids"].shape[1]
    if mode in ("latent", "hybrid"):
        dtype = next(bridge.parameters()).dtype
        with torch.no_grad():
            slots = bridge(
                trace.conflict.hidden_states.to(model_device(model), dtype=dtype),
                trace.imagination_hidden_states.to(model_device(model), dtype=dtype),
            )["latent_slots"]
        embeddings = model.get_input_embeddings()(inputs["input_ids"])
        placeholder = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        inserted = inject_latent_slots(
            inputs["input_ids"], embeddings, inputs["attention_mask"], slots.to(embeddings.dtype),
            torch.tensor([original_len], device=inputs["input_ids"].device),
            placeholder_token_id=int(placeholder),
        )
        inputs.update(inserted)
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in inputs and inputs[key].shape[1] == original_len:
                inputs[key] = insert_constant_slots(
                    inputs[key], torch.tensor([original_len], device=inputs[key].device), slots.shape[1], value=0
                )
    kwargs: dict[str, Any] = {
        "do_sample": temperature > 0, "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_candidates, "repetition_penalty": 1.05,
    }
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p)
        if top_k is not None:
            kwargs["top_k"] = top_k
    generated = model.generate(**inputs, **kwargs)
    start = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(generated[:, start:], skip_special_tokens=True)
    return [clean_generated_caption(value, prompt=prompt) for value in decoded]


def generator_prompt(mode: CommunicationMode, plan_text: str) -> str:
    instruction = "Generate one short, natural, image-specific humorous caption. Do not explain."
    if mode in ("text", "hybrid"):
        return f"{instruction}\n\nHumor plan:\n{plan_text.strip()}"
    if mode == "latent":
        return instruction
    raise ValueError(f"Unknown communication mode: {mode}")


def _encode_messages(processor: Any, process_vision_info: Any, messages: list[dict[str, Any]]) -> Any:
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[rendered], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )


def generate_generator_candidates(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    bridge: LearnedLatentBridge,
    trace: PlannerTrace,
    *,
    image: str | Path,
    mode: CommunicationMode,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None = None,
) -> list[str]:
    """Generate Text, Latent, or Hybrid candidates from one online trace."""
    model.set_adapter("generator")
    prompt = generator_prompt(mode, trace.text)
    messages = build_image_message(image, prompt)
    inputs = _encode_messages(processor, process_vision_info, messages).to(model_device(model))
    generation_kwargs: dict[str, Any] = {
        "do_sample": temperature > 0,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_candidates,
        "repetition_penalty": 1.05,
    }
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p)
        if top_k is not None:
            generation_kwargs["top_k"] = top_k

    original_len = inputs["input_ids"].shape[1]
    if mode in ("latent", "hybrid"):
        sender = trace.hidden_states.to(model_device(model), dtype=torch.bfloat16)
        with torch.no_grad():
            # no policy gradients at inference; Bridge remains an ordinary module
            latent = bridge(sender).latent_slots
        embeddings = model.get_input_embeddings()(inputs["input_ids"])
        placeholder = processor.tokenizer.pad_token_id
        if placeholder is None:
            placeholder = processor.tokenizer.eos_token_id
        inserted = inject_latent_slots(
            inputs["input_ids"],
            embeddings,
            inputs["attention_mask"],
            latent.to(embeddings.dtype),
            torch.tensor([original_len], device=inputs["input_ids"].device),
            placeholder_token_id=int(placeholder),
        )
        inputs["input_ids"] = inserted["input_ids"]
        inputs["inputs_embeds"] = inserted["inputs_embeds"]
        inputs["attention_mask"] = inserted["attention_mask"]
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in inputs:
                inputs[key] = insert_constant_slots(
                    inputs[key],
                    torch.tensor([original_len], device=inputs[key].device),
                    latent.shape[1],
                    value=0,
                )
    generated = model.generate(**inputs, **generation_kwargs)
    decode_start = inputs["input_ids"].shape[1]
    new_tokens = generated[:, decode_start:] if generated.shape[1] > decode_start else generated
    decoded = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [clean_generated_caption(value, prompt=prompt) for value in decoded]
