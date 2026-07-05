from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


VISUAL_FACTS_PROMPT = """You are a conservative visual fact extractor for image caption generation.

Your first priority is accuracy. It is better to return fewer facts than to guess.
Do NOT write a humorous caption. Do NOT infer a joke.

Return only valid JSON with this exact schema:
{
  "literal_description": "two short sentences describing only clearly visible content",
  "visible_objects": ["clearly visible object, person, animal, or scene element"],
  "visible_actions": ["clearly visible physical action or pose"],
  "salient_points": ["large, central, or visually dominant visible content"],
  "visible_text": ["exact readable text visible in the image"],
  "uncertain_or_unreadable": ["important detail that is unclear, unreadable, or risky to guess"]
}

Strict grounding rules:
- Mention only things that are clearly visible in the image.
- Do not infer hidden story, personality, motivation, social role, emotion, relationship, profession, age, nationality, brand, or identity.
- Do not guess what a person is thinking or intending.
- Do not describe unreadable text as readable text.
- visible_text must contain exact words only. If text is not clearly readable, put "unreadable text" in uncertain_or_unreadable.
- Do not use speculative words in literal_description or fact lists: likely, possibly, appears, seems, suggesting, maybe.
- Prefer generic object names over risky specific labels when the object is ambiguous.
- Do not label something as a weapon, gun, knife, camera, tool, machine, badge, uniform, sign, or readable document unless it is unmistakable.
- For ambiguous objects, write a generic phrase such as "large object", "handheld object", "wooden object", or "dark object", and add the ambiguity to uncertain_or_unreadable.
- Do not describe facial expression unless it is unmistakable; if uncertain, omit it.
- Do not generate abnormal points, conflict points, or humor angles.
- Keep lists short: at most 5 visible_objects, 4 visible_actions, 4 salient_points, 3 visible_text items, 4 uncertain_or_unreadable items.
- If a field has no reliable content, use an empty list.

Return only JSON. No markdown. No explanation."""


IMAGE_DESCRIPTION_PROMPT = """Describe only the clearly visible content of this image.

Return 2 short sentences only:
1. The background or setting, if clearly visible.
2. The main visible subjects, objects, and actions.

Rules:
- Be conservative and visual.
- If something is uncertain, do not mention it.
- Do not use speculative words: likely, possibly, appears, seems, suggesting, maybe.
- Do not guess emotion, identity, profession, relationship, hidden story, or unreadable text.
- Do not write a joke.
- Do not mention that you are an AI."""


STRUCTURED_HUMOR_ZERO_SHOT_PROMPT = """You are a visual humor analyst preparing structured guidance for a caption generator.

Analyze the attached image only. Extract the visible anchors and the likely humor mechanism, but do NOT write a caption or a punchline.

Return only valid JSON with exactly this schema:
{
  "visible_facts": {
    "entities": [
      {
        "id": "e1",
        "label": "generic visible entity label",
        "attributes": ["short visible attribute"]
      }
    ],
    "relations": [
      {
        "subject": "e1",
        "predicate": "short visible relation",
        "object": "e2"
      }
    ]
  },
  "inferred_context": {
    "items": [
      {
        "claim": "reasonable interpretation separated from visible facts",
        "confidence": "low|medium|high",
        "basis": "visible evidence for the interpretation"
      }
    ]
  },
  "humor_mechanism": {
    "type": "none|scale_contrast|role_mismatch|action_contrast|visual_incongruity|composition_misread|text_image_contrast|cultural_reference|other",
    "anchors": ["e1"],
    "expected_frame": "normal expectation or empty string",
    "observed_violation": "visible mismatch or empty string",
    "resolution": "how a caption could reinterpret the mismatch, or empty string",
    "caption_strategy": "none|understatement|absurdity|deadpan|role_reversal|literal_misread|contrast|wordplay|other"
  },
  "generator_guidance": {
    "useful": true,
    "one_line_cue": "one compact non-caption cue for the generator, or empty string"
  },
  "warnings": ["uncertain detail to avoid"]
}

Rules:
- Keep visible_facts strictly visual. Use generic labels such as person, animal, object, clothing, text, sign, vehicle, food, or furniture.
- Prefer "person" over child, adult, man, woman, boy, or girl unless the age or gender category is unmistakable and necessary for the humor mechanism.
- Put interpretation, social context, emotion, intention, identity, cultural reference, or common-sense knowledge only in inferred_context.
- Do not pretend inferred_context is directly visible.
- Do not use appears, seems, suggests, likely, maybe, probably, trying, wants, or intends in visible_facts, humor_mechanism, or generator_guidance.
- In inferred_context, avoid unverifiable stories such as playing, communicating, working, driving, stealing, guarding, or escaping unless the physical action is directly visible.
- Use at most 4 entities, 4 relations, 3 inferred_context items, and 3 warnings.
- Attributes must be short and grounded in the image.
- If no clear humor mechanism is visible, set humor_mechanism.type to "none", caption_strategy to "none", generator_guidance.useful to false, and one_line_cue to an empty string.
- If a mechanism is useful, one_line_cue must describe the visual mismatch or reinterpretation. It must not be a final caption.
- Do not use markdown, comments, trailing commas, or extra top-level keys.
- Use English.

Return JSON only."""


STRUCTURED_HUMOR_FEW_SHOT_PROMPT = STRUCTURED_HUMOR_ZERO_SHOT_PROMPT + """

Text-only formatting examples. These are examples of the output style, not facts about the attached image.

Example A:
{
  "visible_facts": {
    "entities": [
      {"id": "e1", "label": "person", "attributes": ["holding a cup"]},
      {"id": "e2", "label": "cup", "attributes": ["large relative to the hand"]}
    ],
    "relations": [
      {"subject": "e1", "predicate": "holding", "object": "e2"}
    ]
  },
  "inferred_context": {
    "items": [
      {"claim": "The cup may be comically oversized for an ordinary drink.", "confidence": "medium", "basis": "visible scale contrast between the cup and hand"}
    ]
  },
  "humor_mechanism": {
    "type": "scale_contrast",
    "anchors": ["e1", "e2"],
    "expected_frame": "A handheld cup normally fits comfortably in one hand.",
    "observed_violation": "The cup is disproportionately large.",
    "resolution": "Treat an ordinary drink as absurdly oversized.",
    "caption_strategy": "understatement"
  },
  "generator_guidance": {
    "useful": true,
    "one_line_cue": "The ordinary cup moment can be framed as absurd because the cup is oversized."
  },
  "warnings": []
}

Example B:
{
  "visible_facts": {
    "entities": [
      {"id": "e1", "label": "dog", "attributes": ["positioned at a steering wheel"]},
      {"id": "e2", "label": "steering wheel", "attributes": []}
    ],
    "relations": [
      {"subject": "e1", "predicate": "positioned behind", "object": "e2"}
    ]
  },
  "inferred_context": {
    "items": [
      {"claim": "The animal can be interpreted as occupying a human driver role.", "confidence": "medium", "basis": "the dog is placed behind the steering wheel"}
    ]
  },
  "humor_mechanism": {
    "type": "role_mismatch",
    "anchors": ["e1", "e2"],
    "expected_frame": "A person normally controls a steering wheel.",
    "observed_violation": "A dog is positioned where the driver would be.",
    "resolution": "Treat the dog as if it has taken a human driving role.",
    "caption_strategy": "role_reversal"
  },
  "generator_guidance": {
    "useful": true,
    "one_line_cue": "The animal-in-human-role mismatch is the useful joke anchor."
  },
  "warnings": []
}

Now analyze the attached image. Return JSON only."""


STRUCTURED_HUMOR_PROMPTS = {
    "structured-v1": STRUCTURED_HUMOR_ZERO_SHOT_PROMPT,
    "structured-v1-fewshot": STRUCTURED_HUMOR_FEW_SHOT_PROMPT,
}

ALLOWED_HUMOR_TYPES = {
    "none",
    "scale_contrast",
    "role_mismatch",
    "action_contrast",
    "visual_incongruity",
    "composition_misread",
    "text_image_contrast",
    "cultural_reference",
    "other",
}
ALLOWED_CAPTION_STRATEGIES = {
    "none",
    "understatement",
    "absurdity",
    "deadpan",
    "role_reversal",
    "literal_misread",
    "contrast",
    "wordplay",
    "other",
}


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def get_structured_humor_prompt(version: str = "structured-v1") -> str:
    try:
        return STRUCTURED_HUMOR_PROMPTS[version]
    except KeyError as exc:
        choices = ", ".join(sorted(STRUCTURED_HUMOR_PROMPTS))
        raise ValueError(f"Unknown structured humor prompt version: {version}. Choices: {choices}") from exc


def get_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def load_qwen_vl(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[Any, Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Missing Qwen-VL inference dependencies. Install requirements.txt first."
        ) from exc

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    processor_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    model.eval()
    return model, processor, process_vision_info


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"VLM response did not contain JSON: {text[:300]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("VLM response JSON is not an object")
    return value


def _extract_field_block(text: str, field: str, ordered_fields: list[str]) -> str:
    match = re.search(rf'"?{re.escape(field)}"?\s*:\s*', text)
    if not match:
        return ""
    start = match.end()
    end = len(text)
    current_index = ordered_fields.index(field)
    for next_field in ordered_fields[current_index + 1 :]:
        next_match = re.search(rf',?\s*\n?\s*"?{re.escape(next_field)}"?\s*:', text[start:])
        if next_match:
            end = start + next_match.start()
            break
    return text[start:end].strip().strip(",").strip()


def _clean_loose_value(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^[\[\]\{\},\s]+", "", value)
    value = re.sub(r"[\[\]\{\},\s]+$", "", value)
    value = re.sub(r"^\d+[\.\)]\s*", "", value)
    value = re.sub(r"^[-*]\s*", "", value)
    value = value.strip().strip('"').strip("'").strip()
    return " ".join(value.split())


def _parse_loose_list(block: str, max_items: int = 4) -> list[str]:
    if not block:
        return []
    block = block.strip()
    if block.startswith("[") and "]" in block:
        block = block[1 : block.rfind("]")]

    candidates = [line.strip() for line in block.replace("\r", "\n").split("\n") if line.strip()]
    if len(candidates) <= 1:
        candidates = re.split(r'"\s*,\s*"|,\s*(?=[A-Z0-9])|;\s*', block)

    items: list[str] = []
    for candidate in candidates:
        text = _clean_loose_value(candidate)
        if text and text not in items:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def extract_visual_facts_loose(text: str) -> dict[str, Any]:
    """Recover expected fields from almost-JSON VLM output.

    Qwen-VL sometimes returns valid-looking JSON with an unescaped quote or a
    missing comma inside a list item. For this task, losing one item is worse
    than dropping the whole image, so we recover the known schema by field name.
    """
    fields = [
        "literal_description",
        "visible_objects",
        "visible_actions",
        "salient_points",
        "visible_text",
        "uncertain_or_unreadable",
    ]
    recovered: dict[str, Any] = {}
    for field in fields:
        block = _extract_field_block(text, field, fields)
        if field == "literal_description":
            lines = [line for line in block.replace("\r", "\n").split("\n") if line.strip()]
            recovered[field] = _clean_loose_value(lines[0] if lines else block)
        else:
            max_items = 3 if field == "visible_text" else 4
            recovered[field] = _parse_loose_list(block, max_items=max_items)
    return recovered


def _clean_text(value: Any, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    text = text.strip().strip('"').strip("'").strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."
    return text


def _as_list(value: Any, max_items: int = 4, max_chars: int = 180) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = _clean_text(item, max_chars=max_chars)
        if text and text not in items:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _clean_choice(value: Any, allowed: set[str], default: str) -> str:
    text = _clean_text(value, max_chars=80).lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    if text in allowed:
        return text
    return default


def _normalize_entity(value: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    entity_id = _clean_text(value.get("id"), max_chars=24) or f"e{index}"
    entity_id = re.sub(r"[^A-Za-z0-9_-]+", "", entity_id) or f"e{index}"
    label = _clean_text(value.get("label"), max_chars=80)
    if not label:
        return None
    return {
        "id": entity_id,
        "label": label,
        "attributes": _as_list(value.get("attributes"), max_items=4, max_chars=120),
    }


def _normalize_relation(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    subject = _clean_text(value.get("subject"), max_chars=24)
    predicate = _clean_text(value.get("predicate"), max_chars=80)
    obj = _clean_text(value.get("object"), max_chars=24)
    if not subject or not predicate or not obj:
        return None
    return {"subject": subject, "predicate": predicate, "object": obj}


def _normalize_inferred_item(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    claim = _clean_text(value.get("claim"), max_chars=220)
    basis = _clean_text(value.get("basis"), max_chars=180)
    if not claim:
        return None
    confidence = _clean_choice(value.get("confidence"), {"low", "medium", "high"}, "low")
    return {"claim": claim, "confidence": confidence, "basis": basis}


def normalize_visual_facts(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "literal_description": _clean_text(value.get("literal_description"), max_chars=320),
        "visible_objects": _as_list(value.get("visible_objects"), max_items=5),
        "visible_actions": _as_list(value.get("visible_actions"), max_items=4),
        "salient_points": _as_list(value.get("salient_points"), max_items=4),
        "visible_text": _as_list(value.get("visible_text"), max_items=3),
        "uncertain_or_unreadable": _as_list(value.get("uncertain_or_unreadable"), max_items=4),
    }


def normalize_structured_humor(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    visible = value.get("visible_facts") if isinstance(value.get("visible_facts"), dict) else {}
    inferred = value.get("inferred_context") if isinstance(value.get("inferred_context"), dict) else {}
    mechanism = value.get("humor_mechanism") if isinstance(value.get("humor_mechanism"), dict) else {}
    guidance = value.get("generator_guidance") if isinstance(value.get("generator_guidance"), dict) else {}

    raw_entities = visible.get("entities") if isinstance(visible.get("entities"), list) else []
    entities: list[dict[str, Any]] = []
    for index, entity in enumerate(raw_entities, start=1):
        normalized = _normalize_entity(entity, index)
        if normalized is not None and normalized not in entities:
            entities.append(normalized)
        if len(entities) >= 4:
            break

    raw_relations = visible.get("relations") if isinstance(visible.get("relations"), list) else []
    relations: list[dict[str, str]] = []
    for relation in raw_relations:
        normalized_relation = _normalize_relation(relation)
        if normalized_relation is not None and normalized_relation not in relations:
            relations.append(normalized_relation)
        if len(relations) >= 4:
            break

    raw_inferred_items = inferred.get("items") if isinstance(inferred.get("items"), list) else []
    inferred_items: list[dict[str, str]] = []
    for item in raw_inferred_items:
        normalized_item = _normalize_inferred_item(item)
        if normalized_item is not None and normalized_item not in inferred_items:
            inferred_items.append(normalized_item)
        if len(inferred_items) >= 3:
            break

    mechanism_type = _clean_choice(mechanism.get("type"), ALLOWED_HUMOR_TYPES, "none")
    caption_strategy = _clean_choice(
        mechanism.get("caption_strategy"),
        ALLOWED_CAPTION_STRATEGIES,
        "none" if mechanism_type == "none" else "other",
    )
    one_line_cue = _clean_text(guidance.get("one_line_cue"), max_chars=240)
    useful = bool(guidance.get("useful")) and mechanism_type != "none"
    if mechanism_type == "none":
        caption_strategy = "none"
        useful = False
        one_line_cue = ""

    return {
        "visible_facts": {
            "entities": entities,
            "relations": relations,
        },
        "inferred_context": {
            "items": inferred_items,
        },
        "humor_mechanism": {
            "type": mechanism_type,
            "anchors": _as_list(mechanism.get("anchors"), max_items=4, max_chars=32),
            "expected_frame": _clean_text(mechanism.get("expected_frame"), max_chars=220),
            "observed_violation": _clean_text(mechanism.get("observed_violation"), max_chars=220),
            "resolution": _clean_text(mechanism.get("resolution"), max_chars=220),
            "caption_strategy": caption_strategy,
        },
        "generator_guidance": {
            "useful": useful,
            "one_line_cue": one_line_cue,
        },
        "warnings": _as_list(value.get("warnings"), max_items=3, max_chars=160),
    }


class QwenVLHumorContextExtractor:
    """VLM extractor for visual facts and structured humor guidance."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.model, self.processor, self.process_vision_info = load_qwen_vl(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    def _generate(
        self,
        image_path: str | Path,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(get_model_device(self.model))
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
        }
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generation_kwargs["do_sample"] = False
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = generated_ids[:, inputs["input_ids"].shape[-1] :]
        return self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def describe_image(
        self,
        image_path: str | Path,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
    ) -> tuple[str, str]:
        raw = self._generate(
            image_path,
            IMAGE_DESCRIPTION_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        description = " ".join(raw.replace("\r", "\n").split())
        return description, raw

    def extract_visual_facts(
        self,
        image_path: str | Path,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], str]:
        raw = self._generate(
            image_path,
            VISUAL_FACTS_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        try:
            parsed = extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            parsed = extract_visual_facts_loose(raw)
        return normalize_visual_facts(parsed), raw

    def extract_structured_humor(
        self,
        image_path: str | Path,
        max_new_tokens: int = 768,
        prompt_version: str = "structured-v1",
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], str, str | None]:
        prompt = get_structured_humor_prompt(prompt_version)
        raw = self._generate(
            image_path,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        try:
            parsed = extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            return normalize_structured_humor({}), raw, str(exc)
        return normalize_structured_humor(parsed), raw, None

    def analyze_image(
        self,
        image_path: str | Path,
        description_max_new_tokens: int = 96,
        humor_max_new_tokens: int = 256,
        include_visual_facts: bool = True,
        include_structured_humor: bool = False,
        structured_humor_max_new_tokens: int = 768,
        structured_humor_prompt_version: str = "structured-v1",
        structured_humor_temperature: float = 0.0,
        visual_facts_temperature: float = 0.0,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "extractor_model": self.model_name,
        }
        if include_visual_facts:
            visual_facts, raw_visual_facts = self.extract_visual_facts(
                image_path,
                max_new_tokens=humor_max_new_tokens,
                temperature=visual_facts_temperature,
            )
            result.update(
                {
                    "image_description": visual_facts.get("literal_description", ""),
                    "visual_facts": visual_facts,
                    "raw_visual_facts_response": raw_visual_facts,
                }
            )
        else:
            result["image_description"] = ""

        if include_structured_humor:
            prompt = get_structured_humor_prompt(structured_humor_prompt_version)
            structured_humor, raw_structured_humor, parse_error = self.extract_structured_humor(
                image_path,
                max_new_tokens=structured_humor_max_new_tokens,
                prompt_version=structured_humor_prompt_version,
                temperature=structured_humor_temperature,
            )
            result.update(
                {
                    "structured_humor": structured_humor,
                    "raw_structured_humor_response": raw_structured_humor,
                    "structured_humor_parse_error": parse_error,
                    "structured_humor_prompt_version": structured_humor_prompt_version,
                    "structured_humor_prompt_sha256": prompt_sha256(prompt),
                    "structured_humor_temperature": structured_humor_temperature,
                }
            )
        return result
