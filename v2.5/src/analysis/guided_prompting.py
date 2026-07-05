from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_BASE_PROMPT = (
    "Generate one short, natural, image-specific humorous caption for this image. "
    "Do not explain."
)

GUIDED_PROMPT_METHODS = (
    "plain",
    "description-only",
    "prompt-method",
    "feature-method",
    "structured-brief",
    "structured-nl",
    "structured-json",
    "hic-humor-point",
    "hic-viewpoint-tags",
    "hic-anchor-viewpoint",
    "hic-compact-json",
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_items(values: Any, max_items: int = 5) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    items: list[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in items:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _strip_source_caption(text: str, gold_caption: str | None = None) -> str:
    text = _clean_text(text)
    if not text:
        return ""
    if gold_caption:
        caption = _clean_text(gold_caption)
        if caption:
            text = re.sub(re.escape(caption), "the target joke", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe\s+gold\s+caption\b", "the target joke", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgold\s+caption\b", "target joke", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe\s+caption\b", "the target joke", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcaption\s+['\"][^'\"]{1,160}['\"]", "target joke", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def format_visual_facts(
    visual_facts: dict[str, Any] | None,
    image_description: str = "",
) -> dict[str, Any]:
    visual_facts = visual_facts or {}

    # Backward compatibility for older context files that stored humor_points.
    if "literal_description" not in visual_facts and any(
        key in visual_facts for key in ("abnormal_points", "conflict_points", "humor_angle", "avoid")
    ):
        return {
            "literal_description": _clean_text(image_description),
            "visible_objects": [],
            "visible_actions": [],
            "salient_points": _clean_items(visual_facts.get("salient_points"), max_items=4),
            "visible_text": [],
            "uncertain_or_unreadable": _clean_items(visual_facts.get("avoid"), max_items=4),
        }

    return {
        "literal_description": _clean_text(visual_facts.get("literal_description") or image_description),
        "visible_objects": _clean_items(visual_facts.get("visible_objects"), max_items=5),
        "visible_actions": _clean_items(visual_facts.get("visible_actions"), max_items=4),
        "salient_points": _clean_items(visual_facts.get("salient_points"), max_items=4),
        "visible_text": _clean_items(visual_facts.get("visible_text"), max_items=3),
        "uncertain_or_unreadable": _clean_items(visual_facts.get("uncertain_or_unreadable"), max_items=4),
    }


def _bullet_block(title: str, items: list[str]) -> str:
    if not items:
        return f"- {title}: None."
    return f"- {title}: " + "; ".join(items)


def _safe_list_dicts(value: Any, max_items: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        if len(result) >= max_items:
            break
    return result


def format_structured_humor(structured_humor: dict[str, Any] | None) -> dict[str, Any]:
    structured_humor = structured_humor or {}
    visible = structured_humor.get("visible_facts") if isinstance(structured_humor.get("visible_facts"), dict) else {}
    inferred = structured_humor.get("inferred_context") if isinstance(structured_humor.get("inferred_context"), dict) else {}
    mechanism = structured_humor.get("humor_mechanism") if isinstance(structured_humor.get("humor_mechanism"), dict) else {}
    guidance = structured_humor.get("generator_guidance") if isinstance(structured_humor.get("generator_guidance"), dict) else {}
    return {
        "visible_facts": {
            "entities": _safe_list_dicts(visible.get("entities"), max_items=4),
            "relations": _safe_list_dicts(visible.get("relations"), max_items=4),
        },
        "inferred_context": {
            "items": _safe_list_dicts(inferred.get("items"), max_items=3),
        },
        "humor_mechanism": {
            "type": _clean_text(mechanism.get("type") or "none"),
            "anchors": _clean_items(mechanism.get("anchors"), max_items=4),
            "expected_frame": _clean_text(mechanism.get("expected_frame")),
            "observed_violation": _clean_text(mechanism.get("observed_violation")),
            "resolution": _clean_text(mechanism.get("resolution")),
            "caption_strategy": _clean_text(mechanism.get("caption_strategy") or "none"),
        },
        "generator_guidance": {
            "useful": bool(guidance.get("useful")),
            "one_line_cue": _clean_text(guidance.get("one_line_cue")),
        },
        "warnings": _clean_items(structured_humor.get("warnings"), max_items=3),
    }


def format_hic_humor_viewpoint(
    humor_viewpoint: dict[str, Any] | None,
    gold_caption: str | None = None,
) -> dict[str, Any]:
    value = humor_viewpoint or {}
    anchors = []
    raw_anchors = value.get("visual_anchors")
    if isinstance(raw_anchors, list):
        for item in raw_anchors:
            if not isinstance(item, dict):
                continue
            label = _clean_text(item.get("label"))
            if not label:
                continue
            anchors.append(
                {
                    "id": _clean_text(item.get("id")),
                    "label": label,
                    "role": _strip_source_caption(str(item.get("role") or ""), gold_caption=gold_caption),
                    "evidence": _clean_text(item.get("evidence")),
                }
            )
            if len(anchors) >= 4:
                break
    required = _clean_items(value.get("required_viewpoints"), max_items=4)
    primary = _clean_text(value.get("primary_viewpoint")) or (required[0] if required else "full_image")
    if primary not in required:
        required.insert(0, primary)
    return {
        "literal_image_description": _clean_text(value.get("literal_image_description")),
        "humor_type": _clean_text(value.get("humor_type") or "unclear_or_weak"),
        "humor_point": _strip_source_caption(str(value.get("humor_point") or ""), gold_caption=gold_caption),
        "gold_joke_explanation": _strip_source_caption(
            str(value.get("gold_joke_explanation") or ""), gold_caption=gold_caption
        ),
        "visual_anchors": anchors,
        "required_viewpoints": required[:4] or ["full_image"],
        "primary_viewpoint": primary,
        "needs_external_knowledge": bool(value.get("needs_external_knowledge")),
        "confidence": _clean_text(value.get("confidence") or "low"),
        "uncertainty": _clean_text(value.get("uncertainty")),
    }


def _entity_label(entity: dict[str, Any]) -> str:
    entity_id = _clean_text(entity.get("id"))
    label = _clean_text(entity.get("label"))
    attributes = _clean_items(entity.get("attributes"), max_items=4)
    prefix = f"{entity_id}: " if entity_id else ""
    text = f"{prefix}{label or 'visible item'}"
    if attributes:
        text += " (" + "; ".join(attributes) + ")"
    return text


def _relation_label(relation: dict[str, Any]) -> str:
    subject = _clean_text(relation.get("subject"))
    predicate = _clean_text(relation.get("predicate"))
    obj = _clean_text(relation.get("object"))
    return " ".join(part for part in (subject, predicate, obj) if part)


def _structured_json_payload(structured_humor: dict[str, Any]) -> dict[str, Any]:
    structured = format_structured_humor(structured_humor)
    inferred_items = []
    for item in structured["inferred_context"]["items"]:
        inferred_items.append(
            {
                "claim": _clean_text(item.get("claim")),
                "basis": _clean_text(item.get("basis")),
            }
        )
    return {
        "visible_facts": structured["visible_facts"],
        "inferred_context": {"items": inferred_items},
        "humor_mechanism": structured["humor_mechanism"],
        "generator_guidance": structured["generator_guidance"],
        "avoid": structured["warnings"],
    }


def build_description_only_prompt(
    image_description: str,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    lines = [
        "A conservative visual description from another vision model is provided below.",
        "Trust the image first, and ignore the description if it conflicts with the image.",
        "Do not repeat or mention the description.",
        "Return only one short caption.",
        "",
        f"Image description: {_clean_text(image_description) or 'None.'}",
        "",
        base_prompt,
    ]
    return "\n".join(lines).strip()


def build_prompt_method_prompt(
    image_description: str,
    visual_facts: dict[str, Any] | None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    """Build a natural-language prompt using conservative VLM visual facts."""
    facts = format_visual_facts(visual_facts, image_description=image_description)
    lines = [
        "Possible visual facts from another vision model are provided below.",
        "They may be imperfect. Trust the image first, and ignore any fact that conflicts with the image.",
        "Do not invent details beyond the image.",
        "Do not mention the visual notes.",
        "Return only one short caption.",
        "",
        f"Literal description: {facts['literal_description'] or 'None.'}",
        _bullet_block("Visible objects", facts["visible_objects"]),
        _bullet_block("Visible actions or poses", facts["visible_actions"]),
        _bullet_block("Visually salient content", facts["salient_points"]),
        _bullet_block("Readable text", facts["visible_text"]),
    ]
    if facts["uncertain_or_unreadable"]:
        lines.append(_bullet_block("Do not rely on", facts["uncertain_or_unreadable"]))
    lines.extend(["", base_prompt])
    return "\n".join(lines).strip()


def build_feature_method_prompt(
    image_description: str,
    visual_facts: dict[str, Any] | None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    """Build a structured feature block prompt using visual facts."""
    facts = format_visual_facts(visual_facts, image_description=image_description)
    lines = [
        "Treat <visual_facts> as imperfect auxiliary visual facts.",
        "Trust the image first; ignore any feature that conflicts with the image.",
        "Do not repeat the feature text.",
        "Return only one short caption.",
        "",
        "<visual_facts>",
        f"literal_description: {facts['literal_description'] or 'None'}",
        f"visible_objects: {' | '.join(facts['visible_objects']) or 'None'}",
        f"visible_actions: {' | '.join(facts['visible_actions']) or 'None'}",
        f"salient_points: {' | '.join(facts['salient_points']) or 'None'}",
        f"visible_text: {' | '.join(facts['visible_text']) or 'None'}",
        f"uncertain_or_unreadable: {' | '.join(facts['uncertain_or_unreadable']) or 'None'}",
        "</visual_facts>",
        "",
        base_prompt,
    ]
    return "\n".join(lines).strip()


def build_structured_brief_prompt(
    structured_humor: dict[str, Any] | None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    structured = format_structured_humor(structured_humor)
    entities = [_entity_label(entity) for entity in structured["visible_facts"]["entities"]]
    relations = [_relation_label(relation) for relation in structured["visible_facts"]["relations"]]
    mechanism = structured["humor_mechanism"]
    lines = [
        "Brief structured visual guidance from another vision model is provided below.",
        "It may be imperfect. Trust the image first, and ignore anything that conflicts with the image.",
        "Use it only as an attention hint; do not reuse its wording or explain the hint.",
        "Return only one short caption.",
        "",
        "Visible anchors:",
    ]
    lines.extend(f"- {entity}" for entity in entities) if entities else lines.append("- None.")
    lines.append("")
    lines.append("Visible relations:")
    lines.extend(f"- {relation}" for relation in relations if relation) if relations else lines.append("- None.")
    lines.extend(
        [
            "",
            "Humor hint:",
            f"- Mechanism type: {mechanism['type'] or 'none'}",
            f"- Suggested style: {mechanism['caption_strategy'] or 'none'}",
        ]
    )
    if structured["warnings"]:
        lines.extend(["", _bullet_block("Avoid relying on", structured["warnings"])])
    lines.extend(["", base_prompt])
    return "\n".join(lines).strip()


def build_structured_nl_prompt(
    structured_humor: dict[str, Any] | None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    structured = format_structured_humor(structured_humor)
    entities = [_entity_label(entity) for entity in structured["visible_facts"]["entities"]]
    relations = [_relation_label(relation) for relation in structured["visible_facts"]["relations"]]
    inferred = []
    for item in structured["inferred_context"]["items"]:
        claim = _clean_text(item.get("claim"))
        basis = _clean_text(item.get("basis"))
        if claim and basis:
            inferred.append(f"{claim} Basis: {basis}")
        elif claim:
            inferred.append(claim)
    mechanism = structured["humor_mechanism"]
    guidance = structured["generator_guidance"]
    lines = [
        "Structured humor guidance from another vision model is provided below.",
        "It may be imperfect. Trust the image first, and ignore anything that conflicts with the image.",
        "Use the guidance to understand the visual joke, but do not repeat or explain it.",
        "Return only one short caption.",
        "",
        "Relevant visible facts:",
    ]
    lines.extend(f"- {entity}" for entity in entities) if entities else lines.append("- None.")
    lines.append("")
    lines.append("Visible relations:")
    lines.extend(f"- {relation}" for relation in relations if relation) if relations else lines.append("- None.")
    lines.append("")
    lines.append("Possible interpretation:")
    lines.extend(f"- {item}" for item in inferred) if inferred else lines.append("- None.")
    lines.extend(
        [
            "",
            "Humor structure:",
            f"- Type: {mechanism['type'] or 'none'}",
            f"- Expected: {mechanism['expected_frame'] or 'None.'}",
            f"- Violation: {mechanism['observed_violation'] or 'None.'}",
            f"- Reinterpretation: {mechanism['resolution'] or 'None.'}",
            f"- Suggested strategy: {mechanism['caption_strategy'] or 'none'}",
            f"- Compact cue: {guidance['one_line_cue'] or 'None.'}",
        ]
    )
    if structured["warnings"]:
        lines.extend(["", _bullet_block("Avoid relying on", structured["warnings"])])
    lines.extend(["", base_prompt])
    return "\n".join(lines).strip()


def build_structured_json_prompt(
    structured_humor: dict[str, Any] | None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    payload = _structured_json_payload(structured_humor or {})
    compact_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    lines = [
        "Auxiliary structured humor guidance from another vision model is provided below as JSON.",
        "It may be imperfect. Trust the image first, and ignore anything that conflicts with the image.",
        "Do not repeat the JSON, do not explain it, and do not output multiple captions.",
        "Return only one short caption.",
        "",
        f"<structured_humor>{compact_json}</structured_humor>",
        "",
        base_prompt,
    ]
    return "\n".join(lines).strip()


def _anchor_viewpoint_rows(viewpoint: dict[str, Any]) -> list[str]:
    rows = []
    for anchor in viewpoint["visual_anchors"]:
        label = _clean_text(anchor.get("label"))
        evidence = _clean_text(anchor.get("evidence"))
        role = _clean_text(anchor.get("role"))
        parts = [label]
        if evidence:
            parts.append(f"visible evidence: {evidence}")
        if role:
            parts.append(f"joke role: {role}")
        text = "; ".join(part for part in parts if part)
        if text:
            rows.append(text)
    return rows


def build_hic_humor_point_prompt(
    humor_viewpoint: dict[str, Any] | None,
    gold_caption: str | None = None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    viewpoint = format_hic_humor_viewpoint(humor_viewpoint, gold_caption=gold_caption)
    lines = [
        "A gold-caption-derived humor target note is provided below for analysis only.",
        "It may be imperfect. Trust the image first.",
        "Do not quote, copy, or paraphrase the note; use it only to aim at the same kind of visual joke.",
        "Return only one short caption.",
        "",
        f"Literal scene: {viewpoint['literal_image_description'] or 'None.'}",
        f"Humor type: {viewpoint['humor_type'] or 'none'}",
        f"Humor target: {viewpoint['humor_point'] or 'None.'}",
    ]
    if viewpoint["needs_external_knowledge"]:
        lines.append("External knowledge may be needed; avoid obscure references unless they are obvious from the image.")
    lines.extend(["", base_prompt])
    return "\n".join(lines).strip()


def build_hic_viewpoint_tags_prompt(
    humor_viewpoint: dict[str, Any] | None,
    gold_caption: str | None = None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    viewpoint = format_hic_humor_viewpoint(humor_viewpoint, gold_caption=gold_caption)
    lines = [
        "Minimal visual attention tags from another vision model are provided below.",
        "Use them only to decide where to look in the image. Do not mention the tags.",
        "Return only one short caption.",
        "",
        f"Humor type: {viewpoint['humor_type'] or 'none'}",
        f"Primary viewpoint: {viewpoint['primary_viewpoint']}",
        f"Required viewpoints: {' | '.join(viewpoint['required_viewpoints'])}",
    ]
    if viewpoint["uncertainty"]:
        lines.append(f"Uncertainty: {viewpoint['uncertainty']}")
    lines.extend(["", base_prompt])
    return "\n".join(lines).strip()


def build_hic_anchor_viewpoint_prompt(
    humor_viewpoint: dict[str, Any] | None,
    gold_caption: str | None = None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    viewpoint = format_hic_humor_viewpoint(humor_viewpoint, gold_caption=gold_caption)
    anchor_rows = _anchor_viewpoint_rows(viewpoint)
    lines = [
        "Visual joke annotations from another vision model are provided below.",
        "They may be imperfect. Trust the image first and ignore anything that conflicts with it.",
        "Use the annotations as an attention map; do not repeat or explain them.",
        "Return only one short caption.",
        "",
        f"Literal scene: {viewpoint['literal_image_description'] or 'None.'}",
        f"Humor type: {viewpoint['humor_type'] or 'none'}",
        f"Primary viewpoint: {viewpoint['primary_viewpoint']}",
        f"Required viewpoints: {' | '.join(viewpoint['required_viewpoints'])}",
        "",
        "Visual anchors:",
    ]
    lines.extend(f"- {row}" for row in anchor_rows) if anchor_rows else lines.append("- None.")
    lines.extend(["", f"Humor target: {viewpoint['humor_point'] or 'None.'}"])
    if viewpoint["needs_external_knowledge"]:
        lines.append("External knowledge flag: yes. Prefer an image-grounded caption over obscure reference copying.")
    lines.extend(["", base_prompt])
    return "\n".join(lines).strip()


def build_hic_compact_json_prompt(
    humor_viewpoint: dict[str, Any] | None,
    gold_caption: str | None = None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> str:
    viewpoint = format_hic_humor_viewpoint(humor_viewpoint, gold_caption=gold_caption)
    payload = {
        "scene": viewpoint["literal_image_description"],
        "type": viewpoint["humor_type"],
        "target": viewpoint["humor_point"],
        "primary_view": viewpoint["primary_viewpoint"],
        "views": viewpoint["required_viewpoints"],
        "anchors": [
            {
                "label": anchor.get("label"),
                "evidence": anchor.get("evidence"),
                "role": anchor.get("role"),
            }
            for anchor in viewpoint["visual_anchors"]
        ],
        "external_knowledge": viewpoint["needs_external_knowledge"],
    }
    compact_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    lines = [
        "Auxiliary visual joke annotations are provided below as compact JSON.",
        "Trust the image first. Do not repeat the JSON or explain it.",
        "Return only one short caption.",
        "",
        f"<joke_annotations>{compact_json}</joke_annotations>",
        "",
        base_prompt,
    ]
    return "\n".join(lines).strip()


def build_guided_prompt(
    method: str,
    image_description: str,
    visual_facts: dict[str, Any] | None,
    base_prompt: str = DEFAULT_BASE_PROMPT,
    structured_humor: dict[str, Any] | None = None,
    humor_viewpoint: dict[str, Any] | None = None,
    gold_caption: str | None = None,
) -> str:
    if method == "plain":
        return base_prompt
    if method == "description-only":
        return build_description_only_prompt(image_description, base_prompt=base_prompt)
    if method == "prompt-method":
        return build_prompt_method_prompt(image_description, visual_facts, base_prompt=base_prompt)
    if method == "feature-method":
        return build_feature_method_prompt(image_description, visual_facts, base_prompt=base_prompt)
    if method == "structured-brief":
        return build_structured_brief_prompt(structured_humor, base_prompt=base_prompt)
    if method == "structured-nl":
        return build_structured_nl_prompt(structured_humor, base_prompt=base_prompt)
    if method == "structured-json":
        return build_structured_json_prompt(structured_humor, base_prompt=base_prompt)
    if method == "hic-humor-point":
        return build_hic_humor_point_prompt(humor_viewpoint, gold_caption=gold_caption, base_prompt=base_prompt)
    if method == "hic-viewpoint-tags":
        return build_hic_viewpoint_tags_prompt(humor_viewpoint, gold_caption=gold_caption, base_prompt=base_prompt)
    if method == "hic-anchor-viewpoint":
        return build_hic_anchor_viewpoint_prompt(humor_viewpoint, gold_caption=gold_caption, base_prompt=base_prompt)
    if method == "hic-compact-json":
        return build_hic_compact_json_prompt(humor_viewpoint, gold_caption=gold_caption, base_prompt=base_prompt)
    raise ValueError(f"Unknown guided generation method: {method}")
