"""Verbatim public prompts from HOMER Appendix B, with role separation.

Only whitespace is normalized.  The paper does not disclose a verbatim prompt
for situation-description generation, so strict reproduction consumes the
standard descriptions shipped with the benchmark instead of inventing one.
"""
from __future__ import annotations

from typing import Any

CONFLICT_SYSTEM = (
    "Based on the Script Opposition theory from the General Theory of Verbal Humor (GTVH), "
    "analyze the given cartoon description and identify two or more conflict scripts exists in the "
    "description. A script refers to a bundle of knowledge or expectations about a particular "
    "situation. Script opposition occurs when two conflicting scripts (i.e., scenarios, expectations, "
    "or frames) are brought into contrast within the description, creating a basis for potential humor. "
    "In your answer, list pairs of conflicting scripts (each as phrases or a short sentence) that are "
    "opposed or contrasted within the description. Just present the conflicting script pairs directly."
)

IMAGINATION_GLOBAL_SYSTEM = (
    "Given conflict scripts and a cartoon, your task is to identify the main entities mentioned. "
    "For each identified entity, generate a logical chain of three relevant entities, each based "
    "directly on the previous one. Associations may include ingredients, containers, sources, related "
    "objects, or common companions. Output JSON: key is the entity, value is a list of three such imaginations."
)

IMAGINATION_LOCAL_SYSTEM = (
    "Given conflict scripts and a cartoon description, your task is to identify the main entities "
    "mentioned. For each identified entity, generate a logical chain of three relevant entities, each "
    "based directly on the previous one. Associations may include ingredients, containers, sources, "
    "related objects, or common companions. Output JSON: key is the entity, value is a list of three such imaginations."
)

CAPTION_SYSTEM = (
    "Using the provided free-association chains, conflict scripts, and the cartoon description, "
    "generate a witty, funny and smart caption that spotlights the central incongruity and naturally "
    "combines key keywords in chains. Consider techniques such as narrative setups, linguistic styles "
    "and puns, but keep it short, concise and suitable as a cartoon tagline."
)


def text_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def image_part(image: str) -> dict[str, str]:
    return {"type": "image", "image": image}


def conflict_messages(description: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [text_part(CONFLICT_SYSTEM)]},
        {"role": "user", "content": [text_part(f"Description of the image: {description}")]},
    ]


def local_imagination_messages(description: str, conflicts: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [text_part(IMAGINATION_LOCAL_SYSTEM)]},
        {"role": "user", "content": [text_part(f"Description: {description}\nConflicting scripts: {conflicts}")]},
    ]


def global_imagination_messages(image: str, conflicts: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [text_part(IMAGINATION_GLOBAL_SYSTEM)]},
        {"role": "user", "content": [image_part(image), text_part(f"Conflicting scripts: {conflicts}")]},
    ]


def caption_messages(description: str, conflicts: str, path: list[str], options: str = "") -> list[dict[str, Any]]:
    user = (
        f"Description: {description}\n\nConflicting Scripts: {conflicts}\n\n"
        f"Free-associating chains of cartoon:\n{path}"
    )
    if options:
        user += f"\n\nNarrative strategy and linguistic style: {options}"
    return [
        {"role": "system", "content": [text_part(CAPTION_SYSTEM)]},
        {"role": "user", "content": [text_part(user)]},
    ]
