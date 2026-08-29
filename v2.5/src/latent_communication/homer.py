"""HOMER-compatible staged planning contracts.

The wording below follows Appendix B.1--B.3 of HOMER (ICLR 2026).  The
pipeline deliberately keeps description, script opposition, and local/global
association as separate calls; collapsing them into one JSON response changes
the method and makes field-specific latent communication impossible to audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


DESCRIPTION_PROMPT = """Describe the cartoon image in detail. Include the location, characters, facial expressions, actions, objects, relationships, and any visibly unusual or conflicting elements. Stay grounded in visible evidence and do not propose a caption."""

CONFLICT_SYSTEM_PROMPT = """Based on the Script Opposition theory from the General Theory of Verbal Humor (GTVH), analyze the given cartoon description and identify two or more conflict scripts exists in the description. A script refers to a bundle of knowledge or expectations about a particular situation. Script opposition occurs when two conflicting scripts (i.e., scenarios, expectations, or frames) are brought into contrast within the description, creating a basis for potential humor. In your answer, list pairs of conflicting scripts (each as phrases or a short sentence) that are opposed or contrasted within the description. Just present the conflicting script pairs directly."""

IMAGINATION_GLOBAL_SYSTEM_PROMPT = """Given conflict scripts and a cartoon, your task is to identify the main entities mentioned. For each identified entity, generate a logical chain of three relevant entities, each based directly on the previous one. Associations may include ingredients, containers, sources, related objects, or common companions. Output JSON: key is the entity, value is a list of three such imaginations."""

IMAGINATION_LOCAL_SYSTEM_PROMPT = """Given conflict scripts and a cartoon description, your task is to identify the main entities mentioned. For each identified entity, generate a logical chain of three relevant entities, each based directly on the previous one. Associations may include ingredients, containers, sources, related objects, or common companions. Output JSON: key is the entity, value is a list of three such imaginations."""

CAPTION_SYSTEM_PROMPT = """Using the provided free-association chains, conflict scripts, and the cartoon description, generate a witty, funny and smart caption that spotlights the central incongruity and naturally combines key keywords in chains. Consider techniques such as narrative setups, linguistic styles and puns, but keep it short, concise and suitable as a cartoon tagline."""


@dataclass(frozen=True)
class HomerPlan:
    grounding: str
    conflict: str
    associative_imagination_local: str
    associative_imagination_global: str
    culture_context: str | None = None

    @property
    def associative_imagination(self) -> str:
        return json.dumps(
            {
                "local": _json_or_text(self.associative_imagination_local),
                "global": _json_or_text(self.associative_imagination_global),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounding": self.grounding,
            "conflict": self.conflict,
            "associative_imagination": {
                "local": _json_or_text(self.associative_imagination_local),
                "global": _json_or_text(self.associative_imagination_global),
            },
            "culture_context": self.culture_context,
        }


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip()


def conflict_user_prompt(description: str) -> str:
    return f"Description:\n{description.strip()}"


def local_imagination_user_prompt(description: str, conflict: str) -> str:
    return f"Cartoon description:\n{description.strip()}\n\nConflict scripts:\n{conflict.strip()}"


def global_imagination_user_prompt(conflict: str) -> str:
    return f"Conflict scripts:\n{conflict.strip()}"


def text_generator_context(plan: HomerPlan, *, include_culture: bool = False) -> str:
    parts = [
        f"Cartoon description:\n{plan.grounding}",
        f"Conflict scripts:\n{plan.conflict}",
        f"Free-association chains:\n{plan.associative_imagination}",
    ]
    if include_culture and plan.culture_context:
        parts.append(f"Optional culture context:\n{plan.culture_context}")
    return CAPTION_SYSTEM_PROMPT + "\n\n" + "\n\n".join(parts)


def latent_generator_context(plan: HomerPlan, *, include_culture: bool = False) -> str:
    """Only grounding remains textual; conflict/imagination arrive as typed slots."""
    result = CAPTION_SYSTEM_PROMPT + f"\n\nCartoon description:\n{plan.grounding}"
    if include_culture and plan.culture_context:
        result += f"\n\nOptional culture context:\n{plan.culture_context}"
    return result
