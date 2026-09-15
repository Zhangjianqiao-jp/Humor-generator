"""Project-defined narrative/style controls for the HOMER extension.

The HOMER paper writes the generation option as ``Omega in NS x LA`` and
discusses narrative strategy and linguistic style, but the pinned public
prompt does not publish a finite NS/LA vocabulary.  This module therefore
provides a versioned, explicit vocabulary for our *omega extension* only.
It must never be enabled in the public-code baseline used for reproduction
claims.
"""
from __future__ import annotations

from dataclasses import dataclass


# These labels are grounded in the GTVH resources and examples discussed by
# HOMER (setup/twist, role reversal, exaggeration, puns, etc.).  They are a
# project-controlled experimental vocabulary, not a claim that HOMER released
# this exact enumeration.
NARRATIVE_STRATEGIES: tuple[str, ...] = (
    "setup_punchline",
    "deadpan_observation",
    "question_answer",
    "dialogue_voice",
    "role_reversal",
    "exaggerated_consequence",
)

LINGUISTIC_STYLES: tuple[str, ...] = (
    "plain_wit",
    "pun_wordplay",
    "idiom_twist",
    "sarcastic_understatement",
    "personification",
    "metaphorical_comparison",
)


@dataclass(frozen=True)
class Omega:
    """A single controlled narrative-strategy/style pair."""

    narrative_strategy: str
    linguistic_style: str

    def __post_init__(self) -> None:
        if self.narrative_strategy not in NARRATIVE_STRATEGIES:
            raise ValueError(f"unknown narrative strategy: {self.narrative_strategy}")
        if self.linguistic_style not in LINGUISTIC_STYLES:
            raise ValueError(f"unknown linguistic style: {self.linguistic_style}")

    def prompt_text(self) -> str:
        # Keep the instruction short and explicit.  This text is only used by
        # the omega extension; the public-code baseline passes omega=None.
        return (
            "Narrative strategy: "
            f"{self.narrative_strategy}. Linguistic style: {self.linguistic_style}. "
            "Use this control while preserving image grounding, the selected "
            "conflict, and a concise caption."
        )


def parse(value: str) -> Omega:
    """Parse ``narrative_strategy|linguistic_style`` from a CLI/config value."""

    if value.count("|") != 1:
        raise ValueError("omega must be narrative_strategy|linguistic_style")
    narrative, style = (item.strip() for item in value.split("|"))
    if not narrative or not style:
        raise ValueError("omega must be narrative_strategy|linguistic_style")
    return Omega(narrative, style)


def balanced_grid() -> tuple[Omega, ...]:
    """Return the pre-registered full factorial Ω grid in stable order."""

    return tuple(
        Omega(narrative, style)
        for narrative in NARRATIVE_STRATEGIES
        for style in LINGUISTIC_STYLES
    )
