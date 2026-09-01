"""Paper-disclosed HOMER text pipeline, independent of any model vendor."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Protocol

from .contracts import (
    AssociationChain,
    HomerPlan,
    ValidationLimits,
    parse_conflicts,
    validate_description,
    validate_plan,
)
from .prompts import (
    caption_messages,
    conflict_messages,
    global_imagination_messages,
    local_imagination_messages,
)


class GenerationBackend(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> str: ...


class RetrievalAugmenter(Protocol):
    def augment(self, plan: HomerPlan) -> HomerPlan: ...


@dataclass(frozen=True)
class HomerRun:
    plan: HomerPlan
    selected_conflict: str
    selected_target: str
    selected_path: tuple[str, ...]
    caption: str
    seed: int


class HomerTextPipeline:
    """Extractor -> hierarchical imaginator -> retrieval -> generator.

    Strict mode requires benchmark-provided standard descriptions and a humor
    retrieval augmenter.  This prevents the previous prompt-only approximation
    from being mislabeled as a complete HOMER reproduction.
    """

    def __init__(
        self,
        backend: GenerationBackend,
        *,
        retriever: RetrievalAugmenter | None,
        strict_reproduction: bool = True,
        limits: ValidationLimits = ValidationLimits(),
    ) -> None:
        if strict_reproduction and retriever is None:
            raise ValueError("strict HOMER reproduction requires humor-retrieval augmentation")
        self.backend = backend
        self.retriever = retriever
        self.strict_reproduction = strict_reproduction
        self.limits = limits

    def plan(self, *, image: str, description: str, seed: int) -> HomerPlan:
        description = validate_description(description, self.limits)
        conflict_raw = self.backend.generate(
            conflict_messages(description), temperature=0.0, max_new_tokens=384, seed=seed
        )
        # Parse before either imagination call.  Malformed conflict output must
        # not silently contaminate both downstream views.
        conflicts = parse_conflicts(conflict_raw, self.limits)
        normalized_conflicts = " ".join(
            f"{index}. {pair.render()}" for index, pair in enumerate(conflicts, 1)
        )
        partial = validate_plan(
            description,
            normalized_conflicts,
            self.backend.generate(
                local_imagination_messages(description, normalized_conflicts),
                temperature=0.0,
                max_new_tokens=512,
                seed=seed + 1,
            ),
            self.backend.generate(
                global_imagination_messages(image, normalized_conflicts),
                temperature=0.0,
                max_new_tokens=512,
                seed=seed + 2,
            ),
            self.limits,
        )
        return self.retriever.augment(partial) if self.retriever is not None else partial

    def generate_caption(
        self,
        plan: HomerPlan,
        *,
        seed: int,
        narrative_and_language: str = "",
    ) -> HomerRun:
        rng = random.Random(seed)
        conflict = rng.choice(plan.conflicts)
        all_chains: tuple[AssociationChain, ...] = plan.local_chains + plan.global_chains
        if plan.imagination_trees:
            relevant_trees = [
                tree for tree in plan.imagination_trees
                if any(token.casefold() in conflict.render().casefold() for token in tree.backbone.path)
            ]
            tree = rng.choice(relevant_trees or list(plan.imagination_trees))
            selected_path = rng.choice(tree.paths())
            selected_target = tree.backbone.root
        else:
            relevant = [
                chain for chain in all_chains
                if chain.root.casefold() in conflict.render().casefold()
                or any(token.casefold() in conflict.render().casefold() for token in chain.steps)
            ]
            chain = rng.choice(relevant or list(all_chains))
            selected_path = chain.path
            selected_target = chain.root
        caption = self.backend.generate(
            caption_messages(
                plan.description,
                conflict.render(),
                list(selected_path),
                narrative_and_language,
            ),
            temperature=1.0,
            max_new_tokens=128,
            seed=seed,
        ).strip()
        if caption.casefold().startswith("caption:"):
            caption = caption.split(":", 1)[1].strip()
        if not caption:
            raise ValueError("caption generator returned an empty result")
        return HomerRun(plan, conflict.render(), selected_target, selected_path, caption, seed)

    def run(
        self,
        *,
        image: str,
        description: str,
        seed: int,
        narrative_and_language: str = "",
    ) -> HomerRun:
        return self.generate_caption(
            self.plan(image=image, description=description, seed=seed),
            seed=seed + 3,
            narrative_and_language=narrative_and_language,
        )
