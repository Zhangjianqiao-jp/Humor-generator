"""Prompt builders copied from the pinned HOMER public repository.

The strings in this module are deliberately kept separate from
``homer.prompts``.  The latter is the historical, paper-aligned v3.5 adapter
and must not be silently changed underneath old artifacts.  This module is the
``public_code_exact`` prompt track: text and punctuation come from the pinned
HOMER commit ``d1334f295cc1a8f8f6dc67ba7e846c5939dddcec``.  A local Qwen
backend can use the same text, but replacing the official ``gpt-4o`` API with
Qwen2.5-VL is still a model-substitution experiment.

The public repository's caption prompt explicitly requests ``##Caption`` and
``##Explanation``.  The paper's mathematical ``Omega`` (narrative strategy
and linguistic style) is not enumerated in that raw prompt; an optional
``omega`` field is therefore exposed by callers as a separate, paper-aligned
variant instead of being inserted into the exact track.
"""
from __future__ import annotations

from typing import Any, Iterable


OFFICIAL_REPOSITORY = "https://github.com/Shang-hub/HOMER-Official-Implementation"
OFFICIAL_COMMIT = "d1334f295cc1a8f8f6dc67ba7e846c5939dddcec"
OFFICIAL_MODEL = "gpt-4o"
OFFICIAL_MAX_TOKENS = 1000
OFFICIAL_TEMPERATURE = 1.0


# These constants are copied from extractor.py, imaginator.py and generator.py
# at OFFICIAL_COMMIT.  Do not reflow or paraphrase them in the exact track.
DESCRIPTION_USER = (
    "In this task, you will see a cartoon, carefully review the cartoon, then write one canny and accurate description about the cartoon, highlighting unnormal or unexpected or conflict elements. Textual words in the cartoon image should be ignored. The description should include location, characters, situations, face expression and actions. Only write the description. The form is ##Vivid Description:\n"
)

CONFLICT_SYSTEM = (
    "Based on the Script Opposition theory from the General Theory of Verbal Humor (GTVH), analyze the given cartoon description and identify two or more conflict scripts exists in the description. A script refers to a bundle of knowledge or expectations about a particular situation. Script opposition occurs when two conflicting scripts (i.e., scenarios, expectations, or frames) are brought into contrast within the description, creating a basis for potential humor. In your answer, list pairs of conflicting scripts (each as phrases or a short sentence) that are opposed or contrasted within the description. Just present the conflicting script pairs directly."
)

GLOBAL_IMAGINATION_SYSTEM = (
    "Given a conflict script and a cartoon, your task is to identify the main entities mentioned. For each identified entity, generate a logical chain of three relevant entities, each based directly on the previous one. Associations may include ingredients, containers, sources, related objects, or common companions. Output JSON: key is the entity, value is a list of three such imaginations. For example: `{entity: ['idea1', 'idea2', 'idea3']}`."
)

LOCAL_IMAGINATION_SYSTEM = (
    "Given a conflict script and a cartoon description, your task is to identify the main entities mentioned. For each identified entity, generate a logical chain of three relevant entities, each based directly on the previous one. Associations may include ingredients, containers, sources, related objects, or common companions. Output JSON: key is the entity, value is a list of three such imaginations. For example: `{entity: ['idea1', 'idea2', 'idea3']}`."
)

SUMMARY_SYSTEM = (
    "Given a conflict script and cartoon description, and two JSON lists of imagined relevant entities, generate a summary by combining them into a single JSON object. Remove any duplicate imagined entities, so the final JSON includes only the most relevant, non-redundant items for each main entity. Only answer the JSON object, do not include any other text."
)

SELECT_CONFLICT_SYSTEM = (
    "Given the following description and a list of relevant conflict scripts derived from it, select two most funny conflict scripts based on the humor theory, general theory of verbal humor. Only answer the conflict script. The form is ##Conflict Scripts:\n "
)

SELECT_ENTITY_SYSTEM = (
    "Given a list of entities in the cartoon, review the given conflict script and identify two key entities that are most relevant to the conflict scripts. Two key entities are in list format, like [entity1, entity2]. DO NOT include any other text in your response."
)

CAPTION_SYSTEM = (
    "Using the provided free-association chains, conflict scripts, and the cartoon description, generate a witty, funny and smart caption that spotlights the central incongruity and naturally combines key keywords in chains. Consider techniques such as narrative setups, linguistic styles and puns, but keep it short, concise and suitable as a cartoon tagline. Please include the caption, and the explanation why this caption funny in your response. The form is ##Caption:\n and ##Explanation:\n"
)


def text_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def image_part(image: str, *, provider: str = "qwen") -> dict[str, Any]:
    """Return a provider-specific image content block.

    Qwen's processor accepts ``type=image, image=<local path>``.  The official
    OpenAI request uses ``type=image_url`` with a data URL; callers that use the
    exact API track should pass a fully formed data URL and ``provider=openai``.
    The prompt text itself is identical in both cases.
    """
    if provider == "qwen":
        return {"type": "image", "image": image}
    if provider == "openai":
        return {"type": "image_url", "image_url": {"url": image, "detail": "high"}}
    raise ValueError(f"unknown image provider: {provider}")


def _system_user(system: str, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the official two-message shape without changing the prompt.

    The pinned repository sends ``system`` as a plain string and the user turn
    as a list of multimodal/text blocks.  Keeping that distinction matters for
    an exact public-code replay: wrapping the system string in a text block is
    semantically close for Qwen, but it is not the same serialized request.
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def description_messages(image: str, *, provider: str = "qwen") -> list[dict[str, Any]]:
    # extractor.py uses one user turn (there is no system turn here).
    return [{"role": "user", "content": [
        text_part(DESCRIPTION_USER),
        image_part(image, provider=provider),
    ]}]


def conflict_messages(description: str) -> list[dict[str, Any]]:
    return _system_user(CONFLICT_SYSTEM, [text_part(f"Description:\n{description}")])


def global_imagination_messages(
    image: str, conflict_scripts: str, *, provider: str = "qwen"
) -> list[dict[str, Any]]:
    return _system_user(GLOBAL_IMAGINATION_SYSTEM, [
        image_part(image, provider=provider),
        text_part(f"The conflict script is: {conflict_scripts or ''}."),
    ])


def local_imagination_messages(description: str, conflict_scripts: str) -> list[dict[str, Any]]:
    return _system_user(LOCAL_IMAGINATION_SYSTEM, [
        text_part(f"The description is: {description or ''}."),
        text_part(f"The conflict script is: {conflict_scripts or ''}."),
    ])


def summary_messages(
    description: str,
    conflict_scripts: str,
    global_payload: str,
    local_payload: str,
) -> list[dict[str, Any]]:
    return _system_user(SUMMARY_SYSTEM, [
        text_part(f"The description is: {description}."),
        text_part(f"The conflict script is: {conflict_scripts}."),
        text_part(f"Global imagination JSON: {global_payload}"),
        text_part(f"Local imagination JSON: {local_payload}"),
    ])


def select_conflict_messages(description: str, conflicts: str) -> list[dict[str, Any]]:
    return _system_user(SELECT_CONFLICT_SYSTEM, [
        text_part(f"Description:\n{description}\n\nConflict Scripts:\n{conflicts}")
    ])


def select_entity_messages(conflict_selection: str, entities: Iterable[str]) -> list[dict[str, Any]]:
    entity_list = list(entities)
    return _system_user(SELECT_ENTITY_SYSTEM, [
        text_part(f"The Conflict Script is:\n{conflict_selection}"),
        text_part(f"List of relevant and associated entities:\n{entity_list}"),
    ])


def caption_messages(
    description: str,
    conflict_selection: str,
    free_association_text: str,
    *,
    omega: str | None = None,
) -> list[dict[str, Any]]:
    # Keep the two user content blocks used by the pinned generator.py.  The
    # text is identical if these blocks are concatenated, but the multimodal
    # request serialization is not; the exact public-code track must preserve
    # the block boundary.
    context = f"Description:\n{description}\n\nConflict Scripts:\n{conflict_selection}"
    if omega:
        # Ω is a paper-level variable; keeping it opt-in prevents the exact
        # public-code prompt from being silently modified.
        context += f"\n\nNarrative strategy and linguistic style (Omega):\n{omega}"
    return _system_user(CAPTION_SYSTEM, [
        text_part(context),
        text_part(f"Free-association chains:\n{free_association_text}"),
    ])


def extract_marked(text: str, marker: str) -> str:
    """Extract a marker section without rewriting its semantic content."""
    token = f"{marker}:"
    if token not in text:
        return text.strip()
    remainder = text.split(token, 1)[1]
    return remainder.split("##", 1)[0].strip()


def parse_caption_blob(text: str) -> tuple[str, str]:
    caption = extract_marked(text, "##Caption")
    explanation = extract_marked(text, "##Explanation")
    if "##Explanation" in caption:
        caption = caption.split("##Explanation", 1)[0].strip()
    return caption.strip(), explanation.strip()
