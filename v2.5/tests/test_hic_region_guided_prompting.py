from __future__ import annotations

import json
import re

from src.analysis.guided_prompting import (
    DEFAULT_BASE_PROMPT,
    GUIDED_PROMPT_METHODS,
    build_guided_prompt,
    build_hic_compact_json_prompt,
)


GOLD = "the dog finally got his license"


def _humor_viewpoint() -> dict:
    return {
        "literal_image_description": "A dog sits behind a steering wheel.",
        "humor_type": "role_mismatch",
        "humor_point": f"The gold caption says {GOLD}, because the dog is framed as a driver.",
        "visual_anchors": [
            {
                "id": "a1",
                "label": "dog at steering wheel",
                "role": f"sets up {GOLD}",
                "evidence": "dog is behind the wheel",
            }
        ],
        "required_viewpoints": ["relation_crop"],
        "primary_viewpoint": "relation_crop",
        "needs_external_knowledge": False,
    }


def _region_annotation() -> dict:
    return {
        "annotation_version": "hic-region-v1",
        "primary_viewpoint": "relation_crop",
        "required_viewpoints": ["relation_crop"],
        "needs_full_image": False,
        "annotation_confidence": "high",
        "uncertainty": "",
        "anchors": [
            {
                "id": "a1",
                "label": "dog at steering wheel",
                "role": "humor target",
                "source_anchor_id": "a1",
                "viewpoint": "relation_crop",
                "region": {
                    "kind": "bbox",
                    "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
                    "point_xy_norm": [0.45, 0.55],
                    "confidence": "high",
                    "evidence": "dog and wheel are visible",
                },
            }
        ],
        "relations": [
            {
                "subject": "a1",
                "predicate": "positioned_behind",
                "object": "steering wheel",
                "confidence": "medium",
            }
        ],
    }


def test_region_methods_are_registered() -> None:
    assert "hic-compact-json-region" in GUIDED_PROMPT_METHODS
    assert "hic-compact-json-overlay" in GUIDED_PROMPT_METHODS
    assert "hic-compact-json-crop" in GUIDED_PROMPT_METHODS


def test_region_methods_end_with_exact_base_prompt_once() -> None:
    for method in ("hic-compact-json-region", "hic-compact-json-overlay", "hic-compact-json-crop"):
        prompt = build_guided_prompt(
            method=method,
            image_description="",
            visual_facts={},
            humor_viewpoint=_humor_viewpoint(),
            region_annotation=_region_annotation(),
            gold_caption=GOLD,
            base_prompt=DEFAULT_BASE_PROMPT,
        )

        assert prompt.endswith(DEFAULT_BASE_PROMPT)
        assert prompt.count(DEFAULT_BASE_PROMPT) == 1


def test_region_prompt_includes_compact_region_anchors_and_boxes() -> None:
    prompt = build_guided_prompt(
        method="hic-compact-json-region",
        image_description="",
        visual_facts={},
        humor_viewpoint=_humor_viewpoint(),
        region_annotation=_region_annotation(),
        gold_caption=GOLD,
        base_prompt=DEFAULT_BASE_PROMPT,
    )

    assert "<joke_annotations>" in prompt
    assert "<region_annotations>" in prompt
    assert "dog at steering wheel" in prompt
    assert "[0.1,0.2,0.8,0.9]" in prompt
    assert "the dog finally got his license" not in prompt
    assert "the target joke" in prompt


def test_overlay_and_crop_methods_describe_auxiliary_images() -> None:
    overlay = build_guided_prompt(
        method="hic-compact-json-overlay",
        image_description="",
        visual_facts={},
        humor_viewpoint=_humor_viewpoint(),
        region_annotation=_region_annotation(),
        gold_caption=GOLD,
    )
    crop = build_guided_prompt(
        method="hic-compact-json-crop",
        image_description="",
        visual_facts={},
        humor_viewpoint=_humor_viewpoint(),
        region_annotation=_region_annotation(),
        gold_caption=GOLD,
    )

    assert "overlay image" in overlay.lower()
    assert "crop sheet" in crop.lower()


def test_hic_compact_json_prompt_keeps_payload_and_base_prompt_contract() -> None:
    prompt = build_hic_compact_json_prompt(_humor_viewpoint(), gold_caption=GOLD, base_prompt=DEFAULT_BASE_PROMPT)

    assert prompt.endswith(DEFAULT_BASE_PROMPT)
    assert prompt.count(DEFAULT_BASE_PROMPT) == 1
    assert GOLD not in prompt
    assert "the target joke" in prompt

    match = re.search(r"<joke_annotations>(.+)</joke_annotations>", prompt)
    assert match is not None
    assert json.loads(match.group(1)) == {
        "scene": "A dog sits behind a steering wheel.",
        "type": "role_mismatch",
        "target": "the target joke says the target joke, because the dog is framed as a driver.",
        "primary_view": "relation_crop",
        "views": ["relation_crop"],
        "anchors": [
            {
                "label": "dog at steering wheel",
                "evidence": "dog is behind the wheel",
                "role": "sets up the target joke",
            }
        ],
        "external_knowledge": False,
    }


def test_hic_compact_json_prompt_discourages_explanatory_descriptions() -> None:
    prompt = build_hic_compact_json_prompt(_humor_viewpoint(), gold_caption=GOLD, base_prompt=DEFAULT_BASE_PROMPT)

    assert "Use the compact JSON as joke clues, not wording to copy." in prompt
    assert "Write a caption, not an image description." in prompt
    assert "Prefer a punchline or meme-style line over a full sentence explanation." in prompt
    assert "Maximum 12 words." in prompt
    assert "Do not use because, since, which is, creating, visual effect, image, photo, scene, joke, humor, or funny." in prompt
    assert "Do not name the humor type, viewpoint, annotation labels, or JSON fields." in prompt
    assert "Do not use abstract analysis words such as contrast, mismatch, reversal, unexpected, target, anchor, viewpoint, label, role, or scale." in prompt
