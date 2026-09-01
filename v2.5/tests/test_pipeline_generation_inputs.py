from __future__ import annotations

import json

import pytest

from scripts.build_captioner_inputs_from_plans import (
    CAPTIONER_INSTRUCTION,
    build_captioner_rows,
    build_compact_viewpoint_rows,
    build_hic_compact_json_rows,
)
from scripts.build_hic_viewpoint_sft import build_rows as build_viewpoint_sft_rows
from scripts.build_blind_caption_comparison import blind_rows
from scripts.generate_lora_sft import load_prompt_override, load_prompt_template, select_unique_image_rows
from scripts.group_gold_captions_for_compact_labels import group_rows
from scripts.prepare_newyorker_full_test import prepare as prepare_full_test
from scripts.repair_compact_viewpoint_json import repair_candidate
from scripts.verify_sft_generations import parse_compact_viewpoint, parse_hic_viewpoint


def test_select_unique_image_rows_preserves_first_row_order() -> None:
    rows = [
        {"image_id": "a", "image": "a.jpg", "caption": "first"},
        {"image_id": "a", "image": "a.jpg", "caption": "second"},
        {"image_id": "b", "image": "b.jpg", "caption": "third"},
    ]

    selected = select_unique_image_rows(rows)

    assert [row["caption"] for row in selected] == ["first", "third"]


def test_load_prompt_override_reads_exact_multiline_file(tmp_path) -> None:
    prompt_file = tmp_path / "planner.txt"
    prompt_file.write_text("first line\n\nsecond line\n", encoding="utf-8")

    assert load_prompt_override(None, prompt_file) == "first line\n\nsecond line"


def test_load_prompt_override_rejects_two_sources(tmp_path) -> None:
    prompt_file = tmp_path / "planner.txt"
    prompt_file.write_text("file prompt", encoding="utf-8")

    with pytest.raises(ValueError, match="only one"):
        load_prompt_override("inline prompt", prompt_file)


def test_load_prompt_template_requires_caption_placeholder(tmp_path) -> None:
    prompt_file = tmp_path / "planner.txt"
    prompt_file.write_text("Gold caption:\n{caption}", encoding="utf-8")

    assert load_prompt_template(prompt_file) == "Gold caption:\n{caption}"

    prompt_file.write_text("No placeholder", encoding="utf-8")
    with pytest.raises(ValueError, match=r"no \{caption\} placeholder"):
        load_prompt_template(prompt_file)


def test_build_captioner_rows_preserves_generated_plan_verbatim() -> None:
    plan = "ANCHOR: a visible object\nCONTRAST: an odd action\nANGLE: dry reversal"
    rows = build_captioner_rows(
        [
            {
                "image": "cartoon.jpg",
                "image_id": "nycc_1",
                "gold_caption": "gold plan",
                "candidates": [plan],
            }
        ]
    )

    assert rows == [
        {
            "image": "cartoon.jpg",
            "image_id": "nycc_1",
            "prompt": f"{CAPTIONER_INSTRUCTION}\n\nHumor plan:\n{plan}",
            "planner_candidate": plan,
            "planner_gold": "gold plan",
            "gold_caption": "gold plan",
        }
    ]


def test_build_captioner_rows_rejects_duplicate_images() -> None:
    row = {"image": "cartoon.jpg", "image_id": "nycc_1", "candidates": ["a plan"]}
    with pytest.raises(ValueError, match="duplicate image_id"):
        build_captioner_rows([row, row])


def test_build_hic_compact_json_rows_restores_historical_renderer() -> None:
    annotation = {
        "literal_image_description": "An elephant stands in a living room.",
        "gold_joke_explanation": "The gold caption treats it as routine.",
        "humor_type": "role_mismatch",
        "humor_point": "The gold caption treats an elephant as an ordinary guest.",
        "visual_anchors": [
            {"id": "a1", "label": "elephant", "role": "The gold caption's unexpected guest", "evidence": "indoors"}
        ],
        "required_viewpoints": ["relation_crop"],
        "primary_viewpoint": "relation_crop",
        "needs_external_knowledge": False,
        "confidence": "high",
        "uncertainty": "",
    }
    rows = build_hic_compact_json_rows(
        [
            {
                "image": "cartoon.jpg",
                "image_id": "nycc_543",
                "gold_caption": "A test caption.",
                "candidates": [json.dumps(annotation)],
            }
        ]
    )

    payload = json.loads(rows[0]["compact_json"])
    assert list(payload) == ["scene", "type", "target", "primary_view", "views", "anchors", "external_knowledge"]
    assert "<joke_annotations>" in rows[0]["prompt"]
    assert rows[0]["prompt"].endswith(
        "Generate one short, natural, image-specific humorous caption for this image. Do not explain."
    )
    assert "gold caption" not in rows[0]["prompt"].lower()


def test_parse_hic_viewpoint_accepts_historical_schema() -> None:
    payload = {
        "literal_image_description": "A person stands beside an elephant.",
        "gold_joke_explanation": "The line treats the elephant as routine.",
        "humor_type": "role_mismatch",
        "humor_point": "An elephant is treated like an ordinary guest.",
        "visual_anchors": [{"id": "a1", "label": "elephant", "role": "intruder", "evidence": "indoors"}],
        "required_viewpoints": ["relation_crop"],
        "primary_viewpoint": "relation_crop",
        "needs_external_knowledge": False,
        "confidence": "high",
        "uncertainty": "",
    }

    assert parse_hic_viewpoint(json.dumps(payload)) == payload
    assert parse_hic_viewpoint(f"```json\n{json.dumps(payload)}\n```") == payload


def test_build_viewpoint_sft_rows_removes_gold_conditioning() -> None:
    payload = {
        "scene": "An elephant stands in a room.",
        "type": "role_mismatch",
        "target": "An elephant is treated as an ordinary guest.",
        "primary_view": "relation_crop",
        "views": ["relation_crop"],
        "anchors": [{"label": "elephant", "evidence": "indoors", "role": "unexpected guest"}],
        "external_knowledge": False,
    }
    teacher_prompt = "Gold caption:\nA guest arrived.\n\nReturn JSON only."
    planner_prompt = "Analyze the image and return compact JSON."
    outputs = build_viewpoint_sft_rows(
        [
            {
                "image": "one.jpg",
                "image_id": "nycc_1",
                "gold_caption": "A guest arrived.",
                "prompt": teacher_prompt,
                "candidates": [json.dumps(payload)],
            }
        ],
        planner_prompt=planner_prompt,
    )

    assert outputs[0]["messages"][0]["content"][1]["text"] == planner_prompt
    assert "A guest arrived" not in json.dumps(outputs[0]["messages"])
    assert json.loads(outputs[0]["messages"][1]["content"][0]["text"]) == payload


def test_compact_viewpoint_renderer_passes_planner_json_to_captioner() -> None:
    payload = {
        "scene": "An elephant stands in a room.",
        "type": "role_mismatch",
        "target": "An elephant is treated as an ordinary guest.",
        "primary_view": "relation_crop",
        "views": ["relation_crop"],
        "anchors": [{"label": "elephant", "evidence": "indoors", "role": "unexpected guest"}],
        "external_knowledge": False,
    }

    outputs = build_compact_viewpoint_rows(
        [{"image": "one.jpg", "image_id": "nycc_1", "candidates": [json.dumps(payload)]}]
    )

    assert json.loads(outputs[0]["compact_json"]) == payload
    assert parse_compact_viewpoint(f"```json\n{json.dumps(payload)}\n```") == payload
    assert "<joke_annotations>" in outputs[0]["prompt"]


def test_repair_compact_viewpoint_only_inserts_missing_anchor_comma() -> None:
    malformed = """{
"scene":"An elephant stands indoors.",
"type":"role_mismatch",
"target":"The elephant is treated as an ordinary guest.",
"primary_view":"relation_crop",
"views":["relation_crop"],
"anchors":[{"label":"elephant","evidence":"standing indoors"
"role":"unexpected guest"}],
"external_knowledge":false
}"""

    repaired, changes = repair_candidate(malformed)

    assert parse_compact_viewpoint(repaired)["anchors"][0]["role"] == "unexpected guest"
    assert changes == ["inserted_missing_evidence_role_comma:1"]


def test_group_gold_captions_uses_every_caption_once() -> None:
    outputs = group_rows(
        [
            {"image": "a.jpg", "image_id": "a", "gold_caption": "first"},
            {"image": "a.jpg", "image_id": "a", "gold_caption": "second"},
            {"image": "a.jpg", "image_id": "a", "gold_caption": "first"},
        ]
    )

    assert outputs[0]["source_rows"] == 3
    assert outputs[0]["caption_count"] == 2
    assert outputs[0]["gold_captions"] == ["first", "second"]
    assert outputs[0]["gold_caption"].endswith("Ranked high-rated captions:\n1. first\n2. second")


def test_prepare_full_test_keeps_pairs_and_unique_images() -> None:
    rows = [
        {"image": "a.jpg", "image_id": "a", "gold_caption": "first"},
        {"image": "a.jpg", "image_id": "a", "gold_caption": "second"},
        {"image": "b.jpg", "image_id": "b", "gold_caption": "third"},
    ]

    pairs, unique = prepare_full_test(rows)

    assert len(pairs) == 3
    assert [row["image_id"] for row in unique] == ["a", "b"]
    assert [row["pair_id"] for row in pairs] == ["a::000000", "a::000001", "b::000002"]
    assert unique[0]["gold_captions"] == ["first", "second"]
    assert unique[0]["reference_count"] == 2


def test_blind_rows_retains_both_systems_without_exposing_names() -> None:
    joint = [{"image_id": "nycc_1", "image": "one.jpg", "candidates": ["j1", "j2"]}]
    direct = [{"image_id": "nycc_1", "image": "one.jpg", "candidates": ["d1", "d2"]}]

    public, key = blind_rows(joint, direct, seed=7)

    assert len(public) == len(key) == 4
    assert {row["caption"] for row in public} == {"j1", "j2", "d1", "d2"}
    assert {row["system_label"] for row in public} == {"A", "B"}
    assert all("joint" not in row and "direct" not in row for row in public)
    assert {row["system"] for row in key} == {"joint", "direct"}
    assert len({row["blind_id"] for row in public}) == 4
