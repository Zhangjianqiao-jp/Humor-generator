from __future__ import annotations

import json
from pathlib import Path

from humor_generator_v35.homer.official_prompts import CAPTION_SYSTEM
from humor_generator_v35.training.public_bridge import (
    PUBLIC_BRIDGE_PROMPT_TRACK,
    load_context_index,
    public_latent_caption_messages,
    public_text_caption_messages,
    target_caption_text,
)
from scripts.cache_pretrained_homer_context import build_context_for_record


class ContextBackend:
    def generate(self, messages, *, temperature, max_new_tokens, seed):
        del temperature, max_new_tokens, seed
        text = " ".join(
            block.get("text", "")
            for message in messages
            for block in (
                message.get("content", [])
                if isinstance(message.get("content"), list)
                else [{"text": message.get("content", "")}]
            )
            if isinstance(block, dict)
        )
        if "two JSON lists" in text:
            return '{"cat":["keyboard","office","deadline"],"keyboard":["office","deadline","audit"]}'
        if "select two most funny" in text:
            return "##Conflict Scripts:\nordinary office work vs. absurd animal bureaucracy"
        if "Two key entities" in text:
            return "[cat, keyboard]"
        raise AssertionError(f"unexpected context prompt: {text}")


class ContextRetriever:
    def retrieve(self, summary, *, description, conflicts):
        del description, conflicts
        assert "cat" in summary
        return {
            "cat": [["cat", "keyboard"], ["keyboard", "office"], ["office", "deadline"]],
            "keyboard": [["keyboard", "office"], ["office", "deadline"], ["deadline", "audit"]],
        }


def _trace() -> dict:
    return {
        "cluster_id": "nycc_530",
        "contest_number": 530,
        "image": "data/external/benchmarks/humor_in_ai/cartoons/source/530.jpg",
        "image_sha256": "image-hash",
        "split": "train",
        "standard_description": "A cat in an office files a tax return.",
        "trace_path": "data/cache/test/states/nycc_530.pt",
        "trace_sha256": "a" * 64,
        "plan": {
            "description": "A cat in an office files a tax return.",
            "conflicts": [
                {"left": "ordinary office work", "right": "absurd animal bureaucracy"},
                {"left": "cat", "right": "worker"},
            ],
            "local_chains": [{"root": "cat", "steps": ["keyboard", "office", "deadline"], "view": "local"}],
            "global_chains": [{"root": "keyboard", "steps": ["office", "deadline", "audit"], "view": "global"}],
        },
        "planner_outputs": {
            "conflict": "1. ordinary office work vs. absurd animal bureaucracy 2. cat vs. worker",
            "global": '{"cat":["keyboard","office","deadline"]}',
            "local": '{"keyboard":["office","deadline","audit"]}',
        },
    }


def _row() -> dict:
    return {
        "cluster_id": "nycc_530",
        "contest_number": 530,
        "dataset": "humor_in_ai",
        "split": "train",
        "image": "data/external/benchmarks/humor_in_ai/cartoons/source/530.jpg",
        "image_sha256": "image-hash",
        "standard_description": "A cat in an office files a tax return.",
    }


def test_current_context_replay_keeps_selection_and_path() -> None:
    context = build_context_for_record(
        ContextBackend(), _row(), _trace(), ContextRetriever(), seed=17
    )
    assert context["context_prompt_track"] == PUBLIC_BRIDGE_PROMPT_TRACK
    assert context["selected_entities"] == ["cat", "keyboard"]
    assert context["selected_paths"]["cat"][0] == "cat"
    assert "The free-association chain of cat is" in context["free_association_text"]
    assert context["call_log"][0]["stage"] == "imaginator.summary"
    assert [item["stage"] for item in context["call_log"]] == [
        "imaginator.summary", "generator.select_conflict", "generator.select_entities"
    ]


def test_public_bridge_uses_official_system_and_replaces_only_plan_block() -> None:
    value = {
        "schema_version": 1,
        "data_version": "homer_pretrained_7b_public_release_362",
        "cluster_id": "nycc_530",
        "contest_number": 530,
        "image": "data/external/benchmarks/humor_in_ai/cartoons/source/530.jpg",
        "image_sha256": "image-hash",
        "standard_description": "A cat in an office files a tax return.",
        "summary_imagination": {"cat": ["keyboard"]},
        "summary_raw": '{"cat":["keyboard"]}',
        "retrieved_imagination": {"cat": [["cat", "keyboard"]]},
        "selected_conflict": "ordinary vs. absurd",
        "selection_raw": {
            "conflict": "##Conflict Scripts:\nordinary vs. absurd",
            "entities": "[cat, keyboard]",
        },
        "selected_entities": ["cat", "keyboard"],
        "selected_paths": {
            "cat": ["cat", "keyboard"],
            "keyboard": ["keyboard", "office"],
        },
        "free_association_text": (
            "The free-association chain of cat is cat -> keyboard\n"
            "The free-association chain of keyboard is keyboard -> office"
        ),
        "trace_path": "data/cache/test/states/nycc_530.pt",
        "trace_sha256": "a" * 64,
        "planner_outputs_sha256": "b" * 64,
        "call_log": [{"stage": "imaginator.summary", "seed": 1}],
        "selection_policy": "official_public_code_selection_plus_seeded_dfs_path",
        "context_prompt_track": PUBLIC_BRIDGE_PROMPT_TRACK,
        "provenance": {"git_commit": "0" * 40},
    }
    path = Path("/tmp/public_bridge_context_test.jsonl")
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    loaded = load_context_index(path)
    context = loaded["nycc_530"]
    text = public_text_caption_messages(context)
    latent = public_latent_caption_messages(context)
    assert text[0]["content"] == CAPTION_SYSTEM
    assert latent[0]["content"] == CAPTION_SYSTEM
    assert "ordinary vs. absurd" in text[1]["content"][0]["text"]
    assert "ordinary vs. absurd" not in latent[1]["content"][0]["text"]
    assert len(text[1]["content"]) == len(latent[1]["content"]) == 2
    assert target_caption_text({"row_id": "r", "caption": "A good caption."}) == "A good caption."
