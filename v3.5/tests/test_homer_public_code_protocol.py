from __future__ import annotations

from pathlib import Path

from humor_generator_v35.homer.official_prompts import (
    CAPTION_SYSTEM,
    DESCRIPTION_USER,
    GLOBAL_IMAGINATION_SYSTEM,
    LOCAL_IMAGINATION_SYSTEM,
    SUMMARY_SYSTEM,
    caption_messages,
    description_messages,
)
from humor_generator_v35.homer.omega import balanced_grid, parse as parse_omega
from humor_generator_v35.homer.public_code_pipeline import HomerPublicCodePipeline, _json_object
from scripts.check_pretrained_route import check
from scripts.verify_homer_evaluation_assets import verify


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate(self, messages, *, temperature, max_new_tokens, seed):
        del temperature, max_new_tokens
        text = " ".join(
            part
            for message in messages
            for part in (
                [message.get("content")]
                if isinstance(message.get("content"), str)
                else [block.get("text", "") for block in (message.get("content") or []) if isinstance(block, dict)]
            )
            if isinstance(part, str)
        )
        if "canny and accurate description" in text:
            stage, output = "description", "##Vivid Description:\nA cat in an office files a tax return."
        elif "Script Opposition" in text:
            stage, output = "conflict", "ordinary office work vs. absurd animal bureaucracy"
        elif "cartoon description" in text and "Output JSON" in text:
            stage, output = "local", '{"cat":["keyboard","office","deadline"]}'
        elif "a cartoon" in text and "Output JSON" in text:
            stage, output = "global", '{"cat":["keyboard","office","deadline"]}'
        elif "two JSON lists" in text:
            stage, output = "summary", '{"cat":["keyboard","office","deadline"],"keyboard":["office","deadline","audit"]}'
        elif "select two most funny conflict" in text:
            stage, output = "selected-conflict", "##Conflict Scripts:\nordinary office work vs. absurd animal bureaucracy"
        elif "Two key entities" in text:
            stage, output = "selected-entity", "[cat, keyboard]"
        else:
            stage, output = "caption", "##Caption:\nThe cat has an audit trail.\n##Explanation:\nThe office joke is literal."
        self.calls.append((stage, seed))
        return output


def test_official_prompt_constants_are_distinct_from_historical_adapter() -> None:
    assert "canny and accurate description" in DESCRIPTION_USER
    assert "For example:" in GLOBAL_IMAGINATION_SYSTEM
    assert "For example:" in LOCAL_IMAGINATION_SYSTEM
    assert "Remove any duplicate" in SUMMARY_SYSTEM
    assert "##Explanation" in CAPTION_SYSTEM
    assert description_messages("/tmp/cartoon.jpg")[0]["role"] == "user"
    assert caption_messages("d", "c", "The free-association chain")[0]["content"] == CAPTION_SYSTEM
    assert isinstance(caption_messages("d", "c", "The free-association chain")[1]["content"], list)
    assert len(caption_messages("d", "c", "The free-association chain")[1]["content"]) == 2
    assert caption_messages("d", "c", "The free-association chain")[1]["content"][0]["text"].startswith("Description:\n")
    assert caption_messages("d", "c", "The free-association chain")[1]["content"][1]["text"].startswith("Free-association chains:\n")


def test_public_code_json_object_accepts_truncated_opening_fence() -> None:
    assert _json_object('```json\n{"giraffe":["long neck"]}', stage="summary") == {
        "giraffe": ["long neck"]
    }


def test_public_code_pipeline_runs_all_stages_with_standard_description() -> None:
    backend = FakeBackend()
    pipeline = HomerPublicCodePipeline(
        backend,
        retriever=lambda summary, description, conflicts: summary,
        require_retrieval=True,
        successors=3,
    )
    result = pipeline.run(
        image="/tmp/cartoon.jpg",
        standard_description="A cat in an office files a tax return.",
        seed=17,
    )
    # Standard-description replay skips only the online description request:
    # conflict + global + local + summary + two selections + caption = 7.
    assert result.request_count == 7
    assert [stage for stage, _ in backend.calls] == [
        "conflict", "global", "local", "summary",
        "selected-conflict", "selected-entity", "caption",
    ]
    assert result.caption == "The cat has an audit trail."
    assert result.explanation == "The office joke is literal."
    assert result.selected_paths["cat"] == ("cat", "keyboard", "office", "deadline")


def test_public_code_pipeline_online_description_has_eight_raw_requests() -> None:
    backend = FakeBackend()
    pipeline = HomerPublicCodePipeline(
        backend,
        retriever=lambda summary, description, conflicts: summary,
        require_retrieval=True,
    )
    result = pipeline.run(image="/tmp/cartoon.jpg", seed=19)
    assert result.request_count == 8
    assert result.call_log[0]["stage"] == "extractor.description"
    assert result.call_log[-1]["stage"] == "generator.caption"


def test_pretrained_route_has_no_policy_adapters() -> None:
    report = check(Path(__file__).parents[1] / "configs/homer_public_code_pretrained_7b.yaml")
    assert report["status"] == "pass"
    assert report["data_gate"] == "ready"
    # The adapter-free Planner traces and the post-trace HOMER context/bridge
    # view are now sealed as the current 362-contest public-release route.
    # These remain independent gates so a partial cache cannot silently pass.
    assert report["trace_gate"] == "ready"
    assert report["context_gate"] == "ready"
    assert report["bridge_data_gate"] == "ready"
    assert report["population_manifest"]["allowlist_contest_count"] == 362


def test_public_release_population_is_sealed_and_canonical_365_is_not_claimed() -> None:
    from scripts.verify_homer_public_release_362 import verify

    report = verify(Path(__file__).parents[1] / "manifests/homer_population_public_release_362.json")
    assert report["status"] == "pass"
    assert report["contests"] == 362
    assert report["source_caption_rows"] == 1086
    assert report["canonical_365_verified"] is False


def test_public_release_pins_the_full_huggingface_revision() -> None:
    import json

    manifest = json.loads(
        (Path(__file__).parents[1] / "manifests/homer_population_public_release_362.json").read_text()
    )
    allowlist = json.loads(
        (Path(__file__).parents[1] / "manifests/homer_public_release_362_allowlist.json").read_text()
    )
    expected = "1cd70477b6a99a473690a25a2fed359f75184c64"
    assert manifest["source"]["public_dataset_revision"] == expected
    assert allowlist["source_revision"] == expected


def test_hia_and_homer_evaluation_tracks_are_explicitly_distinct() -> None:
    import yaml

    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/homer_public_code_pretrained_7b.yaml").read_text()
    )
    contract = config["evaluation"]["official_evaluator_contract"]
    hia = contract["hia_original_benchmark"]
    assert hia["available_contests_declared"] == 358
    assert hia["evaluator_holdout_contests"] == 91
    assert hia["candidates_per_image"] == 10
    assert hia["reference_groups"] == ["#1-10", "#200-209", "#1000-1009", "median"]
    assert config["evaluation"]["exact_comparable_track"]["candidates_per_image"] == 5


def test_pinned_evaluator_assets_and_model_contract_are_verified() -> None:
    report = verify(Path(__file__).parents[1] / "manifests/homer_official_assets.json")
    assert report["status"] == "pass"
    assert report["model_contract"]["homer_primary_humor_evaluator"] == "GPT-5"
    assert report["model_contract"]["homer_primary_humor_evaluator_model_id"] == "gpt-5-chat-latest"


def test_project_omega_extension_is_explicit_and_not_default() -> None:
    import yaml

    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/homer_public_code_pretrained_7b.yaml").read_text()
    )
    omega = config["protocol"]["omega"]
    assert omega["enabled"] is True
    assert omega["default_enabled"] is False
    assert len(balanced_grid()) == 36
    spec = parse_omega("setup_punchline|pun_wordplay")
    assert "Narrative strategy: setup_punchline" in spec.prompt_text()
    assert "Linguistic style: pun_wordplay" in spec.prompt_text()


def test_omega_parser_rejects_ambiguous_values() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_omega("setup_punchline|pun_wordplay|extra")
