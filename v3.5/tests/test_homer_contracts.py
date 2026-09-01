from __future__ import annotations

import pytest

from humor_generator_v35.homer.contracts import SchemaError, parse_associations, parse_conflicts, validate_plan
from humor_generator_v35.homer.repair import assert_lossless_repair, validator_feedback_messages


def test_strict_homer_plan() -> None:
    plan = validate_plan(
        "A serious board meeting features impossibly large steaming coffee cups on the table.",
        "1. Professional restraint vs. absurd excess. 2. Ordinary coffee break vs. industrial caffeine.",
        '{"oversized cups":["coffee","milk","cow"]}',
        '{"meeting":["agenda","deadline","panic"]}',
    )
    assert len(plan.conflicts) == 2
    assert plan.local_chains[0].path == ("oversized cups", "coffee", "milk", "cow")


def test_conflicts_require_two_explicit_oppositions() -> None:
    with pytest.raises(SchemaError):
        parse_conflicts("1. This is merely unusual.")


def test_association_requires_valid_json_and_exact_three_steps() -> None:
    with pytest.raises(SchemaError):
        parse_associations("[cat, keyboard]", view="local")
    with pytest.raises(SchemaError):
        parse_associations('{"cat":["keyboard","office"]}', view="local")


def test_association_accepts_only_an_exact_json_fence_wrapper() -> None:
    parsed = parse_associations(
        '```json\n{"cat": ["pet", "fur", "lint"]}\n```', view="local"
    )
    assert parsed[0].path == ("cat", "pet", "fur", "lint")
    with pytest.raises(SchemaError):
        parse_associations(
            'Here is JSON: {"cat": ["pet", "fur", "lint"]}', view="local"
        )


def test_qwen_typed_record_schema_preserves_every_three_step_chain() -> None:
    parsed = parse_associations(
        '[{"entity":"cat","associations":['
        '["keyboard","office","deadline"],'
        '["fur","lint","dryer"]]}]',
        view="global",
    )
    assert [item.path for item in parsed] == [
        ("cat", "keyboard", "office", "deadline"),
        ("cat", "fur", "lint", "dryer"),
    ]
    with pytest.raises(SchemaError):
        parse_associations(
            '[{"entity":"cat","associations":["keyboard","office"]}]',
            view="global",
        )


def test_lossless_qwen_mapping_list_wrapper_is_accepted() -> None:
    parsed = parse_associations(
        '[{"cat":["keyboard","office","deadline"]},'
        '{"dog":["leash","park","mud"]}]',
        view="global",
    )
    assert [item.path for item in parsed] == [
        ("cat", "keyboard", "office", "deadline"),
        ("dog", "leash", "park", "mud"),
    ]


def test_contiguous_edge_chain_is_losslessly_reconstructed() -> None:
    parsed = parse_associations(
        '[{"entity":"cat","associations":['
        '["cat","keyboard"],["keyboard","office"],["office","deadline"]]}]',
        view="global",
    )
    assert parsed[0].path == ("cat", "keyboard", "office", "deadline")
    with pytest.raises(SchemaError, match="not contiguous"):
        parse_associations(
            '[{"entity":"cat","associations":['
            '["cat","keyboard"],["park","office"],["office","deadline"]]}]',
            view="global",
        )


def test_lossless_association_repair_changes_schema_only() -> None:
    invalid = '[{"entity":"cat","imaginations":["keyboard","office","deadline"]}]'
    repaired = '{"cat":["keyboard","office","deadline"]}'
    assert_lossless_repair(invalid, repaired, channel="local")
    with pytest.raises(ValueError, match="changed association"):
        assert_lossless_repair(
            invalid, '{"cat":["keyboard","office","vacation"]}', channel="local"
        )


def test_lossless_conflict_repair_only_inserts_opposition_structure() -> None:
    invalid = "1. Dinosaurs playing modern instruments 2. Calm concert during an incoming meteor"
    repaired = (
        '[{"left":"Dinosaurs","right":"playing modern instruments"},'
        '{"left":"Calm concert","right":"during an incoming meteor"}]'
    )
    assert_lossless_repair(invalid, repaired, channel="conflict")
    with pytest.raises(ValueError, match="added or paraphrased"):
        assert_lossless_repair(
            invalid,
            '[{"left":"Dinosaurs","right":"playing jazz instruments"},'
            '{"left":"Calm concert","right":"during an incoming meteor"}]',
            channel="conflict",
        )


def test_repair_turn_preserves_original_homer_messages() -> None:
    original = [{"role": "system", "content": [{"type": "text", "text": "HOMER"}]}]
    repaired = validator_feedback_messages(
        original,
        invalid_output='{"cat":[]}',
        validation_error="needs three steps",
        channel="local",
    )
    assert repaired[:1] == original
    assert repaired[-2]["role"] == "assistant"
    assert repaired[-1]["role"] == "user"
