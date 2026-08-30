from __future__ import annotations

import pytest

from humor_generator_v3.homer.contracts import SchemaError, parse_associations, parse_conflicts, validate_plan


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
