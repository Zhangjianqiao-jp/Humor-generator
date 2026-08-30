from pathlib import Path

from scripts.verify_frozen_artifacts import verify


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_adapters_match_manifest() -> None:
    result = verify(ROOT / "manifests/frozen_7b_adapters.json")
    assert result["status"] == "pass"
    assert set(result["checked"]) == {"planner_sft", "generator_sft"}
