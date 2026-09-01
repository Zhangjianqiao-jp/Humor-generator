from pathlib import Path
import runpy


def test_executable_tree_has_no_v2_runtime_reference() -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "scripts/check_v3_isolation.py"))
    assert namespace["violations"]() == []
