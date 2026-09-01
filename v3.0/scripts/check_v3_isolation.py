#!/usr/bin/env python3
"""Fail if executable v3 code refers to the archived v2.5 tree."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("src", "scripts", "jobs", "configs", "tests")
TEXT_SUFFIXES = {".py", ".sh", ".pjm", ".yaml", ".yml", ".toml"}
FORBIDDEN = ("v2.5", "../v2", "/v2.5/")
SAFETY_CHECKERS = {"check_v3_isolation.py", "check_environment.py"}


def violations() -> list[str]:
    found: list[str] = []
    for directory in SCANNED:
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            if path.name in SAFETY_CHECKERS:
                continue
            text = path.read_text(errors="replace")
            for marker in FORBIDDEN:
                if marker in text:
                    found.append(f"{path.relative_to(ROOT)}: forbidden marker {marker!r}")
    return found


def main() -> None:
    found = violations()
    if found:
        raise SystemExit("v3 isolation failed:\n" + "\n".join(found))
    print("v3 isolation: pass")


if __name__ == "__main__":
    main()
