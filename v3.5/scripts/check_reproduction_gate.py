#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humor_generator_v35.reproduction import audit_reproduction


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/homer_text_reproduction.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    result = audit_reproduction(config, args.config.parent.parent)
    print(
        json.dumps(
            {"ready": result.ready, "failures": result.failures, "warnings": result.warnings},
            indent=2,
        )
    )
    result.require()


if __name__ == "__main__":
    main()
