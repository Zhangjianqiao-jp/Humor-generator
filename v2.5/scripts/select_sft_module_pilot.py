#!/usr/bin/env python3
"""Select the lowest-validation-loss SFT module pilot and build a full config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Candidate must use NAME=CONFIG.yaml syntax")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Candidate must use NAME=CONFIG.yaml syntax")
    return name.strip(), Path(raw_path)


def lora_parameter_count(config: dict[str, Any]) -> int:
    # Qwen2.5-VL-7B language-backbone dimensions from the resolved model config.
    hidden, intermediate, layers, kv_hidden = 3584, 18944, 28, 512
    shapes = {
        "q_proj": (hidden, hidden),
        "k_proj": (hidden, kv_hidden),
        "v_proj": (hidden, kv_hidden),
        "o_proj": (hidden, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    lora = config["model"]["lora"]
    rank = int(lora["rank"])
    return layers * sum(rank * (shapes[name][0] + shapes[name][1]) for name in lora["target_modules"])


def select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No completed module pilots were provided")
    return min(rows, key=lambda row: (float(row["eval_loss"]), str(row["name"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, type=parse_candidate)
    parser.add_argument("--full-template", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    invariant: tuple[Any, ...] | None = None
    for name, config_path in args.candidate:
        config = load_yaml(config_path)
        output_dir = Path(config["output"]["output_dir"])
        metrics_path = output_dir / "final_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Pilot {name} has no final metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "eval_loss" not in metrics:
            raise ValueError(f"Pilot {name} final metrics have no eval_loss: {metrics_path}")
        current_invariant = (
            config["model"]["model_name"],
            config["data"]["train_path"],
            config["data"]["val_path"],
            config["training"]["num_epochs"],
            config["training"]["learning_rate"],
            config["training"]["seed"],
        )
        if invariant is None:
            invariant = current_invariant
        elif current_invariant != invariant:
            raise ValueError(f"Pilot {name} does not share the controlled training setup")
        rows.append(
            {
                "name": name,
                "config": str(config_path),
                "eval_loss": float(metrics["eval_loss"]),
                "target_modules": list(config["model"]["lora"]["target_modules"]),
                "rank": int(config["model"]["lora"]["rank"]),
                "alpha": int(config["model"]["lora"]["alpha"]),
                "estimated_trainable_lora_parameters": lora_parameter_count(config),
            }
        )

    winner = select(rows)
    full_config = load_yaml(args.full_template)
    full_config["model"]["lora"]["target_modules"] = winner["target_modules"]
    full_config["model"]["lora"]["rank"] = winner["rank"]
    full_config["model"]["lora"]["alpha"] = winner["alpha"]
    full_config.setdefault("experiment", {})["selected_module_pilot"] = winner["name"]
    full_config["experiment"]["module_pilot_report"] = str(args.report)

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        yaml.safe_dump(full_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    report = {
        "selection_metric": "final held-out eval_loss (lower is better)",
        "controlled_variables": [
            "base model",
            "train/validation data",
            "epochs",
            "learning rate",
            "seed",
        ],
        "candidates": rows,
        "selected": winner,
        "output_config": str(args.output_config),
        "scope_note": "Low-cost Go/No-Go pilot; not proof of globally optimal module placement.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
