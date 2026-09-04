#!/usr/bin/env python3
"""Validate the A4 channel-isolated real-trace engineering smoke."""
from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path

import yaml


CHANNELS = ("conflict", "local", "global")


def finite_positive(value: object) -> bool:
    number = float(value)
    return math.isfinite(number) and number > 0


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = json.loads(args.report.read_text())
    if config["loss"].get("semantic_objective") != "channel_isolated_v4":
        raise RuntimeError("smoke config is not Phase A4")
    if config["loss"].get("channel_visibility") != "target_only":
        raise RuntimeError("A4 smoke must activate only the target channel")
    if config["training"].get("semantic_prompt_include_image") is not False:
        raise RuntimeError("A4 semantic recovery must disable the image shortcut")
    if float(config["loss"].get("counterfactual_reconstruction", 0.0)) <= 0:
        raise RuntimeError("A4 smoke must exercise donor reconstruction")
    if report.get("status") != "pass" or report.get("scientific_training") is not False:
        raise RuntimeError("A4 GPU smoke did not finish as an engineering-only pass")
    if report.get("policy_trainable_parameters") != 0:
        raise RuntimeError("a frozen 7B policy became trainable")
    if not finite_positive(report.get("bridge_trainable_parameters")):
        raise RuntimeError("bridge has no trainable parameters")
    if not finite_positive(report.get("gradient_norm")) or not finite_positive(report.get("update_norm")):
        raise RuntimeError("bridge gradient/update is absent or non-finite")
    if len(report.get("samples", [])) != 2 or len(set(report.get("clusters", []))) != 2:
        raise RuntimeError("A4 smoke did not exercise two distinct image clusters")

    for sample in report["samples"]:
        loss = sample.get("loss", {})
        for channel in CHANNELS:
            prefix = f"{channel}"
            for metric in ("caption_nll", "matched_minus_shuffled_logp", "counterfactual_reconstruction_nll"):
                value = loss.get(f"{metric}_{prefix}")
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(f"missing/non-finite {metric}_{channel}")
            for visible in CHANNELS:
                value = loss.get(f"mean_channel_weight_{visible}_{prefix}")
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(f"missing/non-finite channel weight {visible}/{channel}")
                expected = 1.0 if visible == channel else 0.0
                if abs(float(value) - expected) > 1e-4:
                    raise RuntimeError(
                        f"A4 channel isolation violated for target={channel}, "
                        f"visible={visible}: {value}"
                    )
    print(json.dumps({
        "status": "pass",
        "contract": "phase_a4_real_trace_channel_isolated_no_image_donor_reconstruction",
        "clusters": report["clusters"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
