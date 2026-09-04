#!/usr/bin/env python3
"""Validate the A5 repair-candidate engineering smoke.

The smoke is intentionally not a scientific result.  It verifies that the
role-specific bridge, receiver-native target alignment and A4 causal masks
are all exercised without unfreezing either 7B policy.
"""
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
    if config["loss"].get("semantic_objective") != "channel_isolated_v5":
        raise RuntimeError("smoke config is not Phase A5")
    if config["bridge"].get("projection_mode") != "per_channel":
        raise RuntimeError("A5 smoke must use per-channel K/V/O projections")
    if config["loss"].get("channel_visibility") != "target_only":
        raise RuntimeError("A5 smoke must activate only the target channel")
    if config["training"].get("semantic_prompt_include_image") is not False:
        raise RuntimeError("A5 semantic recovery must disable the image shortcut")
    if float(config["loss"].get("semantic_target_alignment", 0.0)) <= 0:
        raise RuntimeError("A5 smoke must exercise target-span alignment")
    if report.get("status") != "pass" or report.get("scientific_training") is not False:
        raise RuntimeError("A5 GPU smoke did not finish as an engineering-only pass")
    if report.get("policy_trainable_parameters") != 0:
        raise RuntimeError("a frozen 7B policy became trainable")
    if not finite_positive(report.get("bridge_trainable_parameters")):
        raise RuntimeError("bridge has no trainable parameters")
    if not finite_positive(report.get("gradient_norm")) or not finite_positive(report.get("update_norm")):
        raise RuntimeError("bridge gradient/update is absent or non-finite")
    if len(report.get("samples", [])) != 2 or len(set(report.get("clusters", []))) != 2:
        raise RuntimeError("A5 smoke did not exercise two distinct image clusters")
    for sample in report["samples"]:
        loss = sample.get("loss", {})
        for channel in CHANNELS:
            for metric in (
                "caption_nll", "matched_minus_shuffled_logp",
                "counterfactual_reconstruction_nll", "semantic_target_alignment",
            ):
                value = loss.get(f"{metric}_{channel}")
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(f"missing/non-finite {metric}_{channel}")
            for visible in CHANNELS:
                value = loss.get(f"mean_channel_weight_{visible}_{channel}")
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(f"missing/non-finite channel weight {visible}/{channel}")
                expected = 1.0 if visible == channel else 0.0
                if abs(float(value) - expected) > 1e-4:
                    raise RuntimeError(
                        f"A5 channel isolation violated for target={channel}, "
                        f"visible={visible}: {value}"
                    )
    print(json.dumps({
        "status": "pass",
        "contract": "phase_a5_role_specific_receiver_native_target_alignment",
        "clusters": report["clusters"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
