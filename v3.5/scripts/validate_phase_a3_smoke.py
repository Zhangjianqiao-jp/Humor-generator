#!/usr/bin/env python3
"""Fail closed unless the real two-example Phase A3 GPU smoke is complete."""
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

    if config["loss"].get("semantic_objective") != "channel_balanced_v3":
        raise RuntimeError("smoke config is not Phase A3")
    if config["loss"].get("alignment_teacher") != "receiver_contextual_final_hidden":
        raise RuntimeError("Phase A3 smoke does not use contextual teachers")
    if config["bridge"].get("channel_fusion") != "fixed_equal":
        raise RuntimeError("Phase A3 smoke does not use fixed-equal channel fusion")
    if report.get("status") != "pass" or report.get("scientific_training") is not False:
        raise RuntimeError("GPU smoke did not finish as an engineering-only pass")
    if report.get("policy_trainable_parameters") != 0:
        raise RuntimeError("a frozen 7B policy became trainable")
    if not finite_positive(report.get("bridge_trainable_parameters")):
        raise RuntimeError("bridge has no trainable parameters")
    if not finite_positive(report.get("gradient_norm")) or not finite_positive(report.get("update_norm")):
        raise RuntimeError("bridge gradient/update is absent or non-finite")
    if len(report.get("samples", [])) != 2 or len(set(report.get("clusters", []))) != 2:
        raise RuntimeError("smoke did not exercise two distinct image clusters")

    loss = report.get("last_loss", {})
    contrastive = report.get("contrastive_loss", {})
    for channel in CHANNELS:
        for metric in ("caption_nll", "matched_minus_shuffled_logp"):
            value = loss.get(f"{metric}_{channel}")
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(f"missing/non-finite {metric}_{channel}")
        weight = float(loss[f"mean_channel_weight_{channel}"])
        if abs(weight - 1 / 3) > 1e-4:
            raise RuntimeError(f"fixed-equal channel mass violated for {channel}: {weight}")
        for metric in (f"info_nce_{channel}", f"retrieval_at_1_{channel}"):
            value = contrastive.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(f"missing/non-finite {metric}")
    print(json.dumps({
        "status": "pass",
        "contract": "phase_a3_real_trace_forward_backward_contextual_nce_single_channel_cf",
        "clusters": report["clusters"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
