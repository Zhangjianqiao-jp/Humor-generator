#!/bin/sh
# Submit only after the immutable Planner trace gate passes.
set -eu
cd "$(dirname "$0")/.."
.venv/bin/python scripts/check_trace_completion.py

pjsub -N br_l_base -x BRIDGE_CONFIG=configs/formal/learned_base.yaml,BRIDGE_OUTPUT=outputs/formal/learned_base jobs/formal_bridge_train.pjm
pjsub -N br_t_base -x BRIDGE_CONFIG=configs/formal/typed_base.yaml,BRIDGE_OUTPUT=outputs/formal/typed_base jobs/formal_bridge_train.pjm
pjsub -N br_l_sft -x BRIDGE_CONFIG=configs/formal/learned_sft.yaml,BRIDGE_OUTPUT=outputs/formal/learned_sft jobs/formal_bridge_train.pjm
pjsub -N br_t_sft -x BRIDGE_CONFIG=configs/formal/typed_sft.yaml,BRIDGE_OUTPUT=outputs/formal/typed_sft jobs/formal_bridge_train.pjm
