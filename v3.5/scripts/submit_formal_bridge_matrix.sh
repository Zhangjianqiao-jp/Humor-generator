#!/bin/sh
# Manual fallback: submit only the first pilot. The monitor serializes the rest.
set -eu
cd "$(dirname "$0")/.."
.venv/bin/python scripts/run_formal_preflight.py

pjsub -N p_l_sft_k2 -x BRIDGE_CONFIG=configs/pilot/learned_sft_kl.yaml,BRIDGE_OUTPUT=outputs/pilot/learned_sft_kl_visualcap_v2 jobs/formal_bridge_train.pjm
echo "Only Learned+KL was submitted. Use formal_pipeline_monitor.py for serialized continuation."
