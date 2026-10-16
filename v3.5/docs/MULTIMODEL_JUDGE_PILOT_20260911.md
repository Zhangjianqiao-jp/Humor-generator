# Multi-model judge pilot (2026-09-11)

## Scope

This is a small inter-rater consistency pilot on 120 public, HMAC-blinded HOMER packets. It is not the canonical HOMER result and it does not replace the pending 35,250-packet external evaluation. The three agents saw the same packet file and were forbidden from opening the private mapping or inferring the system condition.

- Sample: `/tmp/v35_homer_multimodel_sample120.jsonl`
- Sample SHA-256: `93a1315d446be10412996a113fbc36f279fdde556ae146194d0f0ea2a6d227c7`
- Packets per evaluator: 120
- Allowed decision values: `A` or `B`
- Protocol prompt: the official HOMER-style binary humor comparison prompt in each public packet

## Evaluators and integrity checks

The current collaboration environment exposes `gpt-5.5`, `gpt-5.6-sol`, and `gpt-5.6-terra`; it does not expose a `gpt-5.4` model. The third evaluator is therefore explicitly **gpt-5.6-terra**, not gpt-5.4.

| evaluator | file | rows | A | B | packet/order/hash gate |
|---|---|---:|---:|---:|---|
| gpt-5.5 | `/tmp/v35_judgments_gpt55.jsonl` | 120 | 3 | 117 | pass |
| gpt-5.6-sol | `/tmp/v35_judgments_gpt56sol.jsonl` | 120 | 4 | 116 | pass |
| gpt-5.6-terra | `/tmp/v35_judgments_gpt56terra.jsonl` | 120 | 2 | 118 | pass |

Each file has unique packet IDs in the exact sample order, only `A/B` decisions, and the expected sample hash. No private mapping or condition identifier was present in the decision files.

## Agreement

- All three evaluators agreed on 115/120 packets (95.83%).
- Pairwise raw agreement: 117/120 (97.50%) for gpt-5.5 vs gpt-5.6-sol; 117/120 (97.50%) for gpt-5.5 vs gpt-5.6-terra; 116/120 (96.67%) for gpt-5.6-sol vs gpt-5.6-terra.
- Majority vote: A=2, B=118.

The low number of A decisions is a property of this small packet sample and should not be interpreted as a system win rate without the full packet-to-condition aggregation and confidence intervals.

## Important limitations

1. The collaboration interface does not expose an independently verifiable sampling-temperature control. The files record `temperature_protocol: requested-0-not-exposed`; consequently this pilot is a model-consistency check, not a formal temperature-0 API evaluation.
2. It covers 120/35,250 packets only. The official primary evaluator remains the pinned `gpt-5-chat-latest` runner, which requires an external API key and has not yet produced judgments.
3. The requested gpt-5.4 comparison is not completed because no callable gpt-5.4 endpoint is available in this environment. Do not label the gpt-5.6-terra results as gpt-5.4.
4. The pilot must not be merged into the canonical HOMER table; if expanded, store it as a separate auxiliary multi-model track with its own provenance.

## Full-run command (after local credentials are configured)

Set the key in the server shell only, then run one output file per actual API
model identifier. The runner is resumable; it fails closed if the key is absent,
the model is unavailable, or a packet response is not exactly `A`/`B`.

```bash
P=outputs/caption_judgement/pretrained_bridge_comparison_20260911/homer_primary/homer_primary_prompts.jsonl
V=outputs/caption_judgement/pretrained_bridge_comparison_20260911/homer_primary/provenance.json
O=outputs/caption_judgement/pretrained_bridge_comparison_20260911/multimodel_full
mkdir -p "$O"

for M in gpt-5.5 gpt-5.6-sol gpt-5.4; do
  S=$(printf '%s' "$M" | tr -c 'A-Za-z0-9_.-' '_')
  .venv/bin/python scripts/run_homer_auxiliary_openai.py \
    --packets "$P" --provenance "$V" --output "$O/judgments_${S}.jsonl" \
    --model-id "$M" --max-workers 4 --delay-seconds 0.25 --max-retries 5
done
```

Only after all requested files reach 35,250 rows should the fail-closed
finalizer be run:

```bash
.venv/bin/python scripts/finalize_homer_auxiliary_models.py \
  --private-mapping outputs/caption_judgement/pretrained_bridge_comparison_20260911/homer_primary/homer_primary_private_mapping.jsonl \
  --provenance "$V" \
  --judgments gpt-5.5="$O/judgments_gpt-5.5.jsonl" \
  --judgments gpt-5.6-sol="$O/judgments_gpt-5.6-sol.jsonl" \
  --judgments gpt-5.4="$O/judgments_gpt-5.4.jsonl" \
  --output-dir "$O/final"
```

The `gpt-5.4` line is intentionally not substituted automatically. If that
model ID is unavailable, the run must stop and the result must be reported as
missing rather than relabeling another model.

## References

- Hessel et al., “Do Androids Laugh at Electric Sheep? Humor ‘Understanding’ Benchmarks from The New Yorker Caption Contest,” ACL 2023: <https://aclanthology.org/2023.acl-long.41/>.
- HOMER official implementation: <https://github.com/Shang-hub/HOMER-Official-Implementation>.
