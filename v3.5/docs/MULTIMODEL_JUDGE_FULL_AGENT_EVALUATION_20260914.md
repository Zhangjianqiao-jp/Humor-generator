# Full account-agent evaluation: latent bridge vs pure-text HOMER

## Scope and completion

This report records the completed auxiliary evaluation for the current v3.5
latent-only system. It compares the same anonymous caption pairs under
`latent_bridge` and `text_homer_context_replay` (the pure-text HOMER context
replay condition). The run is complete only because the local finalizer passed
all hard gates:

- 35,250 canonical packets were judged by each of `gpt-5.5`, `gpt-5.6-sol`,
  and `gpt-5.6-terra` (105,750 decisions total);
- every merged JSONL has exactly 35,250 rows in canonical packet order;
- the five Terra mandatory re-audit shards were completed and promoted;
- local merge and finalization returned exit code 0;
- `multimodel_summary.json` has the complete sentinel and a three-way majority
  result.

The authoritative machine-readable result is
[`multimodel_summary.json`](../outputs/caption_judgement/pretrained_bridge_comparison_20260911/multimodel_agent_final/multimodel_summary.json),
and the generated tabular report is
[`multimodel_summary.md`](../outputs/caption_judgement/pretrained_bridge_comparison_20260911/multimodel_agent_final/multimodel_summary.md).

## Protocol

The packet grid is 47 images × 5 trials × 5 candidates × 3 reference groups ×
5 references × 2 conditions = 35,250 pairwise packets. The reference groups
are `1-9`, `200-209`, and `1000-1009`; each image/trial has five candidates.
The evaluator receives only the official HOMER text rubric and an anonymous
`Cartoon description / Caption A / Caption B` pair. Private mapping and
`blind.secret` are read only by the local finalizer. Pass@1, Pass@3, and
Pass@5 are computed with image-level bootstrap (2,000 replicates in this
run), separately for each evaluator and reference group.

Provenance gates passed with:

```text
canonical packet SHA-256: 79f0b66635a6d836c51168628f6578e5f466378f3904e2eecfc868a11c00ad77
official rubric SHA-256:   98ebb56261262dd77112094330b056417ca133e2adf02ab5b6da60afdc31c524
source provenance SHA-256: 44ea9297984ed1040e676a2589706aadc46b89dfa9b5650954a1cd7bec3434b1
evaluators:                gpt-5.5, gpt-5.6-sol, gpt-5.6-terra
transport:                 Codex account-backed auxiliary track; no API key
temperature:               requested 0, not independently exposed by Codex CLI
```

The last line is an explicit limitation: it is not evidence of an exact
temperature-0 API setting, and this track must not be reported as the
canonical `gpt-5-chat-latest` HOMER result.

## Evaluator agreement

| Pair | Agreement | Decisions |
|---|---:|---:|
| gpt-5.5 vs gpt-5.6-sol | 0.865163 | 30,497 / 35,250 |
| gpt-5.5 vs gpt-5.6-terra | 0.735404 | 25,923 / 35,250 |
| gpt-5.6-sol vs gpt-5.6-terra | 0.819461 | 28,886 / 35,250 |

Agreement is adequate for a robustness check but not interchangeable with a
single human gold label. The final analysis therefore keeps evaluator-specific
metrics and the majority vote separate.

## Majority-vote comparison

The values below are `Pass@K(latent_bridge) - Pass@K(text_homer_context_replay)`;
95% intervals are image-cluster bootstrap intervals from the finalizer.

| Reference group | K=1 | K=3 | K=5 |
|---|---:|---:|---:|
| `1-9` | +0.0019 [-0.0054, +0.0095] | +0.0057 [-0.0138, +0.0260] | +0.0102 [-0.0196, +0.0400] |
| `200-209` | -0.0034 [-0.0109, +0.0034] | -0.0071 [-0.0287, +0.0130] | -0.0094 [-0.0400, +0.0221] |
| `1000-1009` | +0.0085 [+0.0002, +0.0175] | +0.0206 [+0.0009, +0.0427] | +0.0289 [-0.0017, +0.0596] |

Thus the latent bridge does not show a uniform, statistically decisive gain over
pure-text HOMER. It is slightly positive for the `1-9` and `1000-1009` groups,
negative for `200-209`, and all K=5 intervals include zero. The K=1/K=3 gain
for `1000-1009` is the only majority result whose interval excludes zero; it
should be treated as a reference-group interaction, not a global claim.

Evaluator-specific results show why the consensus is cautious: gpt-5.5 is
near-tie to slightly text-favouring, gpt-5.6-sol favours latent on these
packets, and gpt-5.6-terra is mixed by reference group. These are descriptive
differences under this account-agent protocol, not proof that latent captions
are absolutely funnier.

## Engineering recovery evidence

One Terra 100-packet request repeatedly returned 99 decisions. The runner now
classifies this as transport truncation, preserves the malformed responses as
`.invalidN`, recursively bisects the range, validates both child ranges, and
atomically writes a parent checkpoint. The controller was also corrected to
retain the Terra gate after promotion moves re-audit files to their official
output names. Regression tests cover both behaviours (16 targeted tests pass).

The finalizer completed after the quota marker was preserved and moved aside;
no packet was filled with a default decision and no valid checkpoint was
deleted. Historical rows that predate the latest runner do not carry optional
`evaluation_transport` fields, so the summary explicitly reports
`codex-exec-account,unspecified`; packet IDs, model IDs, temperature value,
prompt hash, and source-provenance hash still pass strict validation. This
metadata caveat is part of the result and must not be omitted in a paper.

## Reproducibility entry points

- Resume/status without conversation context:
  `scripts/resume_homer_evaluation.sh status --json`
- Run the strict controller: `scripts/resume_homer_evaluation.sh start`
- Mechanism and recovery rules:
  [`HOMER_EVALUATION_RESUMABLE_ZH.md`](HOMER_EVALUATION_RESUMABLE_ZH.md) and
  [`EVALUATION_MECHANISM_ZH.md`](EVALUATION_MECHANISM_ZH.md)
- Implementation commits for adaptive recovery and promotion gate:
  `90749a2` and `250a3ed` on branch `latent`.

## References

1. HOMER, *Humor Generation with Structured Multi-Stage Reasoning*, arXiv:
   [https://arxiv.org/abs/2602.06423](https://arxiv.org/abs/2602.06423), plus the
   [official implementation](https://github.com/Shang-hub/HOMER-Official-Implementation).
2. Hessel et al., *Do Androids Laugh at Electric Sheep?* ACL 2023:
   [https://aclanthology.org/2023.acl-long.41/](https://aclanthology.org/2023.acl-long.41/).

