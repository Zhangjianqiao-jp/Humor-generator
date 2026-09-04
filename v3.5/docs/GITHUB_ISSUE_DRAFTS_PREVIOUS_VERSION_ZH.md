# Previous-version issue drafts

更新时间：2026-09-04

这些是根据旧版本代码、日志和 artifact 整理的 GitHub issue 草稿。它们不把
`pilot_inconclusive` 写成方法失败，也不把没有盲评者的 caption 文件写成质量结果。
由于当前环境没有 GitHub CLI 或 REST token，只能先保存草稿；实际创建 issue 需要在
已认证的 GitHub 环境执行相应命令。

## Issue 1 — A3/A4 outer evaluator protocol mismatch

**Title**

`[v3.5] Use a dedicated outer evaluator for the A4 channel-isolated protocol`

**Body**

The historical `scripts/run_outer_semantic_confirmation.py` evaluates the A3
`channel_balanced_v3` protocol. It does not pass `active_channels=(channel,)` and does not
force `semantic_prompt_include_image=false`. Reusing it for the A4 checkpoint would allow
the unchanged channels or the image to explain the target, so its gap would not identify
channel use.

Evidence:

- `scripts/run_outer_semantic_confirmation.py`
- `configs/pilot/cross_attention_semantic_phase_a4.yaml`
- `outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1/run_manifest.json`

The fix is an independent A4 evaluator with target-only masking and an image-free semantic
prompt. This is a protocol/correctness issue, not evidence that latent communication fails.

## Issue 2 — Donor channel length is a confound

**Title**

`[v3.5] Match counterfactual donors by per-channel token length`

**Body**

The A4 attention normalizes over the donor channel's token positions. If the donor has a
different channel length, the measured gap changes both semantic content and the softmax
denominator. The training and earlier evaluator selected donors without a strict per-channel
length match.

Evidence:

- `src/humor_generator_v35/latent/cross_attention.py`
- `scripts/train_bridge.py::hard_negative_cluster_map`
- `scripts/re_evaluate_failed_semantic_bridges.py::length_matched_channel_donors`

The corrected outer protocol must record length delta, use length-priority donors, and report
gap/length correlation and donor strata.

## Issue 3 — Missing zero-communication control

**Title**

`[v3.5] Add zero-bridge and random-initialization controls to the semantic gate`

**Body**

The previous gate compared later training epochs with the first epoch. That does not show
that communication improves over a frozen receiver with no bridge. A nonzero residual can
also make epoch-to-epoch changes look like a method gain.

Evidence:

- `scripts/check_semantic_training_gate.py`
- `outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1/semantic_gate.json`

The outer report should include zero-bridge matched NLL/log-probability, zero-bridge
counterfactual gap, and a random-init bridge control. This is required before interpreting
any semantic improvement.

## Issue 4 — InfoNCE teacher is not receiver-native

**Title**

`[v3.5] Replace the fixed random projection used by the semantic InfoNCE teacher`

**Body**

The current contextual teacher pools frozen-receiver hidden states and projects them with a
stationary copy of the bridge's initial query weights. This is a fixed auxiliary coordinate,
not a learned receiver-semantic projector. A high retrieval score therefore cannot establish
that the receiver uses task-relevant channel meaning.

Evidence:

- `src/humor_generator_v35/latent/cross_attention.py`
- `src/humor_generator_v35/training/cross_attention_bridge.py`
- `outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1/semantic_gate.json`

Either fit a projector on train-only receiver-contextual states, use target-span pooling, or
report InfoNCE only as an auxiliary identity diagnostic. The causal channel gap remains the
primary mechanism evidence.

## Issue 5 — Legacy caption pilot has invalid outputs and no valid good-rate result

**Title**

`[v3.5] Quarantine pre-A4 caption pilot artifacts from scientific comparison`

**Body**

`outputs/pilot_validation/` was generated under the older visual-caption pilot commit
`cdcfbbde...`, before A4 channel isolation. The set contains 8 condition files, 40 clusters,
and 3 seeds per condition, but no current A4 checkpoint/evaluator provenance and no submitted
blind ratings. Several controls contain empty or malformed outputs, including
`[EMPTY OUTPUT]`, `]`, `] ]`, and `addCriterion`.

Observed in the archived files:

- `token_embedding`: 18/120 rows marked `empty_output=true`;
- `statebridge`: 29/120 rows marked `empty_output=true`;
- `typed_quantized`: 28/120 rows marked `empty_output=true`.

These files may document engineering failures, but they cannot provide an absolute
`good-caption rate` or support a latent-vs-text claim. New generation must wait for the A4
outer gate, use a new output directory, and include checkpoint/evaluator/model/prompt hashes.

## References

- HOMER: https://arxiv.org/html/2602.06423
- Interlat: https://aclanthology.org/2026.acl-long.1248/
- BLIP-2: https://proceedings.mlr.press/v202/li23q
- CPC/InfoNCE: https://arxiv.org/abs/1807.03748
- Statistical testing in NLP: https://aclanthology.org/P18-1128/
