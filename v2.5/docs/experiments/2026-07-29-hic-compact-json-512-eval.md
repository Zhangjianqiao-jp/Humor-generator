---
date: 2026-07-29
project: HIC humorous image captioning
run_id: hic-compact-json-512-vs-base-test200
status: complete
tags:
  - hic-compact-json
  - lora-sft
  - evaluation
  - qwen-vl-3b
---

# HIC Compact JSON 512 Pilot Evaluation

## Research Question

`hic-compact-json` 作为结构化笑点 clue 后，用 512 train / 128 val 的 LoRA SFT 是否能让 `Qwen2.5-VL-3B-Instruct` 在 held-out HIC test caption 生成上变短、少解释、并更接近 gold caption 的幽默点？

## Commit

当前代码主线提交：

```text
3353f1e Add full HIC compact JSON run support
63829e8 Add HIC compact JSON SFT workflow
```

本实验结果产生于这些提交附近；后续复现实验时应优先使用当前分支最新提交，并检查 prompt renderer 是否仍为 style-fixed `hic-compact-json`。

## Dataset

```text
train context: outputs/analysis/hic_humor_viewpoints_sft_train_pilot_512.jsonl
val context: outputs/analysis/hic_humor_viewpoints_sft_val_pilot_128.jsonl
test split: data/processed/sft_test.jsonl
held-out eval subset: first 200 rows
```

## Models

Generator/base：

```text
/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct
```

Teacher/analyzer：

```text
/home/zhang.jianqiao/models/Qwen2.5-VL-7B-Instruct
```

LoRA adapter：

```text
outputs/lora_sft_hic_compact_json_pilot_512/final_lora
outputs/lora_sft_hic_compact_json_pilot_512/best_val_loss
```

## Prompt Renderer

```text
hic-compact-json
```

Style fix requirements:

- compact JSON is only clue, not output format;
- output exactly one short caption;
- avoid explanations and analysis labels;
- final base instruction remains:

```text
Generate one short, natural, image-specific humorous caption for this image. Do not explain.
```

## Generation Parameters

Standard project generation parameters:

```text
temperature = 0.8
top_p = 0.9
max_new_tokens = 48
repetition_penalty = 1.05
candidates_per_image = 8
```

## Outputs

Base prompt with style-fixed `hic-compact-json`:

```text
outputs/generations/hic_compact_json_stylefix_200.jsonl
outputs/evaluations/hic_compact_json_stylefix_200_summary.json
outputs/evaluations/hic_compact_json_stylefix_200_eval.jsonl
```

512-row LoRA pilot:

```text
outputs/generations/hic_compact_json_pilot512_sft_test_200.jsonl
outputs/evaluations/hic_compact_json_pilot512_sft_test_200_summary.json
outputs/evaluations/hic_compact_json_pilot512_sft_test_200_eval.jsonl
```

Training output:

```text
outputs/lora_sft_hic_compact_json_pilot_512/
```

## Metrics

Evaluation threshold:

```text
similarity_threshold = 0.55
image_score = 1 if any candidate has max_text_similarity >= threshold
```

Comparison:

```text
method                    gold_match  candidate_match  avg_max_sim  format_ok  avg_chars
base hic-compact-json        0.0550          0.0181       0.3768     0.9806       59.6
512-row LoRA pilot           0.1200          0.0694       0.4232     0.9963       28.7
```

Style flags:

```text
explains flag:        0.0194 -> 0.0031
generic-pattern flag: 0.0425 -> 0.0169
```

Training:

```text
steps: 64
train_loss: 2.257
eval_loss: 2.1428
eval_ppl: 8.52
best_val_loss: 2.1428 at step 64
```

## Known Caveats

- Gold-caption-derived `hic-compact-json` is an upper-bound context, not deployable image-only context.
- Test subset is only first 200 rows, so final claims need random 1000 or larger held-out evaluation.
- Text similarity to gold caption is useful but incomplete, because humorous captions can have many valid alternatives.
- Refusal-style outputs rose slightly in 512 pilot and must stay tracked.

## Decision

512 pilot is a useful positive signal. Continue with systematic evaluation and preference data construction rather than treating full SFT as the only main path.

Recommended next comparison:

```text
base
base + hic-compact-json prompt
512 LoRA
3000 LoRA if user-provided result is available
full LoRA if eventually completed
```

Recommended next training direction:

```text
real candidate pairs -> reranker / DPO
image-only humor hypotheses -> generator -> reranker
```
