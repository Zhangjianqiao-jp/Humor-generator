# Humor Generator V2.5

V2.5 is the reranker construction workspace forked from `v1.5final`. It keeps the V1.5 LoRA-SFT generator available, but the main goal is to build, train, and evaluate a stronger humor reranker over generated image-caption candidates.

## Workspace Policy

Two copies were prepared:

- GitHub repo copy: `Humor-generator/v2.5/`, source/config/test files only.
- Local working copy: `/home/zhang.jianqiao/projects/v2.5/`, same source files plus `data/processed/` preserved for reranker work.

Generated artifacts stay local and are ignored by Git:

```text
data/processed/
outputs/
__pycache__/
*.pt
*.pth
*.bin
*.safetensors
```

## What To Build Here

- A cleaner pair construction flow for strong, weak, and literal hard-negative reranker data.
- A stronger reranker training loop using `configs/humor_reranker.yaml` as the baseline.
- Reranking and evaluation scripts that select the best caption from V1.5 LoRA candidates.
- Tests around pair loading, missing-image handling, scoring order, and output schema once the reranker changes start.

## Install

```bash
cd /home/zhang.jianqiao/projects/v2.5
python -m pip install -r requirements.txt
```

If you are working from the GitHub checkout instead, use:

```bash
cd v2.5
python -m pip install -r requirements.txt
```

## Verify The Clean Copy

```bash
python -m pytest tests
```

The current tests cover preprocessing path behavior. Add reranker tests as soon as the V2.5 training/data changes begin.

## Reranker Data Already Available Locally

The standalone local copy keeps these processed files:

```text
data/processed/reranker_score_pools/
data/processed/reranker_score_pools_strict/
data/processed/reranker_hard_negatives/
data/processed/sft_train.jsonl
data/processed/sft_val.jsonl
data/processed/sft_test.jsonl
```

The GitHub copy excludes them. To recreate them from a clean checkout, run preprocessing and the score-pool builders after restoring the dataset paths in `configs/data_preprocess.yaml` and `configs/humor_reranker.yaml`.

## Baseline Reranker Training

Start with the strict strong pair file:

```bash
python scripts/train_humor_reranker.py \
  --config configs/humor_reranker.yaml \
  --stage strong \
  --pair-jsonl data/processed/reranker_score_pools_strict/strong_pairs.jsonl \
  --output-dir outputs/humor_reranker_v2_5/strong
```

Then continue with mixed strong/weak/literal data when ready:

```bash
python scripts/train_humor_reranker.py \
  --config configs/humor_reranker.yaml \
  --stage strong \
  --pair-jsonl data/processed/reranker_hard_negatives/mixed_strong_weak_literal_pairs.jsonl \
  --init-checkpoint outputs/humor_reranker_v2_5/strong/checkpoint_best.pt \
  --learning-rate 2.0e-5 \
  --num-epochs 1 \
  --batch-size 128 \
  --eval-steps 500 \
  --save-steps 1000 \
  --output-dir outputs/humor_reranker_v2_5/mixed_literal
```

## Rerank V1.5 Candidates

```bash
python -m scripts.rerank_candidates \
  --checkpoint outputs/humor_reranker_v2_5/mixed_literal/checkpoint_best.pt \
  --input-jsonl outputs/generations/v1_5_candidates_clean.jsonl \
  --output-jsonl outputs/reranked/v2_5_candidates_reranked.jsonl \
  --max-candidates 10 \
  --top-k 10
```


## VLM-Guided Structured Humor Generation

V2.5 now supports a prompt-only structured humor guidance pipeline. The current
research path is:

```text
image + Qwen2.5-VL-7B structured humor extraction at temperature 0
-> rendered guidance prompt + same image
-> base Qwen2.5-VL-3B caption generation
```

The default generator config uses the base model only:

```text
/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct
```

`adapter_dir` is `null` in `configs/vlm_guided_generation.yaml`. Do not use an
old LoRA adapter unless the experiment explicitly requires one and the adapter
path is passed deliberately.

### Extract Structured Humor Context

The extractor sends the image and a constrained JSON prompt to
`Qwen2.5-VL-7B-Instruct`. For structured humor extraction, decoding is greedy
(`structured_humor_temperature: 0.0`).

```bash
/home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/extract_vlm_humor_context.py \
  --config configs/vlm_guided_generation.yaml \
  --input-jsonl data/processed/sft_test.jsonl \
  --output-jsonl outputs/analysis/vlm_structured_humor_100.jsonl \
  --mode both \
  --structured-prompt-version structured-v1 \
  --limit 100 \
  --overwrite
```

Available extraction modes:

```text
visual-facts       old conservative visual-facts JSON only
structured-humor   structured humor JSON only
both               visual facts + structured humor JSON
```

Available structured prompt versions:

```text
structured-v1          zero-shot constrained schema
structured-v1-fewshot  same schema plus text-only formatting examples
```

### Generate Guided Candidates

Supported prompt renderers:

```text
description-only   conservative description only
prompt-method      old natural-language visual facts
feature-method     old tagged visual_facts block
structured-brief   compact structured anchors + mechanism type; current safest default
structured-nl      fuller natural-language structured humor block
structured-json    compact structured_humor JSON block
```

Run the current default structured renderer:

```bash
/home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/generate_guided_lora_candidates.py \
  --config configs/vlm_guided_generation.yaml \
  --method structured-brief \
  --context-jsonl outputs/analysis/vlm_structured_humor_100.jsonl \
  --output-jsonl outputs/generations/vlm_guided_structured_brief_100.jsonl \
  --review-html outputs/reviews/vlm_guided_structured_brief_100.html \
  --num-candidates 8 \
  --limit 100 \
  --overwrite
```

For prompt ablation, generate each renderer with the same context file, seed,
sampling parameters, image set, and candidate count:

```bash
for method in description-only prompt-method feature-method structured-brief structured-nl structured-json; do
  /home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/generate_guided_lora_candidates.py \
    --config configs/vlm_guided_generation.yaml \
    --method "$method" \
    --context-jsonl outputs/analysis/vlm_structured_humor_100.jsonl \
    --output-jsonl "outputs/generations/vlm_guided_${method//-/_}_100.jsonl" \
    --review-html "outputs/reviews/vlm_guided_${method//-/_}_100.html" \
    --num-candidates 8 \
    --limit 100 \
    --overwrite
done
```

The generated HTML files show the image, extracted visual facts, structured
humor fields, final prompt, and candidates for manual review.

### Analyze HIC Humor Viewpoints

Use this script to classify HIC gold-caption humor points and estimate the
minimum visual viewpoint needed for each joke target. The analyzer uses
`Qwen2.5-VL-7B-Instruct`, greedy decoding (`temperature=0.0`), and the image
together with the dataset gold caption.

The current prompt version is `gold-caption-minimal-viewpoint-v2`. It asks the
model to choose the smallest sufficient viewpoint and to avoid `full_image`
unless the whole composition is truly necessary.

Run a clean 1000-pair analysis:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/analyze_hic_humor_viewpoints.py \
  --input /home/zhang.jianqiao/datasets/hic-data \
  --output-jsonl outputs/analysis/hic_humor_viewpoints_pairs_1000_minview.jsonl \
  --summary-json outputs/analysis/hic_humor_viewpoint_summary_pairs_1000_minview.json \
  --report-md outputs/analysis/hic_humor_viewpoint_report_pairs_1000_minview.md \
  --limit 1000 \
  --no-dedupe-image \
  --overwrite
```

If the SSH session drops or the process stops, resume without `--overwrite`.
The script skips successful rows already present in the output JSONL.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/analyze_hic_humor_viewpoints.py \
  --input /home/zhang.jianqiao/datasets/hic-data \
  --output-jsonl outputs/analysis/hic_humor_viewpoints_pairs_1000_minview.jsonl \
  --summary-json outputs/analysis/hic_humor_viewpoint_summary_pairs_1000_minview.json \
  --report-md outputs/analysis/hic_humor_viewpoint_report_pairs_1000_minview.md \
  --limit 1000 \
  --no-dedupe-image
```

The default image bounds are `min_pixels=256*28*28` and
`max_pixels=1024*28*28`, which reduces OOM risk from unusually large images.
For long runs, prefer `nohup` or `tmux`, and avoid running this while another
large VLLM process occupies most GPU memory.


### HIC Viewpoint Prompt Ablation

The HIC viewpoint file can contain failed OOM rows from earlier runs. Prepare a
clean, balanced subset before generation:

```bash
/home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/prepare_hic_viewpoint_ablation.py \
  --input-jsonl outputs/analysis/hic_humor_viewpoints_pairs_1000_minview.jsonl \
  --output-jsonl outputs/analysis/hic_viewpoint_ablation_120.jsonl \
  --summary-json outputs/analysis/hic_viewpoint_ablation_120_summary.json \
  --report-md outputs/analysis/hic_viewpoint_ablation_120.md \
  --limit 120 \
  --min-confidence medium \
  --group-field humor_type
```

Supported HIC annotation renderers:

```text
hic-humor-point       literal scene + gold-derived humor target
hic-viewpoint-tags    humor type + minimal viewpoint tags only
hic-anchor-viewpoint  literal scene + anchors + viewpoint tags + humor target
hic-compact-json      compact JSON version of anchors/viewpoints/target
```

Run the prompt ablation with the base 3B generator. Use the same subset, seed,
sampling parameters, image set, and candidate count for every method:

```bash
for method in plain hic-humor-point hic-viewpoint-tags hic-anchor-viewpoint hic-compact-json; do
  env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/generate_guided_lora_candidates.py \
    --config configs/vlm_guided_generation.yaml \
    --method "$method" \
    --input-jsonl outputs/analysis/hic_viewpoint_ablation_120.jsonl \
    --context-jsonl outputs/analysis/hic_viewpoint_ablation_120.jsonl \
    --output-jsonl "outputs/generations/hic_${method//-/_}_120.jsonl" \
    --review-html "outputs/reviews/hic_${method//-/_}_120.html" \
    --num-candidates 8 \
    --seed 250704 \
    --wait-gpu-free-mb 18000 \
    --wait-gpu-stable-checks 3 \
    --wait-gpu-check-seconds 60 \
    --overwrite \
    > "outputs/generations/hic_${method//-/_}_120.log" 2>&1
done
```

For SSH-safe execution, run that loop inside `tmux`, or wrap it in a small shell
script and start the script with `nohup`.

After generation, judge methods against the gold-caption humor target:

```bash
nohup env PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/judge_hic_guidance_methods_gold.py \
  --generation-jsonl outputs/generations/hic_plain_120.jsonl \
  --generation-jsonl outputs/generations/hic_hic_humor_point_120.jsonl \
  --generation-jsonl outputs/generations/hic_hic_viewpoint_tags_120.jsonl \
  --generation-jsonl outputs/generations/hic_hic_anchor_viewpoint_120.jsonl \
  --generation-jsonl outputs/generations/hic_hic_compact_json_120.jsonl \
  --output-jsonl outputs/evaluations/hic_guidance_gold_judgments_120.jsonl \
  --summary-json outputs/evaluations/hic_guidance_gold_summary_120.json \
  --report-md outputs/evaluations/hic_guidance_gold_report_120.md \
  --wait-gpu-free-mb 22000 \
  --wait-gpu-stable-checks 3 \
  --wait-gpu-check-seconds 60 \
  --overwrite \
  > outputs/evaluations/hic_guidance_gold_judge_120.log 2>&1 &
```

Treat HIC annotations as gold-caption-derived analysis, not as a deployable
image-only extractor. A renderer only becomes a serious candidate if it beats
plain generation under this gold-reference judge and then still helps when the
teacher extracts humor targets without seeing the gold caption.

### Smoke Test Notes

Small wiring tests were run on June 24, 2026:

```text
outputs/analysis/structured_humor_smoke.jsonl
outputs/generations/structured_nl_smoke.jsonl
outputs/generations/structured_json_smoke.jsonl
outputs/generations/structured_brief_smoke.jsonl
outputs/reviews/structured_brief_smoke.html
```

Observed result:

```text
input rows: 3
unique images processed: 2 (the first two test rows share bokete_100043)
structured parse errors: 0
structured non-none mechanisms: 2
structured useful cues: 2
type_counts: {"role_mismatch": 2}
```

The extraction format is stable in this smoke test, so few-shot is not needed
just to force valid JSON. The content is not yet strong enough to call the
prompt optimal: `structured-nl` and `structured-json` tended to make the 3B
model repeat or explain the teacher cue, while `structured-brief` reduced direct
cue copying but still produced some unsupported details. Treat this as a signal
to run prompt-only ablation on 20-100 images before any fine-tuning.

Few-shot should be tested only if the zero-shot extractor shows schema drift or
semantic instability on a larger sample. Fine-tuning should wait until one
prompt-only renderer clearly beats Plain/Base in blind or manual review.

## VLM-Feature LoRA-SFT

Do not start a new LoRA training run until prompt-only guidance passes a small
ablation. If a structured renderer wins, create train and validation context
files first:

```bash
/home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/extract_vlm_humor_context.py \
  --config configs/vlm_guided_generation.yaml \
  --input-jsonl data/processed/sft_train.jsonl \
  --output-jsonl outputs/analysis/vlm_structured_humor_train.jsonl \
  --mode both

/home/zhang.jianqiao/miniconda3/envs/humor/bin/python scripts/extract_vlm_humor_context.py \
  --config configs/vlm_guided_generation.yaml \
  --input-jsonl data/processed/sft_val.jsonl \
  --output-jsonl outputs/analysis/vlm_structured_humor_val.jsonl \
  --mode both
```

Then update the SFT config deliberately and train a fresh adapter only after the
prompt-only gate is convincing.

## Next Implementation Targets

1. Add `src/reranker/` modules if V2.5 moves beyond the current script-level reranker.
2. Add tests for pair JSONL parsing, pair weighting, missing image skip behavior, checkpoint loading, and reranked output ordering.
3. Keep `outputs/` for local checkpoints only; do not commit trained weights to GitHub.
