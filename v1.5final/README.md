# Humor Generator V1.5 Final

V1.5 Final is the clean-prompt Qwen2.5-VL LoRA-SFT snapshot for humorous image caption generation. It also includes the candidate-cleaning, Qwen judging, hard-negative, and CLIP-style humor reranker utilities developed during the V1.5 cycle.

The normalized SFT prompt is:

```text
Generate one short, natural, image-specific humorous caption for this image. Do not explain.
```

When `data.normalize_prompt: true`, the training loader ignores old prompts stored in JSONL and rebuilds each example as image plus normalized prompt to gold caption.

## Repository Snapshot

This GitHub snapshot intentionally excludes generated data and model artifacts:

```text
data/processed/
outputs/
__pycache__/
*.pyc
*.pt
*.pth
*.bin
*.safetensors
```

Rebuild data with the preprocessing scripts, or copy local processed files back into `data/processed/` when resuming experiments. Keep model adapters and checkpoints under `outputs/` locally.

## Included Code

- `scripts/preprocess_data.py`: clean OxfordTVG-HIC style CSVs and write SFT JSONL files.
- `scripts/train_lora_sft.py`: train/debug the clean-prompt Qwen2.5-VL LoRA-SFT adapter.
- `scripts/generate_lora_sft.py`: generate multiple candidates from a trained LoRA adapter.
- `scripts/clean_sft_candidates.py`: remove duplicate, looping, generic, or malformed candidate captions.
- `scripts/evaluate_sft_candidates.py`: run lightweight text and format diagnostics.
- `scripts/judge_sft_candidates_qwen.py`: optionally score candidates with a Qwen2.5-VL judge.
- `scripts/build_reranker_score_pools.py`: build strong/weak positive and negative pools for reranker training.
- `scripts/train_humor_reranker.py`: train the lightweight humor reranker.
- `scripts/rerank_candidates.py`: score generated candidates and emit reranked outputs.
- `configs/lora_sft.yaml`: model, data, LoRA training, generation, and evaluation settings.
- `configs/humor_reranker.yaml`: reranker model, pair data, training, and output settings.

## Install

```bash
cd v1.5final
python -m pip install -r requirements.txt
```

## Preprocess SFT Data

```bash
python -m scripts.preprocess_data --config configs/data_preprocess.yaml
```

Expected local outputs:

```text
data/processed/sft_train.jsonl
data/processed/sft_val.jsonl
data/processed/sft_test.jsonl
data/processed/sft_sample_100.jsonl
```

## Debug LoRA-SFT

```bash
python scripts/train_lora_sft.py --config configs/lora_sft.yaml --debug-data
python scripts/train_lora_sft.py --config configs/lora_sft.yaml --debug-collator --num-debug-samples 3
python scripts/train_lora_sft.py --config configs/lora_sft.yaml --debug-one-step
```

A healthy collator debug should show nonzero supervised target tokens and the supervised text should be the gold caption only.

## Train LoRA-SFT

```bash
python scripts/train_lora_sft.py --config configs/lora_sft.yaml
```

Resume from a Trainer checkpoint only when you need optimizer/scheduler/RNG state:

```bash
python scripts/train_lora_sft.py \
  --config configs/lora_sft.yaml \
  --resume_from_checkpoint outputs/lora_sft_v1_5/checkpoint-10000
```

Use `checkpoint-*` directories for resume. Adapter-only directories such as `latest/`, `best_val_loss/`, and `final_lora/` are for inference/export.

## Generate And Clean Candidates

```bash
python -m scripts.generate_lora_sft \
  --config configs/lora_sft.yaml \
  --adapter outputs/lora_sft_v1_5/best_val_loss \
  --input-jsonl data/processed/sft_val.jsonl \
  --output-jsonl outputs/generations/v1_5_candidates.jsonl \
  --num-candidates 10
```

```bash
python scripts/clean_sft_candidates.py \
  --input-jsonl outputs/generations/v1_5_candidates.jsonl \
  --output-jsonl outputs/generations/v1_5_candidates_clean.jsonl \
  --summary-json outputs/evaluations/v1_5_candidates_clean_summary.json \
  --group-by-image \
  --max-chars 100 \
  --max-words 22 \
  --target-candidates 10 \
  --min-candidates 3 \
  --drop-generic
```

## Evaluate Candidates

```bash
python scripts/evaluate_sft_candidates.py \
  --input-jsonl outputs/generations/v1_5_candidates_clean.jsonl \
  --summary-json outputs/evaluations/v1_5_candidate_clean_eval_summary.json
```

Optional Qwen judge:

```bash
python scripts/judge_sft_candidates_qwen.py \
  --config configs/lora_sft.yaml \
  --input-jsonl outputs/generations/v1_5_candidates_clean.jsonl \
  --output-jsonl outputs/evaluations/v1_5_qwen_judge.jsonl \
  --limit 50 \
  --max-candidates 10
```

## Reranker Utilities

Build strict score pools from judged/cleaned data, then train the reranker:

```bash
python scripts/build_reranker_score_pools.py --config configs/humor_reranker.yaml
python scripts/train_humor_reranker.py --config configs/humor_reranker.yaml --stage strong
```

Rerank generated candidates with a trained checkpoint:

```bash
python -m scripts.rerank_candidates \
  --checkpoint outputs/humor_reranker_v1/mixed/checkpoint_best.pt \
  --input-jsonl outputs/generations/v1_5_candidates_clean.jsonl \
  --output-jsonl outputs/reranked/v1_5_candidates_reranked.jsonl \
  --max-candidates 10 \
  --top-k 10
```

## Literal Hard Negatives

Use base Qwen2.5-VL captions as image-grounded, non-humorous hard negatives:

```bash
python scripts/generate_literal_captions_qwen.py \
  --config configs/lora_sft.yaml \
  --positive-jsonl data/processed/reranker_score_pools_strict/strong_positive.jsonl \
  --output-jsonl data/processed/reranker_hard_negatives/literal_captions_qwen.jsonl \
  --num-captions 5 \
  --resume
```

```bash
python scripts/build_literal_negative_pairs.py \
  --literal-jsonl data/processed/reranker_hard_negatives/literal_captions_qwen.jsonl \
  --positive-jsonl data/processed/reranker_score_pools_strict/strong_positive.jsonl \
  --output-jsonl data/processed/reranker_hard_negatives/literal_pairs.jsonl \
  --neg-score-offset 1.0 \
  --loss-weight 0.6
```

Then mix original strong/weak pairs with literal hard-negative pairs:

```bash
python scripts/mix_reranker_pairs.py \
  --strong-jsonl data/processed/reranker_score_pools_strict/strong_pairs.jsonl \
  --weak-jsonl data/processed/reranker_score_pools_strict/weak_pairs.jsonl \
  --literal-jsonl data/processed/reranker_hard_negatives/literal_pairs.jsonl \
  --output-jsonl data/processed/reranker_hard_negatives/mixed_strong_weak_literal_pairs.jsonl \
  --target-size 300000 \
  --strong-ratio 0.60 \
  --weak-ratio 0.20 \
  --literal-ratio 0.20
```
