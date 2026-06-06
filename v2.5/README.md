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

## Next Implementation Targets

1. Add `src/reranker/` modules if V2.5 moves beyond the current script-level reranker.
2. Add tests for pair JSONL parsing, pair weighting, missing image skip behavior, checkpoint loading, and reranked output ordering.
3. Keep `outputs/` for local checkpoints only; do not commit trained weights to GitHub.
