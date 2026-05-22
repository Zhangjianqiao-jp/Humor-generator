# Humor Generator V1.5

V1.5 is the LoRA-SFT version of the humor generator. It adds a reproducible data preprocessing step and a Qwen2.5-VL LoRA training entry point.

## What Is Included

- `scripts/preprocess_data.py`: cleans OxfordTVG-HIC style CSVs and writes SFT JSONL files.
- `configs/data_preprocess.yaml`: preprocessing configuration with editable fields clearly marked.
- `scripts/train_lora_sft.py`: trains LoRA adapters on the processed SFT data.
- `configs/lora_sft.yaml`: LoRA and training hyperparameters with editable sections clearly marked.
- `tests/test_preprocess_hic.py`: small regression test for preprocessing behavior.

## 1. Install

```bash
cd v1.5
python -m pip install -r requirements.txt
```

## 2. Preprocess Data

```bash
python scripts/preprocess_data.py --config configs/data_preprocess.yaml
```

Outputs:

```text
data/processed/sft_train.jsonl
data/processed/sft_val.jsonl
data/processed/sft_test.jsonl
data/processed/sft_sample_100.jsonl
```

The main places to edit are marked in `configs/data_preprocess.yaml`:

- Dataset columns: `image_id_col`, `caption_col`, `score_col`
- Filtering hyperparameters: `rank_percentile_threshold`, `max_captions_per_image`, caption length limits
- Split hyperparameters: `train_ratio`, `val_ratio`, `test_ratio`, `seed`

## 3. Train LoRA-SFT

```bash
python scripts/train_lora_sft.py --config configs/lora_sft.yaml
```

The main places to edit are marked in `configs/lora_sft.yaml`:

- Base model: `model.model_name`
- LoRA hyperparameters: `rank`, `alpha`, `dropout`, `target_modules`
- Training hyperparameters: `batch_size`, `gradient_accumulation_steps`, `num_epochs`, `learning_rate`
- Memory settings: `gradient_checkpointing`, `bf16`, `fp16`

If training stops and a checkpoint exists, resume with:

```bash
python -m scripts.train_lora_sft --config configs/lora_sft.yaml --resume-from-checkpoint outputs/lora_sft_v1_5/checkpoint-250
```

If no checkpoint directory exists, the run cannot be resumed from optimizer/model state and should be restarted. V1.5 filters missing image paths by default during training and writes reports under `outputs/lora_sft_v1_5/missing_images/`.

## Recommended First Ablation

Keep everything fixed except LoRA rank:

```yaml
rank: 16
rank: 32
rank: 64
```

Then compare validation loss and generated caption quality on the same test split.
