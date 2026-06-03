# Humor Generator

A research project for **multimodal humorous caption generation** using vision-language models, supervised fine-tuning, LoRA, candidate generation, and reranking.

The project explores how to generate image-grounded humorous captions from the Oxford humor caption dataset. It is currently used as a research and engineering pipeline for data preparation, model fine-tuning, candidate generation, and evaluation design.

## Overview

The current pipeline includes:

- Data preprocessing for the Oxford humor caption dataset
- SFT-format dataset construction
- Qwen2.5-3B fine-tuning with LoRA
- Candidate caption generation
- Reranker construction for candidate selection
- Large-scale generation of caption candidates
- Evaluation metric design for humor quality and image relevance

Future work includes replacing the single large model with a combination of smaller specialized models to improve modularity and efficiency.

## Current Status

- Built the data preprocessing pipeline
- Constructed SFT-format training data
- Completed SFT and LoRA fine-tuning with Qwen2.5-3B
- Generated a large number of candidate captions
- Built a reranker for candidate selection
- Evaluation metrics are currently under construction

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Qwen2.5-3B
- LoRA
- Supervised Fine-Tuning (SFT)
- pandas
- JSONL
- Candidate generation
- Reranking

## Expected Dataset Layout

```text
<hic-root>/
  hic_data/
    *.csv
  images/
    ...image files...
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inspect Dataset

```bash
python scripts/inspect_dataset.py --hic-root /path/to/oxford-hic
```

## Build SFT Data

```bash
python scripts/build_sft_data.py \
  --image-csv data/raw/images.csv \
  --caption-csv data/raw/captions.csv \
  --image-id-col image_id \
  --image-path-col image_path \
  --caption-image-id-col image_id \
  --caption-col caption \
  --score-col score
```

Key arguments:

- `--image-csv`: path to the image metadata CSV file
- `--caption-csv`: path to the caption metadata CSV file
- `--image-id-col`: image ID column in the image CSV
- `--image-path-col` or `--image-url-col`: image path or URL column
- `--caption-image-id-col`: image ID column in the caption CSV
- `--caption-col`: caption text column
- `--score-col`: score column

Because the column names in the Oxford dataset may vary, inspect the dataset first before running the full preprocessing pipeline.

## Generate Candidates

### Dry-run Mode

```bash
python scripts/generate_candidates.py \
  --input-jsonl data/processed/sft_test.jsonl \
  --output-jsonl outputs/generations/candidates.jsonl \
  --num-candidates 10 \
  --dry-run true
```

### Qwen Inference Mode

```bash
python scripts/generate_candidates.py \
  --input-jsonl data/processed/sft_test.jsonl \
  --output-jsonl outputs/generations/candidates.jsonl \
  --num-candidates 10 \
  --dry-run false \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct
```

For a local model directory, replace `--model-name` with the local path:

```bash
--model-name /your/local/model/path/Qwen2.5-VL-7B-Instruct
```

## Rank Candidates

```bash
python scripts/rank_candidates.py \
  --input-jsonl outputs/generations/candidates.jsonl \
  --output-jsonl outputs/generations/ranked_top5.jsonl
```

## Dry-run Pipeline

```bash
python scripts/run_v1_dryrun.py
```

This script creates a small synthetic dataset, builds SFT-format data, generates mock candidates, ranks them, and prints the top candidates.

## Suggested Project Structure

```text
humor-generator/
  hic-data/
    images_metadata.csv
    captions_metadata.csv
    images/
      *.jpg
      *.png

  data/
    processed/
      sft_train.jsonl
      sft_val.jsonl
      sft_test.jsonl
      sft_sample_100.jsonl

  outputs/
    generations/
      candidates.jsonl
      ranked_top5.jsonl
```

## Research Direction

This project focuses on multimodal LLM-based humor generation and evaluation. Current research interests include:

- Image-grounded caption generation
- Humor quality evaluation
- Candidate generation and reranking
- Efficient model composition with smaller specialized models
- Multimodal generation evaluation
