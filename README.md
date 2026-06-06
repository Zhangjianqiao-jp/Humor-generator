# Humor Generator

A research project for multimodal humorous caption generation using vision-language models, supervised fine-tuning, LoRA, candidate generation, and reranking.

The project explores how to generate image-grounded humorous captions from OxfordTVG-HIC style data. It is currently used as a research and engineering pipeline for data preparation, model fine-tuning, candidate generation, reranker construction, and evaluation design.

## Overview

The current pipeline includes:

- Data preprocessing for the Oxford humor caption dataset
- SFT-format dataset construction
- Qwen2.5-3B fine-tuning with LoRA
- Candidate caption generation
- Candidate cleaning and diagnostic evaluation
- Reranker construction for candidate selection
- Large-scale generation of caption candidates
- Evaluation metric design for humor quality and image relevance

Future work includes replacing the single large model with a combination of smaller specialized models to improve modularity and efficiency.

## Current Status

- Built the data preprocessing pipeline
- Constructed SFT-format training data
- Completed SFT and LoRA fine-tuning with Qwen2.5-3B
- Generated a large number of candidate captions
- Built the first reranker utilities for candidate selection
- Prepared `v2.5/` for the next reranker build
- Evaluation metrics are currently under construction

## Versions

- `v1/`: first baseline for data preparation, candidate generation, and heuristic ranking.
- `v1.5final/`: final V1.5 clean-prompt Qwen2.5-VL LoRA-SFT snapshot, plus candidate-cleaning, Qwen judging, hard-negative, and reranker utility scripts.
- `v2.5/`: clean reranker construction workspace forked from `v1.5final`. Use this for the next reranker iteration.

The older `v1.5/` folder has been superseded by `v1.5final/`.

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

## Quick Start

```bash
cd v1.5final
python -m pip install -r requirements.txt
python -m pytest tests
```

For the next reranker work:

```bash
cd v2.5
python -m pip install -r requirements.txt
python -m pytest tests
```

## Data And Artifacts

GitHub does not store generated data, model checkpoints, candidate dumps, or logs. These paths are intentionally ignored:

```text
v1.5final/data/processed/
v1.5final/outputs/
v2.5/data/processed/
v2.5/outputs/
```

Rebuild processed files with the scripts in each version, or copy them from the local workspace when continuing experiments. The standalone local workspace at `/home/zhang.jianqiao/projects/v2.5` was prepared with processed data preserved for reranker construction.

## Expected Dataset Layout

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

## Model Paths

Configs default to local model/dataset paths used in this workspace. If you run the project elsewhere, update the relevant paths inside each version directory:

```text
configs/data_preprocess.yaml
configs/lora_sft.yaml
configs/humor_reranker.yaml
```

## Research Direction

This project focuses on multimodal LLM-based humor generation and evaluation. Current research interests include:

- Image-grounded caption generation
- Humor quality evaluation
- Candidate generation and reranking
- Efficient model composition with smaller specialized models
- Multimodal generation evaluation
