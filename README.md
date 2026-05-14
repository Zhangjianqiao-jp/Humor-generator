# Humor Generator (V1 Baseline)

V1 baseline for humorous image captioning using OxfordTVG-HIC data preparation, candidate generation, and heuristic ranking.  
**V1 does not train the model yet.** It prepares SFT data and an inference/ranking baseline.

## Pipeline (text diagram)
1. Inspect OxfordTVG-HIC files
2. Build SFT JSONL from high-score captions
3. Split by `image_id` (train/val/test)
4. Generate N caption candidates per image (mock or Qwen2.5-VL)
5. Score candidates on 6 dimensions (IR/HU/SP/RA/CR/SA)
6. Rank and output Top-5 per image

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected dataset layout
```text
<hic-root>/
  hic_data/
    *.csv
  images/
    ...image files...
```

## Inspect dataset
```bash
python scripts/inspect_dataset.py --hic-root /path/to/oxford-hic
```

## Build SFT data
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

## Run dry-run (no Qwen download needed)
```bash
python scripts/run_v1_dryrun.py
```

## Generate candidates with mock generator
```bash
python scripts/generate_candidates.py --dry-run true
```

## Run real Qwen2.5-VL inference later
```bash
python scripts/generate_candidates.py \
  --dry-run false \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct
```
If loading fails, install/upgrade `transformers` and `qwen-vl-utils`, and ensure model weights are available.

## Rank candidates
```bash
python scripts/rank_candidates.py
```
