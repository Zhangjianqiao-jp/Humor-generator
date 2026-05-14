# Humor Generator (V1 Baseline)

V1 baseline for humorous image captioning using OxfordTVG-HIC data preparation, candidate generation, and heuristic ranking.  
**V1 does not train the model yet.** It prepares SFT data and an inference/ranking baseline.


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

<<<<<<< codex/create-python-project-structure-for-humor-generator-i5dnt8
你要填的关键项：
- `--image-csv`：图片元数据 CSV 文件路径。
- `--caption-csv`：caption 元数据 CSV 文件路径。
- `--image-id-col`：图片 CSV 里图像 ID 列名。
- `--image-path-col` 或 `--image-url-col`：图片路径/URL 列名（二选一）。
- `--caption-image-id-col`：caption CSV 里关联图像 ID 列名。
- `--caption-col`：caption 文本列名。
- `--score-col`：打分列名。

> 因为 OxfordTVG-HIC 的列名可能和示例不同，建议先跑 `inspect_dataset.py` 看列名再填。

### 3.3 候选生成（mock，不需要模型）
```bash
python scripts/generate_candidates.py \
  --input-jsonl data/processed/sft_test.jsonl \
  --output-jsonl outputs/generations/candidates.jsonl \
  --num-candidates 10 \
  --dry-run true
```

### 3.4 候选生成（真实 Qwen 模型）
```bash
python scripts/generate_candidates.py \
  --input-jsonl data/processed/sft_test.jsonl \
  --output-jsonl outputs/generations/candidates.jsonl \
  --num-candidates 10 \
  --dry-run false \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct
```

如果你要用本地模型目录，把 `--model-name` 改成本地路径：
```bash
--model-name /your/local/model/path/Qwen2.5-VL-7B-Instruct
```

### 3.5 排序输出 Top-5
```bash
python scripts/rank_candidates.py \
  --input-jsonl outputs/generations/candidates.jsonl \
  --output-jsonl outputs/generations/ranked_top5.jsonl
```

---

## 4) Dry-run（无需下载模型）
```bash
python scripts/run_v1_dryrun.py
```
该脚本会：
- 生成 `data/demo/` 的小型合成数据
- 构建 SFT
- 生成 mock candidates
- 进行排序并打印 top-5

---

## 5) 数据布局建议（你当前场景）
建议你最终保持类似结构：
```text
humor-generator/
  hic-data/
    # 你的原始 CSV
    images_metadata.csv
    captions_metadata.csv
    # 你的图片目录（两种都支持）
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

---

## 6) V1 范围声明
V1 当前只做：
- SFT 数据准备
- 候选生成（mock + Qwen加载入口）
- 六维启发式打分与Top-5排序

**不包含**：HCL / DPO / GRPO / IRCoT / GTVH / 动态RAG / 多专家LoRA 训练。
=======
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
>>>>>>> main
