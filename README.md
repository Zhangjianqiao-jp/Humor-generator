# Humor Generator (V1 Baseline)

V1 baseline for humorous image captioning using OxfordTVG-HIC data preparation, candidate generation, and heuristic ranking.  
**V1 does not train the model yet.** It prepares SFT data and an inference/ranking baseline.

---

## 1) 你需要准备/下载的内容（必须）

### A. Python 环境
- Python 3.10+
- 安装依赖：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### B. 数据集（你已下载）
你说数据集目录名是 `hic-data`，后续放在项目根目录即可：
```text
humor-generator/
  hic-data/
    ...CSV 和图片等文件...
```

> 代码已兼容两种命名：`hic-data` 和 `hic_data`（优先 `hic-data`）。

### C. （可选）Qwen2.5-VL 模型（真实推理才需要）
- 模型名：`Qwen/Qwen2.5-VL-7B-Instruct`
- 若走真实推理，需安装并可加载：
  - `transformers`（建议较新版本）
  - `qwen-vl-utils`
  - `torch`, `accelerate`

你可以二选一：
1. 用 HF 名称：`Qwen/Qwen2.5-VL-7B-Instruct`
2. 用本地模型目录路径：`/path/to/your/local/Qwen2.5-VL-7B-Instruct`

---

## 2) V1 Pipeline（文本图）
1. Inspect OxfordTVG-HIC files
2. Build SFT JSONL from high-score captions
3. Split by `image_id` (train/val/test)
4. Generate N caption candidates per image (mock or Qwen2.5-VL)
5. Score candidates on 6 dimensions (IR/HU/SP/RA/CR/SA)
6. Rank and output Top-5 per image

---

## 3) 路径填写说明（重点）

下面是你最常需要填写/确认的路径参数。

### 3.1 数据检查脚本
```bash
python scripts/inspect_dataset.py --hic-root ./
```
- `--hic-root`：**项目根目录**（因为你会把 `hic-data` 放在根目录）。
- 脚本会自动找：
  - `<hic-root>/hic-data`
  - `<hic-root>/hic_data`

### 3.2 构建 SFT 数据
你需要明确两张 CSV 路径（示例路径请按你的真实文件改）：
```bash
python scripts/build_sft_data.py \
  --image-csv hic-data/images_metadata.csv \
  --caption-csv hic-data/captions_metadata.csv \
  --output-dir data/processed \
  --image-id-col image_id \
  --image-path-col image_path \
  --caption-image-id-col image_id \
  --caption-col caption \
  --score-col score
```

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
