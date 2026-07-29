---
date: 2026-07-27
project: HIC humorous image captioning
tags:
  - humor-captioning
  - multimodal
  - qwen-vl
  - lora-sft
  - experiment-log
---

# HIC Humor Captioning 项目复盘与下一步

## 当前一句话结论

当前最有价值的路线不是盲目扩大 SFT，而是把 `hic-compact-json` 确认为一个强的 **gold-caption-derived 上限方法**，然后把工作重心转向：

1. 让 full context 生成真正跑起来，或者先暂停全量等待；
2. 把 512/3000 adapter 做系统评估；
3. 构造候选 caption 的偏好数据；
4. 训练 reranker 或 DPO/偏好优化模型；
5. 另开一条 image-only viewpoint extractor，使方法能在没有 gold caption 的测试场景中使用。

## 项目目标

原始目标：使用 Qwen-VL 模型从图片中提取结构化幽默点，并把这些结构化信息作为 prompt / image 输入生成幽默 caption。

后续目标逐步变成：

- 找到最优 prompt；
- 判断是否需要 few-shot 或微调；
- 用 HIC 数据集的 gold caption 判断生成结果是否好笑；
- 设计图片内幽默点标注方案；
- 分类图片幽默类型与 viewpoint；
- 选择最有效的结构化笑点格式；
- 用 `hic-compact-json` 训练 caption 生成模型；
- 判断是否需要进一步做全量 SFT、DPO、reranker 或 image-only extractor。

## 关键模型与数据

- 数据集：`/home/zhang.jianqiao/datasets/hic-data`
- 训练/验证/测试 JSONL：
  - `data/processed/sft_train.jsonl`: 204,481 rows
  - `data/processed/sft_val.jsonl`: 11,365 rows
  - `data/processed/sft_test.jsonl`: 11,438 rows
- 主要 caption generator：
  - `/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct`
- teacher / analyzer：
  - `/home/zhang.jianqiao/models/Qwen2.5-VL-7B-Instruct`
- 最终保留的 base prompt：

```text
Generate one short, natural, image-specific humorous caption for this image. Do not explain.
```

## 已经做过的主要阶段

### 1. 结构化幽默点提取

最初测试了 Qwen7B 作为 teacher，从图片中提取结构化幽默点。后来发现单纯 image-only 提取的幽默点不稳定，所以改成使用 HIC 数据集的 gold caption 辅助判断。

这一步的核心变化：

- 不是让模型凭空猜哪里好笑；
- 而是输入 `image + gold caption`；
- 输出 humor type、humor point、visual anchors、required viewpoints、primary viewpoint 等结构化字段。

代表脚本：

- `scripts/analyze_hic_humor_viewpoints.py`

### 2. Viewpoint taxonomy

对 HIC 图片进行大范围分类后，最后收敛到一组固定 viewpoints：

```text
face_expression_crop
relation_crop
context_scene_view
text_region_crop
object_crop
full_image
pose_action_view
scale_reference_crop
```

这些 viewpoint 的作用是判断“理解这个 gold caption 的笑点最小需要看图片的哪个部分”。

### 3. Prompt ablation

比较过多种 HIC annotation renderer：

```text
plain
hic-humor-point
hic-viewpoint-tags
hic-anchor-viewpoint
hic-compact-json
```

最后选择 `hic-compact-json`，原因是它比自然语言解释更短、更可控，也更不容易让 generator 直接复述 teacher 的分析。

### 4. Prompt style fix

早期 `hic-compact-json` 会导致生成 caption 变得像解释说明，例如出现：

- because
- visual effect
- this image
- humor
- joke
- contrast
- mismatch

后来 prompt 改成严格 caption 风格：

- compact JSON 只是 joke clues；
- 输出 exactly one caption；
- maximum 12 words；
- 不要解释；
- 不要输出分析标签；
- final base prompt 必须原样位于末尾。

相关代码：

- `src/analysis/guided_prompting.py`
- `tests/test_hic_region_guided_prompting.py`

### 5. Base prompt vs style-fixed prompt

在 `sft_test` 前 200 条上测试，style fix 后明显减少解释腔。

base current strict prompt 200 test：

```text
rows: 200
candidates: 1600
gold_match_rate: 0.055
candidate_text_match_rate: 0.018125
avg_max_text_similarity_per_image: 0.3768
format_ok_rate: 0.980625
avg_chars: 59.60
explains: 0.019375
generic_pattern: 0.0425
```

旧 prompt 对比：

```text
gold_match_rate: 0.035
avg_max_text_similarity: 0.3587
format_ok: 0.96
avg_chars: 76.52
explains: 0.035625
generic: 0.065
```

结论：style fix 是有效的。

### 6. 32/8 smoke training

先做了 32 train / 8 val 的 wiring pilot。

结果：

```text
train_loss: 2.504
eval_loss: 2.9267
eval_ppl: 18.67
```

这一步主要证明 pipeline 能跑，不代表最终效果。

### 7. 512/128 pilot training

512 train / 128 val 的 `hic-compact-json` LoRA pilot 已完成。

输出：

```text
outputs/lora_sft_hic_compact_json_pilot_512/final_lora
outputs/lora_sft_hic_compact_json_pilot_512/best_val_loss
```

训练结果：

```text
steps: 64
train_loss: 2.257
eval_loss: 2.1428
eval_ppl: 8.52
best_val_loss: 2.1428 at step 64
```

### 8. 512 adapter held-out 200 evaluation

使用 512-row LoRA adapter 在 `sft_test` 前 200 条上生成 8 candidates/image，并与 base `hic-compact-json` prompt 对比。

```text
method                    gold_match  candidate_match  avg_max_sim  format_ok  avg_chars
base hic-compact-json        0.0550          0.0181       0.3768     0.9806       59.6
512-row LoRA pilot           0.1200          0.0694       0.4232     0.9963       28.7
```

解释腔也下降：

```text
explains flag:        0.0194 -> 0.0031
generic-pattern flag: 0.0425 -> 0.0169
```

风险：拒答类输出上升：

```text
refusal-style generations: about 0.125% -> about 0.875%
```

结论：512 pilot 明显有效，但需要监控 refusal。

### 9. 3000 档

本地 `hic-compact-json-train-3000` 曾长期卡在 GPU 等待，服务器上没有完整跑完的 3000 context / adapter 证据：

```text
outputs/analysis/hic_humor_viewpoints_sft_train_pilot_3000.jsonl: 1 line
outputs/lora_sft_hic_compact_json_pilot_3000: no checkpoint
```

但用户反馈：自己已经用模型测过 3000 档，效果很好。

基于这个外部结果，决定进入下一步 full run。

### 10. Full run v2

错误的 full run 曾经出现过：

```text
outputs/lora_sft_hic_compact_json_full
```

但那个 run 用的是旧配置：

```yaml
feature_method: feature-method
train_context_path: outputs/analysis/vlm_visual_facts_train.jsonl
val_context_path: outputs/analysis/vlm_visual_facts_val.jsonl
```

而这些 context 文件不存在，所以它没有真正训练。

后来启动了新的 full run：

```text
tmux session: hic-compact-json-train-full
log: outputs/analysis/hic_compact_json_train_full.log
train_context: outputs/analysis/hic_humor_viewpoints_sft_train_full.jsonl
val_context: outputs/analysis/hic_humor_viewpoints_sft_val_full.jsonl
output_dir: outputs/lora_sft_hic_compact_json_full_v2
```

脚本做了保护性升级：

- `TRAIN_LIMIT=full`
- `VAL_LIMIT=full`
- `OVERWRITE_CONTEXT=0`
- context 可断点续跑；
- full training 使用更大的 logging/eval/save cadence；
- 避免覆盖旧 context；
- 避免 pilot 的 `eval_steps=16` / `save_steps=16` 拖垮全量训练。

截至 2026-07-27 13:26 左右，full run 还没有真正开始：

```text
tmux session alive: yes
current stage: waiting for GPU memory
train context lines: 0
val context lines: 0
checkpoint: none
final_lora: none
```

原因是 GPU 被两个 `ollama` runner 占着：

```text
GPU free memory: about 3036 MiB
ollama pid 3744923: about 11GB
ollama pid 3472845: about 9.7GB
```

full run 正在等待：

```text
waiting for gpu=0 free memory >= 18000 MiB
```

## 遇到的核心问题

### 问题 1：gold-caption-derived context 不是部署态

`hic-compact-json` 目前依赖 `image + gold caption` 来提取 humor point 和 viewpoint。

这非常适合做上限实验，因为 gold caption 告诉了 teacher “这个图片应该笑在哪里”。

但真实推理时没有 gold caption。因此当前方法还不是完整可部署 pipeline。

需要补一条 image-only 路线：

```text
image -> image-only humor viewpoint extractor -> compact JSON -> caption generator
```

或者：

```text
image -> generator candidates -> judge/reranker -> best caption
```

### 问题 2：全量 context 生成成本极高

全量规模：

```text
train: 204481
val: 11365
```

每一条都要用 Qwen2.5-VL-7B 生成 viewpoint JSON。这个成本可能是几天级别，远高于 LoRA 训练本身。

所以 full run 的价值要明确：它是 upper-bound SFT，不一定是最终路线。

### 问题 3：GPU 资源调度卡住

多次出现训练/分析任务等待 GPU，但 GPU 被 `ollama` 或其他进程占用。

常见表现：

```text
gpu free_memory=3036 MiB stable=0/2
```

建议后续所有长任务都使用：

- `tmux`
- `tail -f`
- `watch nvidia-smi`
- context 断点续跑
- 不要默认 `--overwrite`
- 明确输出目录与日志

### 问题 4：拒答输出

512 adapter 的 held-out evaluation 中出现少量 refusal：

```text
I'm sorry, but I can't comply with that request.
```

这可能来自 gold caption / teacher context 里包含敏感表达，或者模型 safety behavior 被触发。

如果继续 SFT，需要记录 refusal rate，并考虑：

- 过滤高风险 gold captions；
- 在训练 prompt 中避免触发 safety framing；
- evaluation 中单独统计 refusal；
- 用 preference/reranker 惩罚 refusal。

### 问题 5：仅用 gold-caption text similarity 评估不够

当前 `gold_match_rate` 和 text similarity 能衡量是否接近 gold caption，但幽默 caption 可以有很多合理答案。

应该增加：

- pairwise judge；
- best-of-N candidate selection；
- format/style checks；
- refusal checks；
- groundedness checks；
- diversity checks；
- human/manual review subset。

## 论文脉络与对本项目的启发

### 1. New Yorker Caption Contest benchmark

Hessel et al. 2023, "Do Androids Laugh at Electric Sheep? Humor Understanding Benchmarks from The New Yorker Caption Contest"

启发：

- 幽默不是单纯 caption generation；
- 至少包括 image-caption matching、caption ranking、explanation；
- 对本项目来说，不应该只训练 generator，还要训练 judge/reranker。

链接：

- https://aclanthology.org/2023.acl-long.41/
- https://github.com/jmhessel/caption_contest_corpus

### 2. 大规模幽默偏好数据

Guo et al. 2024, "Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning"

启发：

- 幽默主观性很强；
- 大量人类 preference / rating 比单一 reference caption 更适合训练；
- 对本项目来说，下一阶段应该从 SFT 转向 preference data：对每张图生成多个 candidate，用 gold caption / Qwen judge / 人工判断构造 chosen-rejected pairs。

链接：

- https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf
- https://huggingface.co/datasets/yguooo/newyorker_caption_ranking

### 3. Culture-aware humorous captioning

"Multimodal Humor Generation across Cultural Contexts", arXiv 2026

启发：

- 幽默 caption 不只是视觉 grounding，还常常依赖文化语境；
- HIC/Bokete 数据本身有日本文化、网络梗、语言转换问题；
- 如果 caption 要自然，后续需要记录 culture/reference 类型，不要把所有 humor 都压成视觉 incongruity。

链接：

- https://arxiv.org/abs/2604.18091

### 4. Multimodal humor survey / benchmark

"Computational Humor with Multimodal LLMs: Methods, Datasets and Future Directions", arXiv 2026

启发：

- 多模态幽默任务可分为 understanding、explanation、ranking、generation；
- 本项目已经在做 understanding/explanation 到 generation 的桥接；
- 下一步应该把 bridge 明确成两个模块：viewpoint extractor 和 caption generator，而不是把所有能力塞进一个 LoRA。

链接：

- https://arxiv.org/abs/2607.19011

### 5. Visual instruction tuning

LLaVA 2023, "Visual Instruction Tuning"

启发：

- 用 teacher 生成的 multimodal instruction data 做 tuning 是合理路线；
- 但 teacher-generated data 需要 careful filtering；
- 对本项目来说，`image + gold caption -> compact JSON` 是一种 teacher-generated instruction/context data。

链接：

- https://arxiv.org/abs/2304.08485

### 6. InstructBLIP

Dai et al. 2023, "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning"

启发：

- 多任务 instruction tuning 对视觉语言模型有效；
- 单一任务过拟合可能导致模型学到 dataset shortcut；
- 对本项目来说，只用 gold-caption-derived compact JSON 做 SFT，可能让模型依赖“分析提示”，而不是学会图片幽默理解。

链接：

- https://arxiv.org/abs/2305.06500

### 7. DPO

Rafailov et al. 2023, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"

启发：

- 如果已经能生成多个候选，并能判断哪个更好，DPO 比继续做 reference-only SFT 更贴合目标；
- 幽默 caption 本来就是偏好问题，因此 DPO/reranker 是自然下一步。

链接：

- https://arxiv.org/abs/2305.18290

### 8. Self-rewarding / LLM-as-a-Judge

Yuan et al. 2024, "Self-Rewarding Language Models"

启发：

- 可以用强模型作为 judge 生成 preference signal；
- 但 judge 本身要被校准，不能无条件信；
- 对本项目来说，Qwen7B/GPT-style judge 可以先用于筛选候选和构造偏好对，再用人工小样本校准。

链接：

- https://arxiv.org/abs/2401.10020

## 建议的下一步路线

### 建议 1：不要把 full SFT 当成唯一主线

full SFT 可以跑，但定位应该是：

```text
gold-caption-derived upper-bound experiment
```

它回答的问题是：

> 如果模型知道 gold caption 对应的结构化笑点，它能不能学会生成更短、更贴近 gold humor 的 caption？

它不能直接回答：

> 没有 gold caption 时，模型能不能自己看图生成好笑 caption？

### 建议 2：先做 512/3000 adapter 的系统评估表

应该固定同一批 test examples，例如：

```text
sft_test first 200
sft_test random 1000
hard subset: text_image_contrast / knowledge_reference / dialogue_or_nonvisual
```

比较：

```text
base Qwen3B
base + hic-compact-json prompt
512 LoRA
3000 LoRA
full LoRA if eventually available
```

指标：

```text
gold_match_rate
candidate_text_match_rate
avg_max_text_similarity
format_ok_rate
explanation rate
refusal rate
avg length
diversity
Qwen7B pairwise judge win rate
manual win rate on 50 examples
```

### 建议 3：构造 preference dataset

每张图生成多个 candidates：

```text
base
hic-compact-json prompt
512 LoRA
3000 LoRA
maybe human/gold caption
```

然后用 judge 生成 pair：

```text
chosen: 更接近 gold humor point、短、自然、不解释、不拒答
rejected: 解释腔、描述腔、泛泛而谈、拒答、无关
```

这可以训练：

- reranker；
- DPO LoRA；
- 或先只做 best-of-N reranking。

### 建议 4：训练 image-only viewpoint extractor

用当前 gold-derived viewpoint JSON 作为 pseudo-label，训练或评估一个 image-only extractor：

```text
input: image
output: compact humor candidate JSON
```

这里要承认：image-only extractor 不可能完全恢复 gold caption 的笑点，尤其是 `dialogue_or_nonvisual` 和 `knowledge_reference`。但它可以给 generator 提供多个可能笑点。

推荐输出不是一个 JSON，而是 top-k hypotheses：

```json
[
  {"type": "...", "target": "...", "confidence": 0.72},
  {"type": "...", "target": "...", "confidence": 0.41}
]
```

然后 generator 对每个 hypothesis 生成 caption，再由 reranker 选最好的。

### 建议 5：把 pipeline 改成 generate-and-rank

最终更合理的系统：

```text
image
 -> image-only humor hypothesis extractor
 -> compact JSON hypotheses
 -> generator produces N captions
 -> reranker/judge selects best caption
```

这比单次生成更适合幽默，因为幽默本来就是高方差任务。

### 建议 6：短期行动清单

1. 如果要让 full run 继续，先释放 GPU 上的 `ollama`。
2. full run 未开始前，先不要依赖它做决策。
3. 整理 3000 adapter 的实际路径和评估结果。如果 3000 是你在别处跑的，需要把结果文件放回项目目录。
4. 跑一个统一评估：

```text
base vs 512 vs 3000
test random 1000
8 candidates/image
Qwen7B judge + gold caption
```

5. 建 preference pair 数据。
6. 先训练 reranker，再决定是否 DPO。
7. 并行设计 image-only viewpoint extractor。

## 当前最重要的决策

我建议现在不要继续纠结“full SFT 要不要跑”本身，而是把它降级为后台 upper-bound 实验。

真正主线应该变成：

```text
hic-compact-json upper-bound SFT
        +
candidate generation
        +
gold-caption-aware judge/reranker
        +
image-only humor hypothesis extractor
```

这条路线更接近论文里对幽默任务的处理方式：不是只生成一句话，而是同时做理解、解释、排序和偏好优化。

