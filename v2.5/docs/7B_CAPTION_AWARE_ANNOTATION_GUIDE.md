# 7B Caption-Aware Compact 标注说明（v3）

## 1. 这次标注到底是什么

训练时，7B 的输入仍然只有图片；gold captions 只用于离线构造监督标签，不会出现在推理输入里。

监督目标不是图片描述，也不是新的 caption，而是下面这个隐变量桥梁：

```text
可见异常关系
→ 人看到它时会激活的日常脚本、第二词义或文化框架
→ 能生成多个高分 caption 的语义连接
```

同一张图片通常有许多高分 caption。标注应解释它们共享的生成机制，而不是复述其中一句。这样训练出的 7B 才有机会在未见图片上预测一个对 3B 真正有用的 humor plan。

## 2. compact JSON 中各字段的职责

- `scene`：只记录最短、最可靠的画面事实。
- `type`：异常关系的粗类别，用于稳定输出空间。
- `target`：核心监督；说明如何从图片走到高分 caption 家族。
- `primary_view/views`：指出判断异常关系所需的观察尺度。
- `anchors`：把推理绑回可见证据，防止只背主题词。
- `external_knowledge`：桥梁是否调用常识、习语、职业脚本或文化知识。

## 3. 正例

图片：国王坐在王座上，头顶悬着一把剑；高分 captions 涉及 overhead、cuts、longevity、succession、insurance。

合格的 `target`：

> Reframe the lethal object hanging over a ruler as a routine royal overhead problem, opening administrative, financial, medical, cutting, and succession language.

它保留了三层信息：可见危险、日常行政脚本、可供 3B 使用的语义连接；但没有复制任何完整 caption。

不合格的图片描述：

> A sword hangs above a king.

不合格的成品 caption：

> Your overhead is going to kill you.

## 4. 人工复核问题

每条只需回答以下问题：

1. `scene` 和 `anchors` 是否能在图片中直接看到？
2. `target` 是否同时包含“视觉触发点”和“解释框架”？
3. `target` 是否能解释不止一条高分 caption，而非追随单个离群 caption？
4. 若删除图片，只看 `target`，它是否仍过于接近某条 gold caption 的措辞？若是，应改写。
5. 若只看图片、不看 gold captions，这个桥梁是否仍有合理的视觉依据？若否，应隔离。

人工可以把 `review.confidence` 改为 `high`、`medium` 或 `reject`。在人工确认前，这批标签应称为 AI-assisted draft，不应称为 human gold。

## 5. 当前产物

- 全部 79 张图片的审阅记录：`data/annotations/nycc_planner_v3/adjudicated/train_labels_caption_aware_v3.jsonl`
- 可训练的 78 条 teacher 标签：`data/annotations/nycc_planner_v3/adjudicated/train_teacher_caption_aware_v3_clean.jsonl`
- 最终 SFT JSONL：`data/processed/newyorker_compact_viewpoint_sft_caption_aware_v3/train.jsonl`
- 隔离记录：`data/annotations/nycc_planner_v3/adjudicated/train_caption_aware_v3_quarantine.jsonl`
- 逐图 target 源文件：`data/overrides/compact_viewpoint_train_caption_aware_v3_targets.json`
- 训练/推理 prompt：`prompts/7b_image_to_caption_aware_compact_viewpoint.txt`
- 泄漏审计：`data/annotations/nycc_planner_v3/adjudicated/train_caption_aware_v3_leakage_audit.json`

`nycc_656` 已确认应隔离，而且不是因为 gold captions 不幽默：`ranking/655.csv` 与 `ranking/656.csv` 的 5,278 条 caption 集合完全相同（Jaccard = 1.0），均对应 `cartoons/source/655.jpg` 的巨型地球演讲图；`cartoons/source/656.jpg` 的吉他治疗图实际对应 `ranking/657.csv`。但 `source/655.jpg` 位于官方 description 的 test split，不能为了补齐第 79 条而移入训练集，否则会产生训练—测试图像泄漏。

## 6. 研究依据

1. Hessel et al., “Do Androids Laugh at Electric Sheep? Humor ‘Understanding’ Benchmarks from The New Yorker Caption Contest,” ACL 2023 Best Paper. 该工作把视觉实体、异常点和笑话解释拆开标注与评价，支持显式建模中间 humor reasoning。<https://aclanthology.org/2023.acl-long.41/>
2. Hessel et al., “Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning,” NeurIPS 2024 Datasets and Benchmarks. 该工作提供大规模 New Yorker 人类偏好，支持把高分 caption 当作幽默监督，但投票排名本身不等同于解释标签。<https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html>
3. Zhou et al., “Improving Multimodal Humor Understanding and Generation through Visual Annotation, Humor Reasoning and Preference Alignment,” Findings of EMNLP 2025. 该工作支持视觉标注、幽默推理与偏好对齐的分阶段组合。<https://aclanthology.org/2025.findings-emnlp.884/>
