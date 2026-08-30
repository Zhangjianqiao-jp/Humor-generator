# 7B DPO 数据集校验报告

校验日期：2026-08-25（JST）

## 结论

当前数据通过结构、标签方向、图片、split、prompt、reference-logp 与常见文本捷径检查，`ERROR_COUNT=0`，可以继续用于当前 7B DPO baseline。

必须准确描述数据来源：caption ranking 来自 Zhang et al. (NeurIPS 2024) 公开数据；当前 JSONL preference pairs 是本项目按论文规则从 ranking 本地重建的，不是作者发布的一份固定 pair JSONL。随后又进行了本地 train-ready 筛选：每个 contest 保留 16 个无重复、相对字符长度差不超过 0.35、且优先靠近三倍不确定性阈值的 hard pairs。该筛选没有反转或修改 chosen/rejected 标签。

## 实际训练文件

| Split | Pair 数 | Image/contest 数 | 每图 pair 数 |
|---|---:|---:|---:|
| train | 1,264 | 79 | 16 |
| validation | 384 | 24 | 16 |
| test | 384 | 24 | 16 |
| 合计 | 2,032 | 127 | 16 |

训练实际读取：

- `data/processed/newyorker_published_dpo_reference_7b_generator/dpo_train.jsonl`
- `data/processed/newyorker_published_dpo_reference_7b_generator/dpo_validation.jsonl`
- `data/processed/newyorker_published_dpo_reference_7b_generator/dpo_test.jsonl`

## 已通过的检查

1. 所有 pair ID 和 `(image, chosen, rejected)` 均无重复。
2. chosen 与 rejected 文本不相同。
3. 全部满足 `chosen_score > rejected_score`、`chosen_rank < rejected_rank`、`score_margin > 0`。
4. 全部满足 `z_margin >= 3`，即均通过论文使用的三倍组合不确定性门槛。
5. train、validation、test 的 contest、image 和完整文本 pair 交集均为零。
6. 127 张唯一图片全部存在且可以由 PIL 解码。
7. 全部 prompt 具有统一的 `Humor plan / ANCHOR / CONTRAST / ANGLE` schema；同一 image 的 prompt 完全一致。
8. compact plan 的元数据来源全部为 `release_gpt4o_description_only`，不是从当前 chosen/rejected caption 反推；没有任何 chosen 或 rejected caption 被原样包含在 prompt 中。
9. 原始 selected pair 与添加 7B reference-logp 后的行逐字段一致；新增内容只有 `reference_logps`。
10. 全部 chosen/rejected reference log-prop 为有限值，token 数均大于零，reference adapter 均为 `outputs/7b-generator/best_val_loss`。
11. 训练 loader 已确认读取全部 1,264/384 个 train/validation pairs，未跳过样本。

## Preference 强度

| Split | score margin 最小值 | 中位数 | 95 分位 | 最大值 | z-margin 中位数 |
|---|---:|---:|---:|---:|---:|
| train | 0.1109 | 0.2083 | 0.5536 | 1.0328 | 3.0232 |
| validation | 0.1080 | 0.2053 | 0.2580 | 0.2853 | 3.0247 |
| test | 0.1111 | 0.2062 | 0.2925 | 1.0606 | 3.0212 |

这些是统计上高置信、但刻意选择得较难的 pair。它们不是大量明显错误的 easy negatives，适合 preference ranking；同时也意味着 DPO 的 held-out 提升可能不会很大。

冻结 7B SFT reference 已正确偏好 chosen 的比例：

| Split | sequence log-prop | length-normalized log-prop |
|---|---:|---:|
| train | 60.7% | 60.4% |
| validation | 59.6% | 62.2% |
| test | 59.9% | 59.6% |

这说明 SFT 已学到一部分 preference，但仍有约 40% hard pairs 排序错误，存在可供 DPO 调整的 residual ranking signal。

## 长度与模板捷径

chosen 比 rejected 更长的比例为 train/validation/test 的 49.3%/47.1%/47.7%；更短的比例为 48.6%/52.1%/50.5%。平均相对字符长度差约 0.18，最大不超过 0.35，没有明显“chosen 总是更长或更短”的捷径。

全部 split 合计的模板统计：

| 标记 | chosen | rejected | 差值 |
|---|---:|---:|---:|
| POV | 0.00% | 0.00% | 0.00% |
| Bro | 0.05% | 0.15% | -0.10% |
| Meanwhile | 0.05% | 0.00% | +0.05% |
| Emoji | 0.00% | 0.00% | 0.00% |
| `!` | 9.20% | 11.81% | -2.61% |
| `?` | 17.47% | 17.91% | -0.44% |

未发现明显网络模板或标点捷径。

## 明确限制

1. 所有 2,032 个 pair 都是 H2（较强幽默 vs 较弱幽默）。数据不能单独训练或证明 H1 humor-vs-literal、H3 grounding-vs-hallucination 或 H4 image-specific-vs-generic 能力。
2. Caption preference 来自 crowd ranking，不是针对本项目 compact plan 条件重新标注；本项目只保证 pair 两侧共享完全相同的 image 与 prompt。
3. 当前 validation/test 各只有 24 张图片。384 pairs 不是 384 个独立图片样本，置信区间和显著性必须以 image/contest 为 bootstrap 单位，不能把 pair 当作独立样本。
4. 数据许可证为 `CC-BY-NC-4.0`，不允许直接用于商业用途。
5. 7B objective 直接采用 DPO 是项目工程决策；本数据校验不能证明 DPO 在 7B 上优于 SimPO、IPO 或 Anchored objective。

## 数据规模与论文对照

当前 1,264 个 train pairs 足够用于 low-data pilot，但不足以作为最终数据规模。Zhang et al. 的 NeurIPS 2024 实验为每个训练 contest 构造 1,000 个三倍标准差 pair，并最多训练 1 epoch；当前每图 16 pairs 仅为该密度的 1.6%。论文还将 DPO 生成多样性的提升与每张漫画具有多种人类 caption 联系起来。因此，不能仅根据 16/image 的一次训练结果断言 7B 已充分吸收幽默偏好。

通用多模态 preference 数据也通常更大：VLFeedback 包含约 82.4K instructions 和 399.4K pairs；其对比表列出的 LLaVA-RLHF、POVID 分别约为 10K 和 17.2K。它们主要训练 helpfulness、visual faithfulness 与 safety，不直接提供 New Yorker 幽默风格，因此不应直接混入当前 caption DPO 主训练集。

## 已建立的同域扩展集

优先使用当前公开 ranking 的同域人类 preference，而不是 AI 重新标注或混入异域数据。已生成：

`data/processed/newyorker_published_dpo_pairs_expanded64/`

| Split | Pilot | Expanded64 | 说明 |
|---|---:|---:|---|
| train | 1,264 | 5,056 | 每图从 16 扩到 64 |
| validation | 384 | 384 | 逐行保持不变 |
| test | 384 | 384 | 逐行保持不变 |

扩展规则：

1. 当前 1,264 train pairs 全部保留，形成严格嵌套的数据规模 ablation；
2. 新 pair 只来自已按论文规则构造的公开 ranking candidate pool；
3. 继续要求 `z_margin >= 3` 与相对字符长度差不超过 0.35；
4. 优先选择未被已有 pair 使用的 chosen/rejected caption；
5. 不生成、不修改、不反转 preference 标签；
6. 每图 unique chosen 平均从 15.09 增至 61.41，unique rejected 从 15.94 增至 63.94；
7. validation/test 不变，以保证 16/image 与 64/image 的公平比较。

如果 Expanded64 相对 Pilot16 在 image-level blind win rate 上产生稳定正收益，再扩到 128/image（10,112 train pairs）。不直接在单 MIG 上训练完整约 77,944 对，是计算成本控制，而不是数据不足：保持现有标准后，每张训练图仍有至少 214 个、median 507 个可选 pair。

进一步审计发现，本地官方 release 实际包含 385 个 cartoons/rankings，官方 description split 为 train/validation/test = 271/44/47；早期 79/24/24 只覆盖 contest 530–659 的项目 SFT 子集。因此，主扩展不再局限于增加同图 pair 密度，还增加独立图像覆盖。

已生成完整官方 split 版本：

`data/processed/newyorker_published_dpo_pairs_fullsplit64/`

| Split | Pair 数 | Image/contest 数 | 目标 pair/image | 例外 |
|---|---:|---:|---:|---|
| train | 17,297 | 271 | 64 | contest 850 仅 17 |
| validation | 2,783 | 44 | 64 | contest 730 仅 31 |
| test | 3,006 | 47 | 64 | contest 690 仅 62 |

三个例外是因为论文式随机构造达到每 contest 200,000 次尝试上限后，再应用长度匹配，可用 pair 少于 64。没有为补齐数量而降低 `z_margin >= 3`、修改长度阈值或制造标签。Train 只比理论 17,344 少 47 pairs（0.27%）。所有实际 pair 均通过标签方向、置信度、长度、图片可解码、prompt 无答案原文和 split 无重叠检查。

最终数据规模实验固定原 24-image validation/test，以免同时改变训练量与评测分布：

1. Pilot16：1,264 train pairs / 79 images；
2. FullSplit64：17,297 train pairs / 271 images；
3. 两者均在原 384-pair validation 与 384-pair test 上比较，并进一步进行 image-level caption generation blind win rate。

`Expanded64`（5,056 pairs / 79 images）保留为可选密度 ablation，但不作为下一轮首选；增加独立 images 比仅增加同图 pairs 更优先。

## 最终判断

**Go：保留当前 16/image 训练作为 low-data baseline，同时采用 FullSplit64（17,297 pairs / 271 images）作为下一轮主训练数据。** 数据干净、条件一致、难度适当并存在 residual preference signal。训练结束后必须用 image-level held-out generation blind win rate 比较 7B SFT、Pilot16-DPO 与 FullSplit64-DPO；仅凭 DPO loss 或 pair accuracy 上升不能宣布幽默生成质量提高。

## 论文依据

1. Zhang et al. (2024), *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
2. Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
3. Hessel et al. (2023), *Do Androids Laugh at Electric Sheep? Humor Understanding Benchmarks from The New Yorker Caption Contest*, ACL 2023 Best Paper. https://aclanthology.org/2023.acl-long.41/
4. Li et al. (2024), *VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment*. https://arxiv.org/abs/2410.09421
