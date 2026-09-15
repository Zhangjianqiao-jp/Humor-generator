# A5 Caption Group-of-10 盲评结果

## 1. 评测范围

本次正式候选来自同一批 sealed Planner traces：97 张 `internal_test` 图片和 24 张
`official_hia_unseen_test` 图片，共 121 个 image clusters。每个条件使用相同的
10 个 generation seeds，比较：

```text
sft::text_homer       canonical Text-HOMER baseline
sft::full_plan_text   full-plan text control
sft::a5_typed         A5 receiver-driven cross-attention latent
```

共 3,630 条 caption，两个比较各生成正反镜像 packet，共 484 个 Group-of-10 packet、
242 个 mirror pairs。三名评审均完整覆盖 484 个 packet，prompt hash 和
`temperature=0` 校验通过。

评测协议保留 Humor in AI 的 group preference 端点，并采用同图困难比较、镜像 A/B、
图片级 bootstrap 和绝对质量标签。它是 paper-aligned adaptation，不声称逐行复现
官方评测：[Humor in AI](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html)、
[Electronic Sheep](https://aclanthology.org/2023.acl-long.41/)。

## 2. 主要相对结果

这里的 challenger 始终是 `sft::a5_typed`；Tie 按 0.5 计入 win rate。CI 和 p 值在
图片级 mirror collapse 后计算。

| Reference | Endpoint | A5 win rate | 95% CI | Holm p | 结论 |
|---|---|---:|---:|---:|---|
| Text-HOMER | Overall | 0.2982 | [0.2534, 0.3437] | 0.000040 | A5 显著较差 |
| Text-HOMER | Best Pick | 0.4112 | [0.3623, 0.4614] | 0.000060 | A5 显著较差 |
| Full-plan text | Overall | 0.5131 | [0.4663, 0.5613] | 1.000000 | 无显著差异 |
| Full-plan text | Best Pick | 0.4979 | [0.4470, 0.5482] | 1.000000 | 无显著差异 |

### 分 split 稳定性

| Reference | Endpoint | Internal 97 images | Official unseen 24 images |
|---|---|---:|---:|
| Text-HOMER | Overall | 0.2904 [0.2405, 0.3428] | 0.3299 [0.2361, 0.4236] |
| Full-plan text | Overall | 0.5043 [0.4519, 0.5567] | 0.5486 [0.4410, 0.6563] |

方向在两个 split 上一致：A5 没有超过 Text-HOMER；相对 full-plan text 只表现为
近似持平。官方 unseen 只有 24 张图，CI 较宽，不能单独作为强结论。

## 3. 绝对质量标签

以下比例是 mirror collapse 后的 `image × rater × seed` 观察；
`mirror_disagreement` 单独保留，不强行归入 good/weak/bad。

### Candidate-level

| Comparison | System | good | weak | bad | mirror disagreement |
|---|---|---:|---:|---:|---:|
| Text-HOMER vs A5 | A5 | 2.75% | 30.39% | 57.88% | 8.98% |
| Text-HOMER vs A5 | Text-HOMER | 4.10% | 39.72% | 47.47% | 8.71% |
| Full-plan text vs A5 | A5 | 3.09% | 31.32% | 56.50% | 9.09% |
| Full-plan text vs A5 | Full-plan text | 2.73% | 31.07% | 57.74% | 8.46% |

### Group-level

| Comparison | System | good | weak | bad | mirror disagreement |
|---|---|---:|---:|---:|---:|
| Text-HOMER vs A5 | A5 | 1.10% | 71.63% | 19.56% | 7.71% |
| Text-HOMER vs A5 | Text-HOMER | 3.86% | 77.96% | 9.09% | 9.09% |
| Full-plan text vs A5 | A5 | 1.10% | 76.86% | 14.88% | 7.16% |
| Full-plan text vs A5 | Full-plan text | 0.55% | 76.86% | 19.56% | 3.03% |

A5 相比 Text-HOMER 的 good rate 更低、bad rate 更高；相比 full-plan text，绝对标签
接近，但没有足够证据表明 A5 产生更高质量 caption。

## 4. 多维评分

评审分数为 1–5；`hallucination_severity` 越低越好。

| Dimension | A5 − Text-HOMER | A5 − Full-plan text |
|---|---:|---:|
| humor | −0.3395 | −0.0833 |
| image_grounding | −0.3643 | −0.0090 |
| image_relevance | −0.3457 | −0.0606 |
| originality | −0.3526 | −0.0344 |
| specificity | −0.2796 | −0.0744 |
| fluency | +0.0861 | −0.0076 |
| hallucination_severity | +0.3127 | +0.0220 |

A5 相比 Text-HOMER 在 humor、grounding、relevance、originality、specificity 上均
较低，且幻觉严重度更高；仅 fluency 略高。相比 full-plan text，各维度近似，但 humor
和 specificity 仍略低。

## 5. Seed 稳定性与评审一致性

聚合的 absolute score 使用 `good=1`、`weak=0.5`、`bad=0`，镜像争议按两侧平均。
10 个 seed 的 score sample variance 为：

| System | Variance | Seed mean range |
|---|---:|---:|
| A5 | 0.00712 | 0.107–0.326 |
| Full-plan text | 0.00432 | 0.118–0.355 |
| Text-HOMER | 0.00413 | 0.196–0.355 |

位置诊断：raw A-choice rate=`0.5062`，mirror system-choice consistency=`0.8740`，
说明没有明显的固定 A/B 位置偏置。但相对 endpoint 的 Krippendorff nominal α 很低：

```text
Text-HOMER vs A5, Overall:   -0.0292
Full-plan text vs A5, Overall: -0.0029
```

三名评审的 A5 Overall win rate（未合并评审）分别为：

```text
Text-HOMER 对比：0.3264 / 0.1529 / 0.4153
Full-plan 对比：0.5041 / 0.5950 / 0.4401
```

因此 Text-HOMER 对比的方向在三位评审中一致，但强度不同；full-plan 对比存在明显
评审异质性。依据 [Dror et al.](https://aclanthology.org/P18-1128/) 的统计谨慎原则，
低一致性下不能只凭单一 p 值宣称强主观质量结论。正式论文版本应加入目标文化人类
评审或预注册 adjudication，而不是简单增加同一 LLM 的投票次数。

## 6. 结论边界

1. A5 已通过 semantic outer gate，证明 latent channel 对替换消息存在稳定的
   receiver-level causal sensitivity；
2. 该语义敏感性没有转化为 caption humor gain；
3. 相比 Text-HOMER，A5 在 Overall、Best Pick、绝对 good rate 和多数质量维度上
   均较差；
4. 相比 full-plan text，A5 统计上基本持平，不能证明连续 latent 优于等信息量文本；
5. 当前不应启动 DPO，也不应宣称 A5 提升了幽默生成；
6. 多样性指标只能作为辅助。A5 的 candidate set 虽然没有明显的同图重复，但绝对
   质量较低，不能把 lexical diversity 上升写成幽默多样性提升。

## 7. 建议的后续实验

优先做低成本机制修复，而不是扩大训练：

1. 保留 A5 作为 frozen semantic bridge baseline；
2. 增加 `typed text anchor + A5 latent enrichment` 的 hybrid 条件，单独报告 text
   anchor 和 latent residual 的贡献；
3. 检查 receiver cross-attention 的 layer/gate 是否在 caption generation 时实际
   激活，并加入 caption-level matched/shuffled functional test；
4. 重新使用同一 121 图、同一 10 seeds、同一 blind protocol 做 hybrid 对比；
5. 只有下游 Overall 和 absolute good rate 在独立评审下改善，才考虑 preference
   learning。否则问题属于 semantic-to-generation interface，而不是 DPO 数据规模。

