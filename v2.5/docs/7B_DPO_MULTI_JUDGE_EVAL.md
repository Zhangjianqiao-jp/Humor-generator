# 7B SFT vs DPO：三评审盲评报告

日期：2026-08-27（JST）

## 结论

三名评审的点估计均未给出 DPO 的稳定显著优势。三评审多数共识下，DPO neutral-imputed win score 为 **53.90%**，按图片聚类 bootstrap 95% CI 为 **[47.87%, 59.93%]**，仍跨过 50%。因此当前 Pilot16 DPO 不能被宣布优于 SFT，也不应仅凭该 checkpoint 启动大规模 module search。

更重要的是评审一致性很低：三评审 overall Fleiss κ 为 **0.122**，best-pick κ 为 **0.067**。`llm_judge_1` 还表现出显著匿名 A 侧偏好（overall A/B=81/41，排除 ties 的双侧二项检验 p=0.000371；best-pick A/B=100/41，p=7.24e-7）。因此不能把三模型多数票等同于可靠的人类共识。

## 数据与协议

- 47 张未进入 SFT train、DPO train 或 DPO validation 的官方图片；
- 三个 generation seeds：20260827、20260828、20260829；
- 每个模型、图片和 seed 各生成 3 条 caption；
- 141 个匿名 Group-of-3 trials；
- 两个外部 LLM judge 使用独立 blind ID、独立题目顺序和独立 A/B 翻转；
- Codex 判断在解盲前已冻结；
- Tie 计 0.5；无三评审多数的 trial 单独记 `unresolved`，主敏感性统计中按 0.5 中性填补。

## 单评审结果

| Judge | DPO score | Image-clustered 95% CI | 说明 |
|---|---:|---:|---|
| Codex | 54.96% | [46.45%, 63.12%] | 无显著位置偏置 |
| LLM Judge 1 | 54.26% | [45.39%, 63.12%] | 显著偏好匿名 A |
| LLM Judge 2 | 51.06% | [44.33%, 57.80%] | A/B 平衡；绝对标签严重压缩为 weak |

所有区间均跨过 50%。

## 三评审共识

| 共识结果 | Trial 数 |
|---|---:|
| DPO | 65 |
| Tie | 11 |
| SFT | 54 |
| 无多数 | 11 |

- 无多数按 0.5：DPO score 53.90%；
- 只看有多数的 130 项：54.23%；
- 把 11 项无多数全部判给 SFT/DPO 的边界：[50.00%, 57.80%]；
- image-clustered neutral-imputed 95% CI：[47.87%, 59.93%]。

绝对标签共识同样保守：DPO group 为 13 good / 118 weak / 4 bad / 6 unresolved；SFT 为 10 / 116 / 6 / 9。按图片要求三个 seeds 至少两个为 good 时，两者都只有 2/47 张。相对胜出不能被解释成真正好笑。

## 一致性

| Pair | Overall raw | Overall κ | Best-pick κ |
|---|---:|---:|---:|
| Codex vs Judge 1 | 36.17% | -0.057 | -0.124 |
| Codex vs Judge 2 | 58.87% | 0.317 | 0.290 |
| Judge 1 vs Judge 2 | 49.65% | 0.129 | 0.055 |

三评审 Fleiss κ：overall 0.122，best-pick 0.067。绝对 A/B 标签 κ 约为 -0.011/-0.039，主要原因是 Judge 1 不使用 bad，而 Judge 2 几乎不使用 good。说明当前 LLM 绝对量表没有校准到同一阈值。

排除存在显著位置偏置的 Judge 1 后，Codex 与 Judge 2 只在 83/141 项上达成相同判断；其余 58 项 unresolved。将 unresolved 按 0.5 处理时 DPO 为 52.84%，95% CI [47.16%, 58.51%]，结论仍不改变。

## 决策

1. 不把 Pilot16 DPO 升格为已验证优于 SFT 的最终模型。
2. 不进行 Fisher/SVD/dynamic-rank 或大规模 layer-wise module search。
3. 两个 LLM judge 可作为误差分析来源，但不能替代人工幽默标注；11 个无多数 trial 与位置偏置相关样本应进入人工 adjudication。
4. 下一项训练若获批准，只做一个固定配置的 Quality64 DPO 低成本对照。它使用 17,297 个官方 crowd pairs，并提高 chosen 的绝对排名质量；不得同时改变 objective、module placement 和 rank。
5. Quality64 先在 validation images 上筛选；47-image test 不用于反复调参。只有 validation 改善且最终 test 的 image-clustered CI 不再跨 50%，才保留 DPO。

## 11 项重新盲化裁决

项目负责人因不熟悉西方幽默语境，明确委托 Codex 完成 11 个无多数 trial 的裁决。裁决只读取重新随机化的公开页面，在读取 `adjudication_public_private_key.json` 前完成并冻结；判断文件 SHA-256 为 `12de2d38036b05b2cddbf6ca06316776188a173e7cb0d946478ad5a508adfa16`。这能避免模型身份泄漏，但裁决者仍是此前三评审之一，因此不能视为第四个独立评审。

裁决后：

- 三 seed DPO score：54.26%、53.19%、52.13%；
- 三 seed 均值：53.19%，标准差 1.06 pp；
- image-clustered bootstrap 95% CI：[46.81%, 59.57%]；
- 图片多数：DPO win 22 / tie 8 / loss 17；
- 图片 majority-good：DPO 2/47，SFT 2/47。

裁决使点估计从 53.90% 变为 53.19%，但置信区间仍跨 50%，绝对 good 图片数完全相同。最终结论因此保持不变：当前 Pilot16 DPO 只有弱正向趋势，没有稳定优势。本轮盲评任务至此关闭，不再追加模型 judge。

## 可复现文件

- Judge 1：`results/7b_generator/dpo_test47_independent_raters/annotations/llm_judge_1.json`
- Judge 2：`results/7b_generator/dpo_test47_independent_raters/annotations/llm_judge_2.json`
- 一致性：`results/7b_generator/dpo_test47_independent_raters/llm_consensus_report.json`
- Policy 共识：`results/7b_generator/dpo_test47_independent_raters/policy_consensus_report.json`
- 单评审解盲：`llm_judge_1_unblinded_report.json`、`llm_judge_2_unblinded_report.json`
- 裁决判断：`annotations/human_adjudicator.json`
- 裁决后最终统计：`adjudicated_final_report.json`

## 论文依据

1. Zhang et al. (2024), *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*, NeurIPS 2024.
2. Hessel et al. (2023), *Do Androids Laugh at Electric Sheep?*, ACL 2023 Best Paper.
3. Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023.
4. Artstein and Poesio (2008), *Inter-Coder Agreement for Computational Linguistics*, Computational Linguistics.
5. Rafailov et al. (2023), *Direct Preference Optimization*, NeurIPS 2023.
