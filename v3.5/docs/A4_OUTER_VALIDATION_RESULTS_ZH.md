# A4 channel-isolated outer validation（2026-09-04）

## 结论先行

新的 A4 方法通过了工程和 provenance validator，但没有通过预注册的联合语义
`Go` 门禁；状态是 **`outer_semantic_inconclusive`**。这不是“latent 已被证伪”，
也不是 caption 质量结果。当前没有新 caption 盲评数据，因此不能计算或声称
`good-caption rate` 提升。

## 运行身份

| 项目 | 值 |
|---|---|
| 作业 | `6711872` |
| 资源 | `c-batch + gpu=1`，完整 H100 80GB，共享节点 |
| 运行时间 | 247 s（23:07:10–23:11:17 JST） |
| protocol | `A4_channel_isolated_v4` |
| 统计单位 | image cluster |
| outer clusters | 40（排除用于 early stopping 的 24 clusters） |
| seeds | `20260830, 20260831, 20260832` |
| detail rows | 360 = 40 × 3 × 3 channels |
| validator | pass |
| checkpoint SHA-256 | `c757ace014f163ce3439e8d49e77f4dc4a46bc7af53e8bfc0fd4d9dccebfc70e` |
| config SHA-256 | `10b13c5119b35ce3ebdb7d457fdd313cfb8e2868448488f45246de1f4ee9b1f4` |
| evaluator SHA-256 | `047b09794a14da86353cfb7aad063058a6313e5b3758c6b1748db08658b4e911` |
| evaluator commit | `086bf94055fc2bad5b3149277cf10580b056ecd4` |

协议固定为：image-free semantic recovery prompt、target-only one-hot channel mask、
逐 channel length-priority donor、donor-side reconstruction 和 zero-bridge control。
这一步没有生成 caption，也没有调用 DPO/preference pipeline。

## 主要结果

gap 定义为

```text
log p(target semantics | matched channel)
- log p(target semantics | length-matched counterfactual channel)
```

| channel | mean gap | fraction gap > 0 | image-cluster bootstrap 95% CI | 预注册 mean ≥ 0.01 | 结论 |
|---|---:|---:|---:|---:|---|
| conflict | 0.01150 | 0.600 | [0.00140, 0.02260] | 通过 | 达到点门禁 |
| local/grounding | 0.00494 | 0.725 | [0.00201, 0.00807] | 未通过 | 正向但效应量不足 |
| global/association | 0.00321 | 0.600 | [0.00043, 0.00605] | 未通过 | 正向但效应量不足 |

三路 bootstrap CI 下界均高于 0，说明在这 40 个 outer clusters 上，receiver 的
teacher-forced semantic log-probability 对替换 channel 有稳定的正向敏感性；但是
当前预注册规则要求每个 channel 同时达到 `mean gap ≥ 0.01` 和
`fraction gap > 0 ≥ 0.60`，因此 local/global 未达到联合 `Go`。

matched 与 zero-bridge 的平均 NLL（120 rows/channel）如下：

| channel | matched NLL | zero-bridge NLL | matched − zero（越低越好） |
|---|---:|---:|---:|
| conflict | 2.2830 | 3.7734 | −1.4904 |
| local | 1.1491 | 1.6373 | −0.4882 |
| global | 0.9432 | 1.2967 | −0.3535 |

这支持“bridge 能改善语义重建”这一较弱结论，但不能替代 caption-level downstream
效用，也不能证明三个 channel 对幽默生成同等重要。

## 重要限制

1. 语义 outer 的 log-probability 是 teacher-forced、确定性 forward；三个 seed 的
   seed-level mean 完全相同（seed std=0），表示可复现的数值重放，不是三次独立的
   随机 caption generation。caption 阶段仍必须使用独立 generation seeds。
2. `zero_bridge_gap=0` 是通信残差不存在时的控制定义，不是“随机 bridge”对照。
   随机初始化 bridge 仍属于后续可选敏感性分析，不能从本次结果推断。
3. 正向 semantic gap 不能推出 humorous caption 的 `good` 比例、grounding 或
   originality 提升。旧的 `outputs/pilot_validation/` 属于 A4 之前的协议，没有
   当前 checkpoint/evaluator provenance 和有效绝对评分，继续 quarantine。

## 决策

当前按 fail-closed 规则：

```text
不启动 caption bridge 训练
不启动 Group-of-3/10 caption 盲评
不启动 DPO 或 preference learning
不把旧 caption packet 汇入新统计
```

下一步必须先由项目负责人预注册一种方案：

1. 保持本次阈值不变，增加 outer clusters 或重复独立 A4 bridge 训练以估计效应量；或
2. 依据仅在训练集确定的 power/效应量依据重新校准 `0.01`，**不能看到本次结果后
   临时降阈值**，并在新输出目录重跑。

只有明确记录为 `outer_semantic_go` 后，才可生成新的 Text-HOMER/latent/hybrid
caption，并用共同 seeds、Group-of-3 筛选、Group-of-10 主评测、镜像 A/B、多评审、
`good/weak/bad` 绝对标签和 image-clustered bootstrap 报告 good-caption rate。

## 权威依据

- [HOMER（ICLR 2026）](https://openreview.net/forum?id=SzaRhPom4o)：结构化冲突与联想链的 cartoon humor 规划。
- [InterLat（ACL 2026）](https://aclanthology.org/2026.acl-long.1248/)：latent communication 需要表示分离与下游任务约束。
- [BLIP-2（PMLR 2023）](https://proceedings.mlr.press/v202/li23q)：表示对齐后再进行下游生成训练的两阶段范式。
- [CPC / InfoNCE](https://arxiv.org/abs/1807.03748)：对比表示学习目标；本项目仅将其作为辅助诊断，未把它当作因果 channel-use 证明。
- [Dror et al.（ACL 2018）](https://aclanthology.org/P18-1128/) 与 [Card et al.（EMNLP 2020）](https://aclanthology.org/2020.emnlp-main.745/)：NLP 评测的显著性与统计功效注意事项。
