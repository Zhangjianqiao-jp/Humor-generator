# v3.5 实验失败与计划变更准则

本文件规定所有 v3.5 实验失败的记录方式。机器可读记录位于
`docs/EXPERIMENT_FAILURES.jsonl`，使用
`scripts/record_experiment_failure.py` 追加，禁止静默重试或覆盖旧输出。

## 强制字段

每次失败必须记录：唯一 ID、时间、失败层级、状态、可观察现象、原始证据、
根因、采取的修复、对实验计划的影响、相关 artifact。失败层级只能是：

- `environment`：调度器、CUDA、NVML、依赖或文件系统；
- `data`：缺失、泄漏、schema、hash 或 provenance；
- `engineering`：shape、dtype、OOM、hidden/token 对齐或实现错误；
- `method`：代码正确运行，但预注册的机制/统计 gate 未通过；
- `evaluation`：盲化、评审、统计单位或指标错误。

只有 `method` 失败才允许被写作“方法没有通过”。环境和工程失败不得作为模型
优劣证据。每次修复后必须创建新输出目录，不允许覆盖失败实验。

## 当前结论

1. `cross_attention_semantic_v1` 是方法级 No-Go，不是运行失败。其 reconstruction
   NLL 明显下降，但 validation matched-minus-shuffled gap 只有 `0.004843`，低于
   `0.02` gate，且 `fraction_gap_gt_margin=0`。
2. 代码审计发现 v1 把 conflict/local/global 拼接后做单一 positional softmax，长
   association channel 可能产生长度竞争；同时实现的 InfoNCE 没有进入正式 trainer。
3. v2 改为通道内独立 attention、通道间 receiver-dependent gate，并强制在梯度
   累积窗口上调用真实 InfoNCE。GPU smoke 也必须用两个真实 trace 执行同一
   contrastive backward，不能只由 CPU 单测证明。caption bridge 在新的语义 gate
   通过前仍被禁止。
4. plain-text/latent 不做一次性全 latent 假设。语义 gate 通过后按逐步消融比较：
   Text-HOMER、conflict-text+association-latent、conflict-latent+association-text、
   all-latent；只对胜出的混合方式继续拆分 local/global。
5. v2 GPU engineering smoke 作业 `6688553` 与 stationary-teacher post-fix 作业
   `6688566` 均已通过：两个真实 trace 均完成 hierarchical attention、
   InfoNCE/variance backward 和 optimizer update；policy trainable params 为 0。
   此项只关闭工程风险，不改变 v1 的方法级 No-Go。
6. Hierarchical Phase A v2 作业 `6688689` 已完成但仍为方法级 No-Go：causal
   matched/shuffled gap 为 `0.002664 < 0.02`，conflict router mass 下降到
   `0.0289`。validation retrieval 的原始 `0.190476` 受同 cluster 重复 caption
   false-negative 污染，已单独登记为评测实现错误；不能用它支持或反对方法。
7. 对第 6 项的声明范围已审计收紧：log-probability 与两遍反传公式正确，但 negative
   是跨 image cluster plan，不是同图单通道 counterfactual；`0.02` 也是工程阈值而非
   统计校准阈值。因此结论只能是“all-latent v2 未通过当前操作性 gate”，不能写成
   “latent communication 已被证伪”。
8. v2 的 reconstruction 按全部 target token 平均，较长 local/global channel 获得更多
   监督；unregularized router 随后将 conflict mass 压到 `0.0289`。v2 InfoNCE teacher
   还是固定的 receiver-embedding 随机投影，只约束 trace identity，并不充分证明
   frozen Receiver 能解释这些语义。两项均记录为设计不足。
9. 计划已改为低成本 Phase A3：三通道等权 reconstruction、三路 contextual InfoNCE、
   单通道 matched/shuffled counterfactual、固定等权 gate、cluster bootstrap CI。只有
   三个 channel 都越过校准后的 control 才能进入 caption bridge；否则转入
   conflict-text + association-latent 混合方案。
10. 样本量解释已更正：`64/24` 是开发期机制 pilot，不是方法级 confirmatory experiment。
    24 clusters 对接近 50% 的比例估计，正态近似 95% 半宽约 0.20，无法可靠检测小效果。
    因此 Phase A3 的 24-cluster point pass 只能触发 sealed outer semantic validation；
    v1/v2 也必须用同一 cluster-level evaluator 重算后才能公平比较。

## 权威依据

1. Shang et al., HOMER, ICLR 2026: https://openreview.net/pdf?id=SzaRhPom4o
2. He et al., MoCo, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html
3. Yang et al., Hierarchical Attention Networks, NAACL 2016: https://aclanthology.org/N16-1174/
4. Libovicky and Helcl, Multi-Source Attention, ACL 2017: https://aclanthology.org/P17-2031/
5. van den Oord et al., Contrastive Predictive Coding, 2018: https://arxiv.org/abs/1807.03748
6. Card et al., Statistical Power in NLP, EMNLP 2020: https://aclanthology.org/2020.emnlp-main.745/
7. Dror et al., Significance Testing in NLP, ACL 2018: https://aclanthology.org/P18-1128/
