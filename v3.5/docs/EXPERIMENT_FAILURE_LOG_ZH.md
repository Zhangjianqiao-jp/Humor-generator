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

## 权威依据

1. Shang et al., HOMER, ICLR 2026: https://openreview.net/pdf?id=SzaRhPom4o
2. He et al., MoCo, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html
3. Yang et al., Hierarchical Attention Networks, NAACL 2016: https://aclanthology.org/N16-1174/
4. Libovicky and Helcl, Multi-Source Attention, ACL 2017: https://aclanthology.org/P17-2031/
