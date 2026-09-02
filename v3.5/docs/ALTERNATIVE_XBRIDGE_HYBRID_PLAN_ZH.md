# 备选方案：Grounding Anchor + Latent Enrichment + Gated Cross-Attention

状态：**备选 pipeline 已进入工程实现，尚未提交正式训练**。本文件不改变已完成的 v3.5 pilot，也不将 XBridge 的跨任务结果视为本项目结果。新实现必须先通过 CPU tests、真实模型单步 GPU smoke 和下述语义 Go/No-Go，才能提交 pilot。

## 0. 本轮修正决定

1. 主方法不使用 soft-vocabulary \(p^\top E\)，也不使用推理时 nearest-token 量化。
2. 不再截取每个 channel 的最后 8 个 states；cross-attention 读取 strict Planner trace 中 conflict/local/global 的完整 states 和 mask。
3. 不把 latent slots 追加到 assistant generation marker 后。Generator 保留正常的 image + SFT task instruction；latent 完全通过 out-of-band cross-attention 提供，称为 `zero-prefix`，而不是字面删除任务指令。
4. 采用 representation-first、generation-second 两阶段训练：先精确重建 Planner semantics，再训练 caption。
5. matched/shuffled 权重由 0.1 提高到 0.5，并直接记录 raw gap 与成功比例；是否足够由数据 gate 决定，不靠权重大小宣称。
6. 加入 exact plan reconstruction、receiver-native InfoNCE、同图不同 plan counterfactual 和 channel typing。InfoNCE 已进入正式 Phase A 的 gradient-window 优化路径与 GPU smoke；仅 NLL/KL 下降不再构成 Go 证据。
7. 历史 `typed_quantized` 只作为失败诊断保留，不再进入候选主方法。若未来研究离散 channel，必须从头联合训练 VQ-VAE（straight-through + codebook/commitment）或 FSQ，不允许 post-hoc nearest-neighbor。

注意：这里删除的是把 bridge 输出投影成词表概率的 **vocabulary softmax**。下文 cross-attention 中的 softmax 只是对完整 memory positions 计算可微检索权重；它既不选最近词，也不把语义压成离散 token，更不截断 memory。

## 1. 研究动机

当前 Learned/Typed bridge 先将 Planner 的完整 hidden sequence 压缩成 24 个固定 slots，再插入冻结 Generator 的输入 embedding 序列：

\[
Z=B_\phi(H_P),\qquad
X_R^{(0)}=[E_R(x,p);Z;E_R(s)].
\]

该方案已经避免生成崩坏，但 outer-validation 的 matched-minus-shuffled caption log-probability gap 约为零，尚未证明 Generator 使用了图片特定的 conflict/association。固定 prefix 还要求 Generator 的早期层先解释处于非原生分布的连续向量。

XBridge 的核心启示不是“换一个更大的 projector”，而是将通信拆成：

1. receiver-native lexical/grounding anchor，保存实体与视觉事实；
2. continuous latent enrichment，保存 Sender 对关系、冲突和联想的上下文化表示；
3. receiver-driven cross-attention，让 Generator 按当前 token 和当前层的需要查询 Sender memory；
4. gated residual，控制连续信息对冻结模型的扰动。

本项目对应的候选系统为：

\[
\boxed{
x+\text{text grounding}
+\text{latent conflict/association memory}
+\text{gated cross-attention}
}
\]

## 2. 与当前 Learned Prefix Bridge 的数学差异

### 2.1 当前静态 prefix bridge

对 Planner states \(H_P\in\mathbb R^{T\times d_P}\)，用 \(K\) 个 learned queries 压缩：

\[
Z=\operatorname{LN}\!\left(
W_O\operatorname{MHA}(Q_0,W_KH_P,W_VH_P)
\right),\qquad Z\in\mathbb R^{K\times d_R}.
\]

之后 \(Z\) 只在输入边界插入一次。压缩查询 \(Q_0\) 与 Generator 当前的图片、生成位置和中间推理状态无关。Receiver 后续仍可通过普通 self-attention 使用这些 slots，但所有信息必须先通过固定的 \(T\to K\) 压缩瓶颈。

### 2.2 Receiver-driven latent enrichment

保留 conflict/local/global 的 Sender states 作为 memory：

\[
M_P=\{H_P^{conflict},H_P^{local},H_P^{global}\}.
\]

在 Generator 的候选层 \(l\in\mathcal L\) 中：

\[
Q_l=W_{Q,l}\operatorname{LN}_R(h_R^l),
\]

\[
K_l=W_{K,l}\operatorname{LN}_P(M_P),\qquad
V_l=W_{V,l}\operatorname{LN}_P(M_P),
\]

\[
A_l^k=\operatorname{softmax}_{t\in k}\left(\frac{Q_l(K_l^k)^\top}{\sqrt b}\right)V_l^k,
\]

\[
\rho_l=\operatorname{softmax}_{k\in\{C,L,G\}} f_l(Q_l,A_l^k),
\qquad
h_R^{l\prime}=h_R^l+\tanh(\alpha_l)W_{O,l}\sum_k\rho_l^kA_l^k.
\]

这里的查询来自 Receiver 当前状态，因此不同图片、caption token 和 Transformer 层可以读取不同 Sender positions。它避免把 final-layer Sender states直接冒充 input embeddings。

### 2.3 表达能力与代价

cross-attention 的条件读取通常比固定 prefix 更有表达力，但不具有无条件性能保证：

- 优点：不必过早压缩；查询随 Receiver 状态变化；可在中/后层注入；gate 可限制破坏。
- 风险：参数、显存和延迟更高；小数据更易过拟合；gate 可能收缩到零；仅有 cross-attention 仍不能证明语义被使用。

第一轮必须做参数预算匹配。若 Sender/Receiver 维度均为 \(d\)，低秩宽度为 \(b\)，每层近似参数量为：

\[
P_l\approx b(d_R+d_P+d_P+d_R)=4bd.
\]

选择层数 \(|\mathcal L|\) 和 \(b\) 时，应令总参数量接近当前 Learned bridge，而不是照搬 XBridge 的约 264M 参数。

## 3. Grounding 与 latent 的职责分离

Generator 已直接接收原图，因此不再次传完整图片特征。最终方法可在输入层保留简洁、可核验的 grounding anchor：

\[
X_R^{(0)}=[E_R(x);E_R(g);E_R(\text{caption instruction})],
\]

其中 \(g\) 只包含人物、物体、动作、空间关系等视觉事实。continuous memory 只承载：

\[
M_P=[H_P^{conflict};H_P^{local};H_P^{global}].
\]

这使离散 channel 负责“谁/什么”，latent channel 负责“关系、矛盾和为什么可能好笑”。必须保留以下 controls：

- full-plan text；
- budget-matched text；
- 当前 24-slot Learned/Typed prefix；
- anchor-only（移除 latent enrichment）；
- latent-only（移除 text grounding，仅作机制诊断）；
- shuffled conflict/association memory。

当前 engineering implementation 先采用更严格的 `zero-latent-prefix`：输入仍保留图片和 Generator SFT 时的任务指令，但没有 plan text、grounding text 或伪 latent token。字面意义的空 prompt 会把冻结 Generator 推到与 SFT 不同的任务分布，因此不作为默认方案；`text grounding anchor` 在通过机制 gate 后作为下一项受控 ablation 加入。

## 4. 训练目标：架构与 loss 是正交选择

cross-attention 是融合架构，caption NLL 是训练目标，二者不能作为互斥选项。新结构仍先用 frozen Planner + frozen Generator，只更新 cross-attention bridge：

\[
\mathcal L_{caption}=-\frac1{|y|}\sum_t\log p_R(y_t\mid x,g,M_P,y_{<t}).
\]

为避免仅学到 generic humor steering，加入 receiver-native text teacher、表示对齐和同图反事实：

\[
\mathcal L=
\mathcal L_{caption}
+\lambda_{KL}\mathcal L_{text\text{-}teacher}
+\lambda_{align}\mathcal L_{InfoNCE}
+\lambda_{cf}\mathcal L_{counterfactual}.
\]

其中同图反事实保持 grounding 不变，只替换 conflict/association：

\[
M_i^+=(c_i,a_i),\qquad M_i^-=(c_j,a_j),
\]

\[
\mathcal L_{counterfactual}
=\operatorname{softplus}
\left[-s(y_i\mid x_i,g_i,M_i^+)+s(y_i\mid x_i,g_i,M_i^-)+m\right].
\]

表示对齐以冻结 Generator 对 text plan 的 native representation \(T_i\) 为 teacher：

\[
\mathcal L_{InfoNCE}
=-\log\frac{\exp(\operatorname{sim}(Z_i,T_i)/\tau)}
{\sum_j\exp(\operatorname{sim}(Z_i,T_j)/\tau)}.
\]

## 5. 最小实施方案

### Phase 0：只做 engineering smoke

1. 不改 vision encoder、Planner、Generator 或两套 SFT adapters。
2. cross-attention 只插入 Qwen language backbone，不插入 vision tower。
3. 首选 4 个近似等距层作为候选；具体层号以真实模型层数确定，不预设论文层号最优。
4. 使用低秩 \(b\) 与当前 Learned bridge做参数预算匹配。
5. 验证 image placeholder、3D MRoPE、padding、gradient isolation、finite update 和峰值显存。
6. gate 使用保守初始化，并记录每层 \(\tanh(\alpha_l)\)、attention entropy 和 bridge-output norm。
7. 真实单步 smoke 严格最多两条：固定 pilot subset 中原始像素最大的图片，以及完整 latent memory 最长的 trace；若为同一 cluster 则只运行一条。

当前预算匹配实现采用 Qwen hidden size 3584、层 \([6,13,20,27]\)、bottleneck 48、4 heads，约 2.82M 参数；旧 Typed prefix bridge 约 3.04M 参数，相差约 7%，避免用参数量解释收益。

### Phase A：语义重建预训练

输入仅包含图片、固定的 recovery task instruction 和 out-of-band full typed memory，不包含任何 plan text 或 pseudo-token。目标是逐字重建：

```text
<CONFLICT>...</CONFLICT>
<LOCAL>...</LOCAL>
<GLOBAL>...</GLOBAL>
```

这一步使“语义可以被冻结 Generator 从 latent 中读回”成为直接训练目标。使用 64 train / 24 validation cluster pilot，最多 5 epochs；配置为 `configs/pilot/cross_attention_semantic_reconstruction.yaml`。

### Phase B：caption adaptation

从 Phase A 最佳 bridge 权重初始化，但重建 optimizer；Generator 使用正常 image + caption task instruction，输入序列中 latent token 数严格为零。配置为 `configs/pilot/cross_attention_caption.yaml`。

正式 loss 包括 caption NLL、text-teacher KL 和 matched/shuffled counterfactual。Phase A 使用四个梯度累积样本组成真实 contrastive batch，只保留小型 alignment graph，并强制启用 receiver-native symmetric InfoNCE；batch=1 时禁止伪造没有 negatives 的 InfoNCE。若后续显存策略迫使 accumulation=1，才切换为带 cluster 过滤的 detached teacher queue。

### Phase C：训练强度 gate

`scripts/check_semantic_training_gate.py` 对 validation 强制检查：

- `matched_minus_shuffled_logp >= 0.02`；
- `fraction_gap_gt_0 >= 0.60`；
- reconstruction/caption NLL 达到配置中的最小改善；
- cross-attention residual update 不超过 25%；
- learned gate 非零；
- 所有指标 finite。

任何一项失败均退出码 2，禁止自动进入下一阶段。该阈值是 pilot 工程/机制 gate，不是统计显著性或论文成功标准。

### Phase 1：小型语义使用 pilot

在与当前 pilot 相同的 64 train / 24 early-stop validation clusters、相同 seed、optimizer steps 和视觉预算上比较：

1. current Typed+KL prefix；
2. anchor-only；
3. anchor + budget-matched gated cross-attention；
4. anchor + cross-attention but shuffled memory。

直接保存：

```text
matched_logp
shuffled_logp
matched_minus_shuffled_logp
fraction_gap_gt_0
fraction_gap_gt_margin
InfoNCE retrieval@1
gate value by layer
trainable params
peak GPU memory
tokens/sec
```

### Phase 2：Go/No-Go

只有同时满足以下条件才在 602 train clusters 上扩展：

1. matched-minus-shuffled gap 明显高于当前 prefix，并且多数样本 \(\Delta>0\)；
2. anchor + latent 明显优于 anchor-only；
3. shuffled memory 导致预期退化，证明 latent 被因果使用；
4. outer-validation 的绝对 good rate 不低于 Text-HOMER；
5. 相同预算下优于当前 Learned/Typed prefix；
6. 无 grounding、hallucination、diversity 或延迟不可接受的退化。

否则保留当前 Text-HOMER/Typed-prefix 主线，不进行 full-data cross-attention training。

## 6. 评测与声明边界

pilot 使用未参与 early stopping 的 40 张 outer-validation 图片、共同 3 seeds 和匿名 Group-of-3，只作筛选。正式结论仍必须使用 121 张 adapter-unseen images、Group-of-10、镜像 A/B、多评审、image-clustered bootstrap 和绝对 good/weak/bad。

若成功，论文可声明为“XBridge-inspired, grounding-anchored latent enrichment for multimodal humor planning”；除非代码、目标、层位、数据和训练协议均逐项一致，不得称为 XBridge 原样复现。

## 7. 权威参考

1. Yang et al. *XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication*. 2026. https://arxiv.org/abs/2608.11676
2. Du et al. *Enabling Agents to Communicate Entirely in Latent Space*. ACL 2026. https://aclanthology.org/2026.acl-long.1248/
3. Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning*. NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html
4. Li et al. *BLIP-2*. ICML 2023. https://proceedings.mlr.press/v202/li23q.html
5. Oord et al. *Representation Learning with Contrastive Predictive Coding*. 2018. https://arxiv.org/abs/1807.03748
6. Shang et al. *HOMER*. ICLR 2026. https://openreview.net/pdf?id=SzaRhPom4o
7. Peng et al. *StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems*. 2026. https://arxiv.org/abs/2608.13317
8. van den Oord et al. *Neural Discrete Representation Learning*. NeurIPS 2017. https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html
9. Mentzer et al. *Finite Scalar Quantization: VQ-VAE Made Simple*. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html
