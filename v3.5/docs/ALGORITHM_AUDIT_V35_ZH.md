# v3.5 方法与算法审计（2026-09-04）

逐项的“方法—代码—证据—引用”映射见 [`docs/METHOD_CITATION_EVIDENCE_ZH.md`](METHOD_CITATION_EVIDENCE_ZH.md)。本审计遵守证据分层：代码通过不等于科学假设成立，文献提出一般原则也不等于本项目变体已被验证。

## 结论先行

当前方案在工程上可执行，在数学上没有发现 bridge 反传符号或 channel mask 的直接错误；但它还不能支持“latent 已被 Generator 因果使用”或“latent 优于文本”的论文结论。

| 层次 | 当前结论 | 依据 |
|---|---|---|
| 工程 | GO | `2846/2846` 数据行、`666/666` trace、冻结 receiver、无 NaN/OOM；单元测试通过 |
| A4 语义机制 | `pilot_inconclusive` | overall gap `0.00650`；conflict/local/global 为 `0.00885/0.00516/0.00549` |
| 因果使用 | 尚未证实 | conflict 的 image-cluster bootstrap CI 跨 0；A4 仍是 24-cluster pilot |
| caption 质量 | NO-GO（尚未开始） | A4 是无图像 semantic-recovery，不是 caption 训练/生成 |
| Preference/DPO | 禁止 | latent 机制和下游 caption 收益尚未完成验证 |

证据文件：

```text
outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1/complete.json
outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1/semantic_gate.json
outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1/metrics.jsonl
```

## 已确认合理的部分

### 1. A4 的 channel-isolated 目标是可识别的机制实验

对目标 channel `c`，当前 receiver 只看到该 channel 的 typed memory：

\[
K_c=W_K\operatorname{LN}(H_c+t_c),\qquad
V_c=W_V\operatorname{LN}(H_c+t_c),
\]

\[
A_c=\operatorname{softmax}\left(\frac{QK_c^\top}{\sqrt{d_k}}\right),\qquad
z=W_O(A_cV_c),\qquad h^+=h+\tanh(g)z .
\]

`active_channels=(c,)` 将其他 channel 的 attention probability 置零，避免 A3 中“其他 channel 或图片替目标 channel 解释”的可识别性问题。fixed-equal channel mass 在 A4 中也确实保持 one-hot target mask，而不是让 learned router 把某一路权重降为零。

### 2. margin 反传的符号正确

当前使用：

\[
L_{cf}=\operatorname{softplus}(-m+c+\gamma),
\quad m=\log p(s_c\mid M_c),\quad c=\log p(s_c\mid M'_c).
\]

其梯度为 \(\partial L/\partial m=-\sigma(-m+c+\gamma)\)、\(\partial L/\partial c=+\sigma(-m+c+\gamma)\)。代码将这一系数 detach 后分两次 forward/backward，保留了一阶梯度并避免保存两份 frozen-VLM activation；这部分通过了现有 loss tests。

### 3. donor reconstruction 关闭了“只压低 matched”的漏洞

A4 还要求：

\[
L_{donor}= -\log p(s'_c\mid M'_c),
\]

因此 swap 后的 memory 要能恢复 donor field，而不能只让原 field 变差。这是正确的必要约束，但不是充分的因果使用证明。

## 必须修正或限制解释的部分

### 1. 当前 outer evaluator 不是 A4 evaluator

`scripts/run_outer_semantic_confirmation.py` 明确要求 `channel_balanced_v3` 和 A3 checkpoint；`evaluate_channel()` 调用 `task._logits()` 时没有传 `active_channels=(channel,)`，并且构造 `ReceiverCrossAttentionTask` 时未显式关闭 `semantic_prompt_include_image`。因此它只能作为 A3 的旧 outer 脚本，不能直接确认 A4 的 channel-isolated protocol。

**后果：**如果直接用该脚本，替换某一路时其他两路仍能补偿，且图片可能成为 shortcut，所得 gap 不能与 A4 的 gap 合并。

**修正：**新增/参数化独立的 A4 outer evaluator，强制检查：

```text
semantic_objective = channel_isolated_v4
channel_visibility = target_only
semantic_prompt_include_image = false
checkpoint = A4 best_bridge.pt
```

### 2. A4 训练阶段的 donor 没有按 channel 长度匹配

正式 A4 的 `hard_negative_cluster_map()` 为每个 cluster 选一个 donor，但没有按被替换 channel 的 token length 匹配。由于每个 channel 的 attention 是：

\[
\operatorname{softmax}_{t=1}^{T_c}(qK_c^\top),
\]

当 donor 的 \(T'_c\) 与 target 的 \(T_c\) 不同，gap 同时反映语义替换和候选数量/长度变化。A4 的 `full_typed_states_no_truncation` 使该混淆更明显。

**修正：**outer 和下一次训练都应使用 `target → channel → donor` 映射，先按 `|T'_c-T_c|` 最小化，再在长度窗口内按 source、description similarity 和 conflict-signature mismatch 选 donor；同时报告 length delta、gap-length correlation，并保留 random/easy/hard donor 三个预注册 strata。

### 3. 缺少“零 bridge”基线

当前 gate 的 `nll_improved` 与 donor improvement 是相对于第一个训练 epoch，而不是相对于 `z=0` 的 frozen receiver。随机初始化 bridge 可能已经改变输出，因而 epoch-5 的改善不能单独说明 bridge 比无通信更好。

**修正：**训练开始前对同一 validation clusters 记录：

```text
zero_bridge matched NLL/logp
zero_bridge swapped logp
random-init bridge matched NLL/logp
random-init bridge gap
```

后续 gate 至少报告 `ΔNLL = NLL_zero - NLL_checkpoint` 和 `Δgap = gap_checkpoint - gap_zero`。当前 A4 结果可以保留，但不能把 epoch-1 → epoch-5 写成 method gain。

### 4. InfoNCE teacher 不是可靠的语义坐标

当前 teacher 是 frozen receiver contextual hidden 的均值，再乘以 bridge 初始化时 query weight 的固定拷贝：

\[
T=P_{Q,0}\,\operatorname{mean}(H_R).
\]

这个 \(P_{Q,0}\) 是随机初始化投影，不是 receiver 训练得到的 semantic projector。InfoNCE retrieval 提升说明 student 学会匹配这个固定坐标中的 identity，但不能单独说明“语义被 receiver 读取”。此外 teacher pooling 当前包含整段 prompt，而不只包含 field token。

**修正优先级：**

1. 最小改动：把 InfoNCE 明确降级为 auxiliary identity/alignment diagnostic，不把 retrieval 当 causal gate 的充分条件；
2. 正式版本：在 train-only contextual states 上拟合并冻结 receiver-native projector（或使用 target-span pooling）；
3. 另一个可比较分支：用 receiver output distribution 的 KL/JS 作为 functional alignment，而不是随机投影后的均值。

### 5. 当前 communication state 是 post-token，不是 Interlat 的 predictor state

trace index 明确记录 `teacher_forced_post_token`。对生成 token \(y_t\)，当前传的是已经读入 \(y_t\) 后的位置 hidden；Interlat 的定义是预测 \(y_t\) 之前的 hidden（\(h_{t-1}\)）。因此不能把当前 trace 称为 Interlat 的原样复现。

**修正：**保留 post-token 作为“完成的 Planner message”协议，同时做 predictor-state ablation；论文中分别命名，不能混写。

### 6. A4 的 `caption_nll` 名称会误导

A4 的 semantic prompt 不含图片，target 是 conflict/local/global field，不是 caption。日志中的 `caption_nll` 应在报告中解释为 `semantic_reconstruction_nll`；只有后续 caption stage 才能报告 caption NLL 和 humor/grounding 结果。

### 7. 24-cluster gate 只能作 pilot，阈值不能冒充统计结论

A4 的 CI 和多通道结果说明 pilot 有方向性信号，但 conflict CI 仍跨 0。`0.01` gap、`0.60` positive fraction 和 retrieval 阈值是工程门槛，尚未通过独立 control 或功效分析校准。三路同时判定还涉及多重比较；outer 应预注册 joint/max-t 或明确“所有 channel 均过”的规则。

### 8. 当前 bridge 是 hook-based layer-output residual adapter

代码在 decoder layer 完成后注册 hook 并执行 `h^+=h+z`，不是插入 Transformer block 内部的标准 cross-attention 子层。该形式工程上合理、显存低，但论文中应准确称为 `receiver-driven layer-output residual cross-attention adapter`，不能直接声称复现 Flamingo/标准 cross-attention。由于 sender memory 没有显式位置编码，后续应至少记录 position/slot encoding 是否作为消融因素。

## 修正后的执行顺序

```text
当前 A4 checkpoint 冻结（不重跑同一训练）
→ 修正 A4 outer evaluator：target-only、no-image、A4 checkpoint
→ per-channel length-matched donors + zero-bridge baseline
→ 40 outer clusters × 3 shared seeds；只做 semantic mechanism confirmation
→ 若结果仍 inconclusive，不宣布 latent 失败，按预注册功效分析决定是否扩到 64/97/121 clusters
→ 只有 semantic gate 通过，才用完整 602 train clusters 训练 caption bridge
→ caption stage 比较 text-HOMER、budget text、StateBridge、learned/typed latent 和 hybrid
→ 在 121 个 adapter-unseen images 上用共同 seeds、Group-of-10、多评审和 image-clustered bootstrap
→ latent 下游收益稳定后再讨论 preference learning；当前不启动 DPO
```

`jobs/cross_attention_phase_a4.pjm` 当前在 `pilot_inconclusive` 时退出是合理的 fail-closed 行为，但计划路由需要单独提交修正后的 A4 outer 作业；不能把 A3 outer 脚本当作自动后续步骤。

## 权威依据

- HOMER：Shang et al., *On the Wings of Imagination*, ICLR 2026（论文/附录与公开实现）：[OpenReview](https://openreview.net/pdf?id=SzaRhPom4o)。HOMER 的文本流程和 latent extension 必须分开声明。
- Interlat：Du et al., *Enabling Agents to Communicate Entirely in Latent Space*, ACL 2026：[ACL Anthology](https://aclanthology.org/2026.acl-long.1248/)。其 task loss、conditional separation、alignment 和 predictor-state 定义支持“表示对齐不等于因果使用”的区分。
- StateBridge：Peng et al., *StateBridge*, COLM 2026：[arXiv](https://arxiv.org/abs/2608.13317)。其 closed-form geometry/norm/vocabulary anchoring 与本项目 learned receiver-driven adapter 不是同一方法。
- BLIP-2：Li et al., ICML 2023：[PMLR](https://proceedings.mlr.press/v202/li23q)。冻结大模型、轻量 bridge、先表示对齐再生成训练支持分阶段设计，但不替代本项目的 channel causality 证据。
- InfoNCE/CPC：van den Oord et al., *Representation Learning with Contrastive Predictive Coding*：[arXiv](https://arxiv.org/abs/1807.03748)。InfoNCE 是表示判别目标，不是 downstream causal-use 证明。
- NLP 配对/bootstrapping：Dror et al., *The Hitchhiker’s Guide to Testing Statistical Significance in NLP*：[ACL Anthology](https://aclanthology.org/P18-1128/)；Peyrard et al., *Better than Average: Paired Evaluation of NLP systems*：[ACL Anthology](https://aclanthology.org/2021.acl-long.179/)。支持 image-cluster paired bootstrap，而不是把重复 caption 行当独立样本。
