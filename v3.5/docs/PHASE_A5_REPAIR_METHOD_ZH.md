# Phase A5：local/global 语义通道修复方法

## 结论范围

A4 的 local `0.004940757` 与 global `0.003208954` 是 40 个 image-cluster 上的
matched-minus-counterfactual sequence log-probability gap，均低于 `0.01` 点阈值。
bootstrap 下界为正，但这只表示方向证据，不能证明接收器稳定地使用了相应通道。本文件
定义 A5 作为新的 repair candidate；A4 输出和代码不覆盖，A5 通过 outer gate 前不进入
caption 质量结论。

## 可归因的模型变化

每个 Qwen decoder bridge layer 读取 Planner 的三个 typed memory：

```text
M_c = H_c + e_c,       c ∈ {conflict, local, global}
Q = W_Q LN(h)
K_c = W^c_K LN(M_c),   V_c = W^c_V LN(M_c)
A_c = softmax(Q K_c^T / sqrt(d_h)) V_c
u_c = W^c_O A_c
h' = h + tanh(g) * (u_conflict + u_local + u_global)/3
```

A4 使用共享的 `W_K/W_V/W_O`；A5 为每个角色建立独立的 `W^c_K/W^c_V/W^c_O`，
其余层索引、bottleneck、head 数和 fixed-equal fusion 保持不变。因而 A5 不是“参数量
公平的 placement ablation”，而是针对角色混叠的机制修复；后续若 A5 有效，必须再做
projection-only 与 alignment-only 消融。

在 channel-isolated 训练/评测中，目标 channel `c` 以外的 memory mask 全为 0，且语义
恢复 prompt 不含图片：

```text
p(s_c | M_c) > p(s_c | M'_c)
```

其中 `M'_c` 来自长度匹配、不同 conflict signature 的 donor。额外的 donor reconstruction
要求 `M'_c` 解码 donor 语义，避免只把原目标概率压低。

## Receiver-native target alignment

冻结的 Receiver 在普通文字条件下得到目标字段的 final-layer target-span 表示：

```text
t_c = mean_t Hidden_R(target token_t | <C> s_c </C>)
z_c = mean_t Hidden_R(target token_t | latent M_c)
L_align = 1 - cos(z_c, stopgrad(t_c))
```

总的 A5 语义训练损失为：

```text
L_A5 = L_reconstruction
     + λ_cf L_matched/counterfactual
     + λ_donor L_donor-reconstruction
     + λ_NCE L_symmetric-InfoNCE
     + λ_var L_variance-floor
     + λ_align L_align
```

本实验 `λ_align=0.5`。`t_c` 在训练开始前由冻结 Receiver 缓存，避免 teacher 坐标随
bridge 更新漂移；teacher 只作为辅助 receiver-native 表征目标，不能替代最终的
matched/shuffled causal gap。

## 为什么选择这些修复

- BLIP-2（Li et al., 2023）证明了在冻结大模型时，轻量 bridge 需要先进行表示对齐，
  再验证下游生成可解释性；这支持 `L_align` 与 bridge-only 训练的顺序。
- HistAlign（Wan et al., EMNLP 2023）将 memory/current hidden misalignment 视为
  context dependency 变差的原因，并用对齐训练改善条件生成；这支持使用 Receiver 原生
  target-span，而非任意 sender embedding 作为 teacher。
- Multi-Source Attention（Libovický & Helcl, ACL 2017）比较 flat/hierarchical 的多源
  attention；A5 的 per-channel K/V/O 是对三类 Planner 角色的工程适配，不是该论文的
  原样复现。
- Flamingo（Alayrac et al., NeurIPS 2022）提供冻结语言模型中插入 gated cross-attention
  的架构背景。
- CPC（van den Oord et al., 2018）支持用 matched positives 和受控 negatives 做
  contrastive representation learning；InfoNCE 仍不是 causal-use 充分条件。
- HOMER（Shang et al., ICLR 2026）定义 conflict、local/global imagination 与 caption
  generator 的任务分工；A5 的 channel gate 是本项目为 latent communication 新增的
  因果审计，不声称 HOMER 原方法包含该损失。

## Gate 与后续动作

1. 先运行 real-trace engineering smoke：两条真实 trace、两步以内、Receiver 7B 全冻结、
   每个 channel 的 one-hot mask、梯度/更新有限。
2. 通过 smoke 后，使用全部 602 train / 64 validation clusters 训练 bridge，并记录每
   epoch validation checkpoint。pilot gate 不通过时只称 `pilot_inconclusive`，不称方法
   失败。
3. 另行生成 97 internal-test + 24 official-unseen 的 121 条 sealed Planner traces，
   独立保存 input manifest/hash；不能把 test trace 混入 666 条训练/validation index。
4. Outer 以 image-cluster 为统计单位、3 seeds，报告每 channel gap、每图正 gap 比例、
   donor 长度差相关与 bootstrap CI。只有三个 channel 的点阈值和 CI 都通过，才解锁 caption。
5. Caption 阶段固定同一模型 revision、SFT adapter、HOMER planner traces；每图每条件
   10 个 generation seeds，Text-HOMER 与 A5 latent 做 mirrored Group-of-10。结果需经
   独立多评审盲评，再按 image cluster bootstrap；不能用语义 gap 代替“真正好笑”。

## 限制

A5 同时改变 projection sharing 与 target alignment，因此即使通过，也只能说明该组合
候选有效；不能单独归因到 per-channel projection 或 alignment。`0.01` 是项目预注册的
操作性门槛而非文献定理；任何未达标结果都必须保留原始 artifact 并标注 inconclusive。

## 参考文献

1. Li et al. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, 2023. https://arxiv.org/abs/2301.12597
2. Wan, Zhang, Bansal. *HistAlign: Improving Context Dependency in Language Generation by Aligning with History*, EMNLP 2023. https://aclanthology.org/2023.emnlp-main.179/
3. Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html
4. Libovický and Helcl. *Attention Strategies for Multi-Source Sequence-to-Sequence Learning*, ACL 2017. https://aclanthology.org/P17-2031/
5. van den Oord, Li, Vinyals. *Representation Learning with Contrastive Predictive Coding*, 2018. https://arxiv.org/abs/1807.03748
6. Shang et al. *On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation (HOMER)*, ICLR 2026. https://arxiv.org/abs/2602.06423
