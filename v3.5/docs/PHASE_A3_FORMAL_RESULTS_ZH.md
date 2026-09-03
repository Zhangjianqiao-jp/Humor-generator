# Phase A3 正式语义恢复结果

## 结论

Phase A3 bridge-only formal job `6707953` 已在一张完整 H100 上以 exit code 0 完成。
两个 Qwen2.5-VL-7B policy（Planner 与 SFT Generator）均冻结，只更新 cross-attention
bridge。训练路径、数据/trace provenance、InfoNCE、单通道 counterfactual 和资源检查均
通过；24-cluster semantic gate 的最终状态为 `pilot_inconclusive`，不是技术失败，也不
是 latent 方法类别的最终 No-Go。

因此当前决策是：

```text
允许 → 40 个未参与 early stopping 的 outer semantic confirmation
禁止 → caption bridge、Group-of-3/10 caption 结论、DPO/preference learning
```

## 作业与输入身份

| 项目 | 值 |
|---|---|
| job | `6707953` |
| resource | `b-batch`, `genkai0001`, 1× full `NVIDIA_H100_80GB_HBM3` |
| elapsed / exit | 25:06 / 0 |
| code commit | `28d530769d3af8051696b21d55bf257cdcaa35dc` |
| config | `configs/pilot/cross_attention_semantic_phase_a3.yaml` |
| output | `outputs/pilot/cross_attention_semantic_phase_a3/` |
| config SHA-256 | `7dfd5222a7aa15b2e90b066a08914fd0f8beb49db826922e5dc5860614c5e0a5` |
| dataset manifest SHA-256 | `c26c5999645d01089a937d0cb86e1982ccd4ea737f807fa698e398279a7197d8` |
| trace index SHA-256 | `f4ffa1ef4a39e616ced0d94f1f6334da930d79c726873889aba05a30bfedfcef` |
| train / validation | 64 / 24 image clusters |
| epochs / optimizer steps | 5 / 80 |
| bridge / policy trainable params | 2,820,612 / 0 |

`run_manifest.json`、`preflight.json`、`metrics.jsonl`、每轮 validation JSONL、逐轮
checkpoint、`complete.json`、`semantic_gate.json` 和 scheduler stats 均保留。`complete.json`
为 `status=complete`、`epochs_completed=5`、`global_step=80`。

## 学习曲线

| epoch | train total | validation total | validation caption NLL | validation InfoNCE retrieval@1 |
|---:|---:|---:|---:|---:|
| 1 | 3.41211 | 4.12281 | 1.80835 | 0.125 |
| 2 | 3.01893 | 3.72306 | 1.58233 | 0.326 |
| 3 | 2.64045 | 3.34665 | 1.36420 | 0.410 |
| 4 | 2.37140 | 3.13855 | 1.21956 | 0.514 |
| 5 | 2.22451 | 3.00137 | 1.12946 | 0.535 |

`validation total` 和 NLL 持续下降，三路 contextual InfoNCE 的 retrieval@1 从 0.125
提高到 0.535，说明 bridge 的训练目标确实被优化；这本身不等价于下游 caption 更幽默。

## 通道反事实结果

最终 epoch 的 validation 汇总：

| channel | matched−shuffled gap | fraction(gap>0) | retrieval@1 | image-cluster bootstrap 95% CI |
|---|---:|---:|---:|---|
| conflict | −0.000040 | 0.458 | 0.396 | [−0.002806, 0.002740] |
| local | −0.000317 | 0.375 | 0.563 | [−0.001935, 0.001354] |
| global | 0.001006 | 0.646 | 0.646 | [−0.000262, 0.002396] |

所有 channel 都保持固定等权 mass `1/3`；最大 relative update norm `0.0430`，小于
配置上限 `0.25`。由于三个 gap 的 bootstrap 下界均未超过 0，且 conflict/local 的
point gate 未通过，validator 写出：

```text
status = pilot_inconclusive
engineering_gate_pass = true
```

24 个 cluster 的点估计不足不能被写成“latent 无效”：这个 pilot 只承担机制筛查，
统计功效不足以排除小到中等效应。相反，NLL/retrieval 的改善也不能被写成“latent 已
被 Receiver 稳定使用”或“caption 质量提高”。

## 下一步

1. 固定本轮 checkpoint、配置、seed、manifest 和代码 commit，不再修改阈值。
2. 从同一 validation split 中排除 early-stopping 使用的 24 个 cluster，得到剩余 40
   个 outer cluster；按同一 image-cluster、length-matched、单通道 counterfactual 协议
   做 sealed confirmation，记录 3 个共同 seed（语义 forward 本身是确定性的，seed 方差
   应明确报告为零/近零，而不能伪装成独立样本）。
3. 只有 outer confirmation 在三路上显示稳定且非零的 semantic sensitivity，才解锁
   `Text-HOMER / C-text+A-latent / C-latent+A-text / All-latent` caption 生成。
4. Caption 阶段使用 Group-of-3 仅筛选；论文主结论使用 Group-of-10、镜像 A/B、多评审、
   `good/weak/bad` 绝对标签和 image-clustered bootstrap。

## 权威参考

1. Shang et al., *HOMER*, ICLR 2026: <https://openreview.net/pdf?id=SzaRhPom4o>
2. Du et al., *InterLat: An Interpretable Latent Communication Framework*, ACL 2026:
   <https://aclanthology.org/2026.acl-long.1248/>
3. Peng et al., *StateBridge*, COLM 2026: <https://arxiv.org/abs/2608.13317>
4. Graham, Haddow & Koehn, *Statistical Power and Translationese in Machine Translation
   Evaluation*, EMNLP 2020: <https://aclanthology.org/2020.emnlp-main.6/>
5. Koehn, *Statistical Significance Tests for Machine Translation Evaluation*, EMNLP 2004:
   <https://aclanthology.org/W04-3250/>
