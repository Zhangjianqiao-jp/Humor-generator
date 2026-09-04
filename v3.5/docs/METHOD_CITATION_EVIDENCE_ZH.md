# v3.5 方法—证据—引用台账

更新时间：2026-09-04

本台账用于防止把“代码已实现”“论文支持的原则”“本项目提出的变体”和“尚未运行的计划”混写。它是方法审计的一部分，不是实验结果论文。除非本文件把状态标为“已运行并有 artifact”，否则不得在报告中写成已经验证的结果。

## 1. 证据等级

| 等级 | 含义 | 可以怎么写 |
|---|---|---|
| E0 | 代码、配置和可追溯 artifact 均存在，并通过对应 smoke/test | “本项目已实现/已运行” |
| E1 | 有外部论文或官方实现支持一般方法，但本项目的参数、接口或数据不同 | “受某方法启发/采用其原则”，不能写“复现” |
| E2 | 本项目新增的实验协议或组合，没有一篇论文验证完全相同的算法 | “本项目提出的变体/诊断” |
| E3 | 只写进计划、尚未运行或当前证据不足 | “待验证/未运行”，不能写可行、有效或有收益 |
| X | 当前不允许作为论文结论 | 明确 No-Go 或仅作控制 |

## 2. 当前主线逐项核对

| 方法/组件 | 代码、配置或 artifact | 状态 | 外部依据 | 证据边界 |
|---|---|---|---|---|
| HOMER 的 description→conflict→imagination→retrieval→caption 文本流程 | `src/humor_generator_v35/homer/`、`docs/HOMER_REPRODUCTION_LEDGER_ZH.md`、666 条 trace | E0/E1 | Shang et al., HOMER 论文（[arXiv HTML](https://arxiv.org/html/2602.06423)；[OpenReview](https://openreview.net/forum?id=SzaRhPom4o)）及[官方实现](https://github.com/Shang-hub/HOMER-Official-Implementation) | 当前是方法/数据流程复现；论文没有公开可核验的原始 Qwen-VL 不可变 revision，因此不能声称权重级 exact reproduction。HOMER 的 caption prompt 使用选出的 `D/C/P/Ω`，但没有逐样本 channel-causal loss；这是我们 latent extension 的新增要求。 |
| Qwen2.5-VL 7B + frozen SFT receiver | `configs/pilot/cross_attention_semantic_phase_a4.yaml`、adapter manifest | E0（依赖记录） | [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923) | 这是固定基座和工程依赖，不是“Qwen 论文方法的复现结果”。 |
| StateBridge alignment baseline | `src/humor_generator_v35/latent/bridges.py`、formal bridge configs | E0（实现）/E1（原则） | Peng et al., [StateBridge](https://arxiv.org/abs/2608.13317) 及[官方代码](https://github.com/YanwenPneg/StateBridge) | 我们的三通道、24-slot、caption 接口不是 StateBridge 原始实验设置；只能称为 StateBridge-inspired/adapted baseline。 |
| Learned continuous bridge | `TypedLatentBridge`/`LearnedLatentBridge` | E0（工程实现）/E1（分阶段 bridge 原则） | [BLIP-2](https://proceedings.mlr.press/v202/li23q) 的冻结模型+轻量 bridge；[Interlat](https://aclanthology.org/2026.acl-long.1248/) 的 learned latent compression/alignment | “三通道 humor bridge + 当前层位 + 当前 loss”没有直接论文验证；属于 E2 组合，收益必须由本项目实验测量。 |
| Typed bridge（conflict/local/global type） | `src/humor_generator_v35/latent/cross_attention.py` | E0（工程实现）/E2（方法变体） | Multi-source attention 可作一般架构背景（[Libovický & Helcl 2017](https://aclanthology.org/P17-2031/)） | 该论文不验证我们的 typed latent channel，也不证明类型注入有效。 |
| Receiver-driven cross-attention | `ReceiverDrivenCrossAttentionBridge` | E0（工程实现）/E1（一般原则） | [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html) 证明 LM 中插入 gated cross-attention 的可行性 | 当前通过 decoder layer output hook 做 residual `h⁺=h+Δ`，不是 Flamingo block 的逐行复现；论文不支持当前 hook 方案的效果。正式名称应为“receiver-driven layer-output residual cross-attention adapter”。 |
| zero-prefix / out-of-band memory | `zero_prefix_caption_messages()`、bridge hook | E0（接口控制）/E2（本项目协议） | 无需声称来自某论文；与 receiver 保留原 SFT task prompt 的工程控制一致 | 它是减少 prompt/interface shift 的 control，不是独立算法贡献。 |
| target-only channel isolation | A4 `channel_visibility=target_only`、`active_channels=(channel,)` | E0（已运行）/E2（因果诊断） | [Interlat](https://aclanthology.org/2026.acl-long.1248/) 的 conditional separation / perturbation-sensitive evaluation 提供原则背景 | exact mask、三 channel 任务和 donor protocol 是本项目变体；不能说 HOMER 或 Interlat 原样采用。 |
| matched↔counterfactual channel swap margin | `_forward_channel_metrics()`、A4 `matched_shuffled_margin` | E0（loss 路径已 smoke/训练）/E2（协议） | Interlat 论文支持用任务损失与条件分离检查 latent 是否被利用 | 该 margin 只是一阶行为约束；当前 24-cluster A4 的 CI 不足以证明因果使用。 |
| donor-side reconstruction | A4 `counterfactual_reconstruction` | E0（路径已运行）/E2（协议） | 受 conditional separation / information-preservation 原则启发（[Interlat](https://aclanthology.org/2026.acl-long.1248/)） | 没有论文验证当前“替换一个 channel 后重建 donor field”的精确损失；应称为本项目的 anti-shortcut control。 |
| output-distribution JS/conditional-separation loss | 仅在审计中作为后续 ablation 提议，当前代码未实现、A4 未运行 | E3（未运行） | conditional separation 可参考[Interlat](https://aclanthology.org/2026.acl-long.1248/) | 不能预先声称 JS 会改善结果；若实现，必须同时固定 matched NLL、donor reconstruction 和随机种子，防止只制造分布扰动。 |
| Channel-wise independent attention + fixed equal fusion | `_attend_channel()`、`channel_fusion=fixed_equal` | E0（已实现/测试）/E2（结构控制） | Multi-source attention 提供一般背景（[Libovický & Helcl 2017](https://aclanthology.org/P17-2031/)） | 独立 softmax 是为了消除长度竞争的工程设计；未由外部论文证明对 humor latent 最优。 |
| InfoNCE / symmetric InfoNCE | `training/losses.py`、`train_bridge.py` 实际调用 | E0（已调用）/E1（表示学习目标） | [CPC / InfoNCE](https://arxiv.org/abs/1807.03748) | 只说明 matched representation 的对比学习目标；不能单独证明 receiver 使用 channel，也不能替代下游 caption/反事实结果。 |
| Variance anti-collapse term | `variance_floor_loss()` | E0（已调用）/E1（正则思想） | [VICReg](https://arxiv.org/abs/2105.04906) | 当前只实现 variance floor，不是完整 VICReg（没有其完整 invariance/covariance 组合）；因此不要称为 VICReg 复现。 |
| Text-teacher forward KL | `text_teacher_forward_kl()`；caption configs 中可配置，A4 的权重为 0 | E0（代码）/E3（A4 未使用、caption 未运行） | [Knowledge Distillation](https://arxiv.org/abs/1503.02531)；冻结视觉/语言模块的分阶段思想见[BLIP-2](https://proceedings.mlr.press/v202/li23q) | KL 只能保持 teacher 行为，不能保证 latent 被使用；A4 当前没有 KL 结果。 |
| Contextual teacher | `_contextual_teacher()` | E0（工程路径）/E2（当前实现） | Receiver-native alignment 的一般原则可参考[Interlat](https://aclanthology.org/2026.acl-long.1248/) | 当前 teacher pooling 是整段 prompt 的 receiver hidden，再过初始化时复制的 query projection；该投影不是训练得到的语义 projector。因此 retrieval 不能被写成 semantic proof。 |
| Fixed random query projection in InfoNCE teacher | `alignment_teacher_projection` | E0（事实）/X（解释限制） | 没有论文依据它能成为语义坐标 | 只能称 stationary auxiliary coordinate；正式方法应改为 train-only receiver-native projector/target-span representation，或把该项降级为 identity diagnostic。 |
| Post-token communication state | `qwen_backend.py`、trace index `teacher_forced_post_token` | E0（事实）/E2（协议） | [Interlat](https://aclanthology.org/2026.acl-long.1248/) 讨论 predictor state | 当前不是 Interlat 的 predictor-state 原样复现；论文/报告必须用 post-token 名称，或新增 predictor-state ablation。 |
| TF-IDF hard-negative donor | `training/formal_bridge.py::hard_negative_cluster_map` | E0（工程）/E2（匹配启发式） | 无需冒充论文方法 | 当前 A4 训练没有按 channel token length 匹配，仍有长度混淆；修复前不能把 donor gap 解释为纯语义差异。 |
| Zero-bridge baseline | 尚未接入 A4 gate | E3（缺失 control） | 标准 ablation/control，不需方法论文 | 必须在下一轮或 outer evaluator 中记录 `z=0`；epoch-1→epoch-5 改善不能代替 zero-bridge 比较。 |
| Image-clustered paired bootstrap | `check_semantic_training_gate.py`、评测计划 | E0（实现/计划）/E1（统计原则） | [Dror et al. 2018](https://aclanthology.org/P18-1128/)；[Peyrard et al. 2021](https://aclanthology.org/2021.acl-long.179/) | cluster 作为统计单位是本项目对重复 caption/同图相关性的合理扩展，不是两篇论文的逐行复现；需预注册 cluster-level aggregation。 |
| 64/24 semantic pilot | A3/A4 configs | E0（已运行）/E2（资源 pilot） | 功效与显著性讨论可参考[Card et al. 2020](https://aclanthology.org/2020.emnlp-main.745/) | 没有论文证明 64/24 足以证明本任务的 latent 因果使用；只能发现工程错误、方向性信号或明显退化。 |
| Text-HOMER / C-text+A-latent / C-latent+A-text / All-latent caption ablation | `docs/WORK_PLAN_ZH.md` 第 12 节；对应 caption configs | E3（尚未解锁） | HOMER 文本系统见[HOMER](https://openreview.net/forum?id=SzaRhPom4o)；latent bridge 原则见[Interlat](https://aclanthology.org/2026.acl-long.1248/) 与 [BLIP-2](https://proceedings.mlr.press/v202/li23q) | 这些是尚未运行的本项目比较矩阵；不得写成“latent 提升 caption”。 |
| Group-of-3/10, mirror A/B, good/weak/bad, five-shot calibration | `docs/GROUP3_JUDGE_PROMPT_ZH.md`、评测脚本 | E0（协议/脚本）/E2（评测变体） | [Humor in AI](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html) 支持大规模 cartoon-caption preference benchmark；配对统计见[Peyrard et al.](https://aclanthology.org/2021.acl-long.179/) | 论文不等于“官方 Group-of-10 + 镜像 + good/weak/bad”协议；必须称为 paper-aligned adaptation，并报告 rater/model/prompt provenance。 |
| Diversity metrics | `evaluation/` 及计划 | E0（部分实现）/E3（主结果未运行） | [Tevet & Berant 2021](https://aclanthology.org/2021.eacl-main.25/)；[Vendi Score](https://arxiv.org/abs/2210.02410)；Humor in AI 官方数据/代码 | lexical diversity 不能替代 good-caption angle coverage；在质量结论之前不得宣称 latent 提升多样性。 |
| DPO / preference learning | v2.5 历史目录 | X（当前禁用） | 相关论文不属于当前 v3.5 方法 | 在 latent semantic gate 和 caption held-out 结果之前不启动，不能用 DPO 补救尚未证明的通信机制。 |

## 2A. 旧 caption pilot artifact 的证据边界

`outputs/pilot_validation/` 下的 8 个生成文件和 880 个 Group-of-3 packet 来自旧的
visual-caption pilot（manifest 中的代码 commit 为 `cdcfbbde...`），早于 A4
channel-isolated protocol。它们可以用于记录工程问题，但不是 A4 的验证结果：没有
通过 A4 semantic gate，manifest 也没有当前要求的 checkpoint/evaluator provenance；
`token_embedding`、`statebridge`、`typed_quantized` 中存在空输出或乱码。因此当前没有
可被引用的“新方法 good-caption rate”。任何绝对 good rate 必须来自 A4 outer 通过后的
新生成和至少两名带完整 provenance 的盲评者；不能用旧 packet 或单一自动 judge 替代。

## 3. 当前真实结果与允许结论

当前唯一可引用的 A4 结果来自：

```text
outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1/complete.json
outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1/semantic_gate.json
```

事实是：5 epochs/80 steps、bridge-only、冻结 7B；overall matched−counterfactual log-probability gap 为 `0.0064997`，三个 channel 分别为 `0.0088532 / 0.0051597 / 0.0054863`；conflict 的 cluster-bootstrap 95% CI 跨 0，gate 状态为 `pilot_inconclusive`，engineering gate 为 true。

因此只能写：

1. A4 的工程路径执行完成，target-only mask 和 donor reconstruction 路径被运行；
2. 24-cluster 机制 pilot 尚未证明稳定的 channel-causal use；
3. 尚未生成 caption-level latent vs text 的质量结果；
4. 尚未有任何证据支持“latent 优于文本”“某 bridge 最优”或“DPO 应该启动”。

不能写：

- “A4 已证明 latent 有效”；
- “InfoNCE retrieval 证明 Generator 使用了语义”；
- “HOMER 原样使用了我们的 channel gate”；
- “当前 hook adapter 复现了 Flamingo/Interlat/StateBridge”；
- “64/24 的不显著结果证明方法失败”。

## 4. 在下一实验前必须修正的计划项

1. **A4 outer 必须独立实现。** 现有 `scripts/run_outer_semantic_confirmation.py` 是 A3 协议：它要求 `channel_balanced_v3`、未传 `active_channels=(channel,)`，且默认可能包含图像；不能直接用于 A4。
2. **donor 必须按 channel 长度匹配。** 训练和 outer 都需要记录 token-length delta，并预注册 random/easy/hard donor strata。
3. **增加 zero-bridge control。** 以冻结 receiver 且 `z=0` 为基线，分别报告 NLL、matched log-probability、counterfactual gap。
4. **修正 alignment teacher 的名称和实现。** 在 train-only states 上拟合并冻结 receiver-native projector，或只做 target-span pooling；在修正前，InfoNCE 只能是 auxiliary identity diagnostic。
5. **重命名 A4 日志字段。** `caption_nll` 应显示为 `semantic_reconstruction_nll`；A4 不是 caption 任务。
6. **预注册主指标。** 先写 primary effect、cluster bootstrap、三 channel joint rule 和多重比较规则，再读取 outer 结果；不能看到结果后调阈值。
7. **区分 post-token 与 predictor-state。** 当前 trace 不是 Interlat 的原始 predictor-state；若论文要比较 Interlat，必须单独实现或明确这是不同协议。

## 5. 权威参考注册表

1. Shang, Sun, Ma, Huang. *On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation*, ICLR 2026. [OpenReview](https://openreview.net/forum?id=SzaRhPom4o)；[arXiv](https://arxiv.org/abs/2602.06423)。
2. Du et al. *Enabling Agents to Communicate Entirely in Latent Space*, ACL 2026. [ACL Anthology](https://aclanthology.org/2026.acl-long.1248/)。
3. Peng et al. *StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems*, COLM 2026. [arXiv](https://arxiv.org/abs/2608.13317)。
4. Li, Li, Savarese, Hoi. *BLIP-2*, ICML 2023. [PMLR](https://proceedings.mlr.press/v202/li23q)。
5. Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)。
6. van den Oord, Li, Vinyals. *Representation Learning with Contrastive Predictive Coding*, 2018. [arXiv](https://arxiv.org/abs/1807.03748)。
7. Bardes, Ponce, LeCun. *VICReg*, ICLR 2022. [arXiv](https://arxiv.org/abs/2105.04906)。
8. Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network*, NeurIPS 2015 workshop. [arXiv](https://arxiv.org/abs/1503.02531)。
9. Dror et al. *The Hitchhiker’s Guide to Testing Statistical Significance in NLP*, ACL 2018. [ACL Anthology](https://aclanthology.org/P18-1128/)。
10. Peyrard et al. *Better than Average: Paired Evaluation of NLP systems*, ACL-IJCNLP 2021. [ACL Anthology](https://aclanthology.org/2021.acl-long.179/)。
11. Card et al. *With Little Power Comes Great Responsibility*, EMNLP 2020. [ACL Anthology](https://aclanthology.org/2020.emnlp-main.745/)。
12. Libovický & Helcl. *Attention Strategies for Multi-Source Sequence-to-Sequence Learning*, ACL 2017. [ACL Anthology](https://aclanthology.org/P17-2031/)。
13. Tevet & Berant. *Evaluating the Evaluation of Diversity in Natural Language Generation*, EACL 2021. [ACL Anthology](https://aclanthology.org/2021.eacl-main.25/)。
14. Friedman & Dieng. *The Vendi Score*, TMLR 2023. [arXiv](https://arxiv.org/abs/2210.02410)。
15. Bai et al. *Qwen2.5-VL Technical Report*, 2025. [arXiv](https://arxiv.org/abs/2502.13923)。
