# Phase A5 修复结果与 caption 阶段解锁记录

## 结论范围

A4 在 40 个 image-cluster 上的 local/grounding gap=`0.0049407572`、
global/association gap=`0.0032089539`，低于预注册的 `0.01` 操作性门槛。
A5 是针对通道混叠、receiver 表示不匹配和 donor 长度混淆的修复候选；它不是
对 A4 的事后改阈值，也不是 HOMER 原论文的逐行复现。

截至本记录，A5 已通过 bridge-only training 和 sealed outer semantic confirmation：
这证明在受控 channel replacement 下，Receiver 的 semantic reconstruction log
probability 对正确/错误 channel 产生稳定差异。它仍不等价于“caption 更好笑”，
caption 结论必须等待本文件所述的独立盲评。

## 实际采用的修复

1. **Channel isolation**：训练和评测目标 channel `c` 时，其他 channel 被 mask；
   训练/评测使用相同协议。
2. **Per-channel receiver projections**：对 `conflict`、`local`、`global` 分别使用
   `K_c/V_c/O_c`，每路独立 softmax，再用 fixed-equal fusion；消除不同长度 memory
   在一个 flat softmax 中竞争。该结构是 multi-source attention 的项目变体。
3. **Length-matched donors**：counterfactual donor 首先按该 channel 的 token 长度
   匹配，并排除自身 cluster；验证 donor 只来自 validation split，保留 donor length
   delta 供审计。
4. **Receiver-native target alignment**：冻结 Receiver 在 image-free typed text
   channel 上的 target-span final hidden 作为 teacher，加入 cosine alignment；只作
   辅助目标，不替代下游反事实 gap。
5. **信息约束**：实际调用 symmetric InfoNCE、variance floor、matched reconstruction、
   donor-side reconstruction 和 channel-isolated counterfactual loss。
6. **Zero-bridge control**：outer evaluator 同时记录 zero bridge，验证通信残差为零时
   gap 为零。

配置与代码：

```text
configs/pilot/cross_attention_semantic_phase_a5.yaml
src/humor_generator_v35/latent/cross_attention.py
src/humor_generator_v35/training/cross_attention_bridge.py
src/humor_generator_v35/training/formal_bridge.py
scripts/train_bridge.py
```

## Bridge training 结果

- GPU job：`6712327`，只训练 bridge；两个 Qwen2.5-VL-7B policy 和 SFT adapter 冻结。
- 数据：602 train clusters、64 validation clusters；5 epochs、755 global steps。
- 选择规则：按 validation total 选择 checkpoint，不按最后 epoch 强行选择。
- 选中：epoch 5，`best_validation_total=3.411963142082095`。
- 验证集（64 clusters，image-clustered bootstrap）：

| channel | mean matched−counterfactual gap | positive-gap fraction | retrieval@1 | 95% CI |
|---|---:|---:|---:|---:|
| conflict | 2.1116899895 | 0.984375 | 0.765625 | [1.7000333, 2.5300903] |
| local/grounding | 0.5140696196 | 0.953125 | 0.937500 | [0.3853335, 0.6611080] |
| global/association | 1.6163532017 | 0.906250 | 0.812500 | [1.2039101, 2.0593347] |

`semantic_gate.json` 的状态为 `strong_go`；三个通道均满足 gap≥0.01、positive
fraction≥0.60、retrieval@1≥0.125、donor reconstruction improvement，以及 frozen
policy、fixed-equal mass 和 bounded update 检查。

## Sealed outer semantic confirmation

- job：`6712454`；held-out traces：`121/121`，trace validator `pass`。
- 统计单位：image cluster；seeds：`20260901/20260902/20260903`。
- 记录：1089（121 clusters × 3 channels × 3 seeds）。
- outer status：`outer_semantic_go`。

| channel | mean gap | positive-gap fraction | 95% image-cluster bootstrap CI |
|---|---:|---:|---:|
| conflict | 2.1176776866 | 1.000000 | [1.8307440, 2.4129452] |
| local/grounding | 0.4252945847 | 0.950413 | [0.3462844, 0.5085145] |
| global/association | 1.6927075682 | 0.975207 | [1.4006081, 1.99284699] |

zero-bridge control 的三通道 mean gap 均为 `0.0`。该结果支持如下有限结论：

> A5 的 receiver-driven latent 在未参与训练的 121 个 image clusters 上具有稳定的
> channel-specific semantic causal sensitivity。

不能由此推出 humorous caption 的 good rate、win rate 或文化幽默效果已经提升。

原始证据：

```text
outputs/pilot/cross_attention_semantic_phase_a5_lengthmatched/semantic_gate.json
outputs/outer_semantic_confirmation/a5_outer121/summary.json
outputs/outer_semantic_confirmation/a5_outer121/validation_contract.json
```

## Caption 阶段

outer gate 通过后才提交 `a5_joint_caption_generation.pjm`（job `6712460`）。它固定：

- 同一 Qwen2.5-VL-7B revision、SFT adapter 和 HOMER traces；
- internal_test 97 + official_hia_unseen_test 24 = 121 图片 clusters；
- 每个条件 10 个固定 generation seeds；
- `Text-HOMER`、`full-plan text`（公平文本控制）和 `A5 receiver cross-attention latent`；
- 不对失败 seed 重采样；逐条 flush JSONL，支持安全续跑；
- 生成完成后构造镜像 Group-of-10 packet。

caption job 只生成候选和匿名 packet，不自动把任何 heuristic 分数写成 judge 结果。
正式评价应把 JSONL 接入 Caption-judgement：先 `adapt/validate/audit`，再由至少三名
独立评审（LLM 或目标文化人类）填写 `overall`、`best_pick`、逐候选
`good/weak/bad` 及 grounding/humor/originality/specificity 维度，最后做镜像折叠、
image-cluster bootstrap、配对检验和 Holm 校正。

### Caption 生成与评测闭环（已完成生成，等待盲评）

作业 `6712460` 已完成并通过逐条审计：

- `internal_test`：97 个 image clusters；`official_hia_unseen_test`：24 个 clusters；
- 条件：`text_homer`、`full_plan_text`（公平文本控制）、`a5_typed`（A5 receiver
  cross-attention）；每个条件和图片使用相同的 10 个固定 seeds；
- 生成记录：`3,630 = 121 × 3 × 10`，无空输出、重复 `(cluster, seed)` 或图片哈希
  不一致；每个 system 的 unique-rate、长度、模板率已写入 `audit.json`；
- Caption-judgement 规范化后系统覆盖为
  `sft::text_homer`、`sft::full_plan_text`、`sft::a5_typed`；
- 两个比较各生成 121 个图片单位的 A/B 镜像，共 `484` 个 Group-of-10 packet、
  `242` 个 mirror pairs；packet blinding audit 通过（system leak=0，malformed
  mirror=0）；
- Caption-judgement 自身测试：`6 passed`。

规范评测运行目录（生成物在 `.gitignore` 下）为：

```text
outputs/caption_judgement/a5_joint_group10_20260905/
```

其中 `generations.jsonl` 是 adapter 后的标准输入，`blind_packets.jsonl` 是公开给
评审的文件，`private_mapping.jsonl` 和 `blind.secret` 只能留在可信机器；
`judge_prompts.jsonl`、`judge-1.json`～`judge-3.json` 是独立评审接口模板，
`audit.json`、`diversity.json` 和 `provenance.json` 是审计产物。`provenance.json`
固定了 job id、Git commit、A5 config/checkpoint/trace-index SHA-256、所有输入输出
哈希和评测状态。

三个模型的直接投喂说明见
[`LLM_RATER_PROMPTS_GROUP10_ZH.md`](LLM_RATER_PROMPTS_GROUP10_ZH.md)；三者使用同一
规范 rubric，仅使用不同 `rater_id`，避免评分标准成为额外变量。

**状态边界：**生成和工程审计已完成，但 `judge-*.json` 仍是空白模板。必须收到至少
三名独立评审（固定 provider/model/version/date、temperature=0、prompt hash）后，才
能运行 aggregate 并报告 Group-of-10 win rate、absolute `good/weak/bad`、图片级
bootstrap CI 和 seed 方差。当前不能把 heuristic 的 unique-rate/模板率或 A5 semantic
gap 当作 humorous-caption improvement。

## 方法依据与证据边界

- **BLIP-2**：冻结视觉/语言模型、先表示对齐再做冻结 LLM 下游生成，支持 bridge-only
  两阶段顺序，但不证明本项目组合最优：[Li et al., ICML 2023](https://proceedings.mlr.press/v202/li23q)。
- **HistAlign**：指出 memory/current hidden misalignment 会削弱 context dependency，
  支持 receiver-native target alignment：[Wan et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.179/)。
- **Flamingo**：冻结 LM 中插入 gated cross-attention 的架构背景；A5 的 layer-output
  residual hook 不是其原样实现：[Alayrac et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)。
- **Multi-source attention**：提供 flat/hierarchical 多源融合的背景；A5 的三路独立
  attention 是受其启发的工程变体：[Libovický & Helcl, ACL 2017](https://aclanthology.org/P17-2031/)。
- **CPC/InfoNCE**：支持 matched positive 与受控 negative 的表示对比学习，但不能单独
  保证 Receiver 因果使用 channel：[van den Oord et al., 2018](https://arxiv.org/abs/1807.03748)。
- **HOMER**：提供 grounding/conflict/local-global imagination 的 humor-caption
  任务分工；A5 的 latent gate 是新增通信审计，不是 HOMER 原训练目标：
  [Shang et al., ICLR 2026](https://arxiv.org/abs/2602.06423)。
- **统计评测**：caption 阶段采用 image-cluster paired bootstrap 与独立评审，原则参考
  [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/) 和
  [Peyrard et al., ACL 2021](https://aclanthology.org/2021.acl-long.179/)。

## 下一步判定

1. 已完成 `6712460` 并校验六个 generation JSONL 完整覆盖 `121 × 10`；
2. 已用 Caption-judgement 生成独立 judge prompt/rating templates；private mapping 不
   发送给评审。
3. 收到至少三份带 provider/model/version/prompt hash 的盲评后，才报告 good-caption
   rate、win rate、CI 和 seed/image-cluster 方差。
4. 若 caption 质量没有改善，结论应是“semantic causal gate 通过但下游 humor benefit
   未证实”，而不是继续增大同一 bridge 或启动 DPO。
