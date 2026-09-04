# v3.5 修正版实验计划

> 本轮完整的算法/协议审计见 [`docs/ALGORITHM_AUDIT_V35_ZH.md`](ALGORITHM_AUDIT_V35_ZH.md)，方法—证据—引用逐项台账见 [`docs/METHOD_CITATION_EVIDENCE_ZH.md`](METHOD_CITATION_EVIDENCE_ZH.md)。文中所有“已实现/已运行”均须能回溯到代码、配置和 artifact；`E1/E2/E3` 方案不能写成已验证结果。审计结论：A4 工程通过但语义机制仍为 `pilot_inconclusive`；在修正 A4 outer、按 channel 匹配 donor 长度并加入 zero-bridge 基线前，不得进入 caption 或 preference learning。

## 0A. A3 gap 诊断后的 A4 修订（2026-09-04）

A3 的真实 channel gap 并不大：overall matched-minus-shuffled log-probability 为
`0.000216`；conflict/local/global 分别为 `-0.000040/-0.000317/0.001006`。此前日志中
约 `0.798` 的 `shuffled_margin` 是
`softplus(margin - gap)` 的损失值，不是语义 gap。A3 因此只能写作
**因果通道使用未证实（pilot_inconclusive）**，不能写作 latent 类方法失败。

A3 的主要可识别性缺陷是：单 channel 目标仍由三路 memory 共同提供，且 semantic
recovery prompt 包含图片；不变 channel 和图像都可以预测目标。固定等权融合还令单路
替换只改变约三分之一的残差。为避免“只把原目标概率压低”而伪造 gap，新增独立的
`configs/pilot/cross_attention_semantic_phase_a4.yaml`，不覆盖 A3：

```text
channel_visibility = target_only
semantic_prompt_include_image = false
counterfactual_reconstruction = 0.75
semantic_objective = channel_isolated_v4
```

对 channel `c`，A4 只允许 receiver 读取 `M_c`：

```text
z_c = B_phi(H_c)
```

matched 条件优化原语义 `s_c`，shuffled 条件同时满足：

```text
log p(s_c | M_c) > log p(s_c | M'_c)
log p(s'_c | M'_c)  较低重建损失
```

第二项由 `counterfactual_reconstruction` 实现，要求替换后的 channel 能解码 donor
语义，而不是产生任意扰动。`src/humor_generator_v35/latent/cross_attention.py` 的
`active_channels` mask 会把未激活 channel 的 attention probability 置零；inactive
channel 全为 mask 时使用有限 score floor，避免全 `-inf` softmax 产生 NaN。A3 默认
`active_channels=None`，保持历史协议可复现。

新增的 `scripts/validate_phase_a4_smoke.py` 会检查：两个真实 trace、冻结 7B、bridge
非零更新、无 NaN、每个 target channel 的 channel weight 为 one-hot、donor reconstruction
路径确实执行。作业 `jobs/cross_attention_phase_a4.pjm` 的顺序为：

```text
locked preflight → CUDA/resource smoke → A4 validator → bridge-only training
→ semantic gate
```

semantic gate 仍只允许 `strong_go` 或 `go_to_outer_semantic_validation` 进入后续验证；
任何 `pilot_inconclusive` 都不会触发 caption 生成。当前不增加 JS/全词表分布分离项，
因为在 frozen 7B 上它会额外占用大块 vocabulary logits，且单独最大化分布差异可能制造
随机行为；A4 先用可解释的 channel isolation + donor reconstruction + 原有 margin。
若 A4 仍失败，再单独比较 JS/conditional-separation ablation，不同时改变多项变量。

2026-09-04 的 outer confirmation 作业 `6708118` 已获得完整节点但在正式 forward 前被
`run_formal_preflight.py` 因 tracked worktree dirty 拒绝，退出码为 1；compile、75 tests、
dataset `2846/2846`、trace `666/666` 均通过。该作业没有产生 outer 结果，也没有改变
A3 科学结论。重新提交前必须形成干净 commit 或从 immutable clean worktree 运行；不得
用 `--allow-dirty` 绕过 provenance guard。

## 0. 当前执行基线（2026-09-03）

本文件当前只管理 **v3.5 latent communication 主线**。旧 v2.5/v3.0 的 DPO、偏好对、
联合训练和 reranker 方案保留为历史证据，但不属于当前可执行计划；任何旧 DPO 脚本、
checkpoint 或 preference 结果都不得被 v3.5 作业自动调用。

当前顺序冻结为：

```text
严格 HOMER Planner traces（666/666 已完成）
→ 修正后的旧 v1/v2 latent 反事实重评（已完成，pilot_inconclusive）
→ Phase A3 replacement engineering smoke（已通过，job 6706516）
→ Phase A3：64 train / 24 validation，只训练 bridge，冻结两个 7B（已完成，job 6707953）
→ Phase A3 gate：engineering pass，semantic pilot_inconclusive
→ A3 outer semantic baseline（job 6708118 在 dirty-worktree preflight 停止；不能代替 A4）
→ A4 channel-isolated functional semantic pilot（real-trace smoke 6711052 已通过；旧 c-batch 作业 6711074 在执行前取消；修正作业 6711109 已完成，gate=pilot_inconclusive）
→ **当前停止：修正 A4 outer evaluator、channel-length donor 与 zero-bridge control**
→ A4 outer semantic confirmation（修正协议，尚未提交）
→ latent/text 混合 caption 消融与盲评（仅当 semantic gate 通过）
→ 只有 latent bridge 有稳定 held-out 收益后，才重新讨论 preference learning
```

这里的“latent 工作”目前是语义通信 bridge 的训练和验证，不是 DPO。A3 smoke 的旧作业
`6689653` 在 forward/backward 前因 `492 > max_target_tokens=384` 的配置错误退出；
配置改为 `768` 后，replacement job `6706516` 已在完整 H100 上以 exit code 0 通过。
这只关闭了执行门禁，不代表 A3 的语义收益或 caption 质量已经得到证明。正式的
`64 train / 24 validation` bridge-only 训练已由 job `6707953` 完成；其工程 gate
通过，但 24-cluster semantic gate 为 `pilot_inconclusive`，因此不能直接进入 caption
bridge 或 preference learning。

截至 2026-09-04，A3 outer evaluator 曾由完整节点作业 `6708118` 启动，但
`run_formal_preflight.py` 因 tracked worktree dirty 在实际 forward 前停止；75 tests、
数据和 trace gate 均已通过，没有生成 outer 结果。该 baseline 必须在 clean commit 后
用新输出目录重提；它与 A4 的协议不同，不能复用为 A4 outer 结果。A4 的新训练不覆盖
该 baseline；MIG 和 DPO 均保持禁用。

### A4 正式作业脚本修正（2026-09-04）

替代 smoke `6711052` 已通过真实 trace、target-only one-hot mask、无图像 semantic
prompt、donor reconstruction、非零 bridge update 和冻结策略检查。随后提交的通用
脚本作业 `6711074` 尚未执行 forward；审查发现它没有调用 A4 专用 semantic gate，且
使用了旧的输出目录，因此在可能产生混淆前主动取消。该事件记录为工程/可追溯性问题，
不构成方法 No-Go，也不覆盖 A4 smoke 证据。

现行 `jobs/cross_attention_phase_a4.pjm` 已满足；修正后的正式作业 `6711109` 已完成：

```text
c-batch + gpu=1（单张完整 native GPU）
独立输出 outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1
locked preflight → CUDA/resource smoke → A4 validator → bridge-only training
→ 动态读取本次输出目录的 semantic_gate.json（完成，状态 `pilot_inconclusive`）
```

本次作业已通过 `.venv/bin/python -m pytest -q`、`git diff --check`、
`scripts/run_formal_preflight.py` 和 A4 validator；semantic gate 为
`pilot_inconclusive`，因此按 fail-closed 规则保留完整日志但未进入 caption 生成。
smoke/训练完成只能证明工程协议可执行和比较条件已建立，不能预先保证 latent 收益；
真正有效性仍由修正后的 A4 outer semantic confirmation 及后续 held-out caption/盲评
共同决定。A4 的完整数值见本文件后面的结果小节和
`outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1/semantic_gate.json`。

`docs/SEMANTIC_REEVALUATION_RESULTS_ZH.md` 是旧 bridge 重评的数值结果；它只能说明
v1/v2 在 24-cluster pilot 上证据不足，不能替代新的 A3 训练。以下各节若与本节的当前
状态冲突，以本节和对应作业的 `complete.json`/gate 文件为准。

## 1. 研究问题与边界

核心问题不是“latent 能否替代文字”，而是：在 Planner 和 Generator 都冻结时，连续通信能否比等信息量文本/离散 token 更准确地传递 `Conflict + Local Association + Global Association`，并提高图片相关幽默 caption 的质量和角度覆盖。

本阶段只训练 bridge。Preference Learning、DPO、联合反传和两个 7B 的参数更新全部禁用。只有 latent 在未见图片上显示稳定收益后，才重新讨论 preference objective。

## 2. 两条不能混淆的系统线

### A. 严格 HOMER 文本线

`standard description -> conflict -> local/global imagination -> retrieval -> selected conflict/path -> caption`

Caption 阶段按 HOMER 公开 prompt 只接收 description/conflict/path。由于论文未公开精确 Qwen-VL revision，本项目只能声明为“固定 Qwen2.5-VL 替代模型的 method/data reproduction”。

必须准确理解 HOMER 的条件使用机制：HOMER 先从候选 conflict 中选择两条，再从
imagination tree 中为两个关键实体各选择一条 path，最后把 `description + selected
conflicts + selected paths` 作为带显式字段名的文本放入 Generator prompt。其 prompt
要求 caption 聚焦 central incongruity 并自然结合 chain keywords。这个流程提高了三个
信息源被读取的可能性，但 HOMER **没有**额外的训练损失、注意力约束或因果门禁来保证
每个样本都使用三者。论文消融支持各组件的系统级效用，不能等价为逐样本“必须使用”。
因此本项目的 channel-wise causal gate 是 latent extension 的新增要求，不得写成 HOMER
原方法的组成部分。

### B. 本项目 7B Generator 通信线

`image + frozen Generator SFT prompt + communication -> caption`

SFT adapter 原本就是 image-conditioned，因此所有 full-plan/budget/token/state/bridge 条件必须保留原图。旧实现只给 standard description，会造成严重 receiver-interface shift，v3.5 已修正。

## 3. 主 baseline 与公平预算

固定三通道为 `conflict/local/global`，主带宽为每通道 8 个位置，共 24 个位置：

1. `full_plan_text`：完整三通道文本，作为语义上界，不做带宽匹配；
2. `budget_text`：每通道 causal tail 8 tokens 后解码为文本；
3. `token_embedding`：相同 24 token 的 receiver input embeddings；
4. `statebridge`：对三个通道分别做 StateBridge alignment，再拼接为 24 slots；
5. `learned_latent`：三通道合并后池化为 24 slots；
6. `typed_learned_latent`：每通道 8 slots，共享 pooler，仅 query 分型。
7. `typed_quantized`：把 Typed bridge 的 24 个连续输出逐槽量化到最近的 receiver vocabulary embedding。它与 Typed 使用相同输入、bridge、槽位和参数，只移除 off-manifold 连续残差，是判断“收益来自连续通信还是只来自学习压缩”的关键 control。

Learned 与 Typed 的 trainable parameter count 必须完全相同。官方 StateBridge 默认 64-token、同质 agent 的结果只作 appendix；三通道各 8 的版本明确称为 channel-preserving adaptation，因为它不是论文原配置。

`text_homer` 只回答完整系统效用，不能单独作为“latent 编码更优”的证据。

## 4. Planner trace 门禁

每个 trace 必须同时满足：

- Conflict 至少 2 对，左右脚本不同；
- Local/Global 每条 association 恰好为 `root -> step1 -> step2 -> step3`；
- generation hook 的 predictive state 与 emitted token 数严格一致，不允许裁剪或补齐，并用 teacher-forced predictive state 做 causal replay；
- 真正传给 bridge 的 communication state 使用 teacher-forced `post-token` 状态，即该位置已经读入对应 emitted token。不能把“预测 token 前的状态”误称为“token 自身语义状态”；
- `semantics` 保存 Planner 的真实原始输出，而不是占位说明；
- plan、sampling、seed、attempt、模型 revision、adapter 与 tensor SHA-256 全部写入 index；
- 每条 trace 还必须记录实际生成它的 Git commit，并固定 `trace_inputs.jsonl`、HOMER prompt 源文件和 frozen-adapter manifest SHA-256。受控重试可以来自多个 commit，但只有后三种实际输入/模型身份哈希完全一致时才能合并；不能把迁移代码的 commit 冒充为 tensor 的生成 commit。`trace_inputs` 只包含实际进入 Planner 的 cluster/split/image hash/description，使 caption 清洗不会伪造 trace 失效，同时任何真实 Planner 输入变化都会被门禁拒绝。未提交或 v3.5 工作树不干净时，正式 trace 生成直接失败。
- 若原始输出只违反 schema，允许统一的 `validator-feedback-format-only-v1` 恢复：在未改动的 HOMER 对话后附原始错误输出和 validator error，要求同一 Planner 只修复序列化或显式 opposition delimiter。自动校验修复前后语义字符串守恒；任何新增、删除或改写都会拒绝。最终 communication states 必须把修复文本放回原始 HOMER prompt 做 teacher-forced post-token replay，不能把 repair prompt 下的 states 混入正式 trace。原始/修复输出、error、seed、生成参数、repair prompt hash 和 alignment 均写入 index。

正式 train/validation trace 路径固定为 `data/cache/planner_traces_homer_strict_v35`。旧 v3.0 trace 禁止复制或引用。test trace 在模型/bridge 选择冻结后另行生成，避免测试集参与开发。

## 5. 数据切分与泄漏处理

| split | image clusters | caption rows | 用途 |
|---|---:|---:|---|
| train | 602 | 2162 | bridge fitting |
| validation | 64 | 216 | early stopping/选择 |
| internal_test | 97 | 327 | sealed primary test |
| official_hia_unseen_test | 24 | 72 | adapter-unseen official test |
| official_hia_seen_diagnostic | 23 | 69 | SFT 已见，仅诊断 |

所有 split 以 NYCC image cluster 切分，overlap 为 0。官方 HIA 47 张全部不参与 bridge 训练；其中只有 24 张对冻结 SFT adapters 真正未见。主 confirmatory 统计合并内部 97 张与官方未见 24 张，并分层单独报告来源。

## 6. Bridge 训练目标

对同一 `(image, caption)`：

```text
L = caption NLL
  + lambda_KL * KL(text-full-plan teacher || latent student)
  + lambda_sem * softplus(-logp(caption|matched plan)
                          +logp(caption|hard-negative plan)+margin)
```

Hard negative 不是随机图片：优先同数据源、standard-description TF-IDF 最接近、但 conflict signature 不同的 image cluster。这样降低只凭题材差异完成 matched/shuffled 判别的风险。

语义恢复 Phase A v2 实际优化跨样本表示判别：

```text
L_phaseA = reconstruction NLL
         + lambda_cf * matched/shuffled margin
         + lambda_NCE * symmetric InfoNCE
         + lambda_var * anti-collapse variance floor
```

InfoNCE 的 batch 来自 4 个梯度累积样本，只保留小型 bridge alignment graph。
teacher 使用初始化时固化的 receiver-native projection；仅 `detach()` 可训练 query
输出不足以构成静态 teacher，因为 optimizer step 后坐标系仍会漂移。
不保留四份 VLM forward activation。正式 trainer 在 Phase A 的 `info_nce<=0` 或
`gradient_accumulation<2` 时直接拒绝启动；validation 必须记录 retrieval@1。

Teacher 与 student 均使用原图和相同 caption；teacher 获得三个真实 Planner 输出。`lambda_KL=0` 时完全跳过 teacher forward，避免无意义算力。

### Phase A2 指标审计结论

`sequence_log_probability` 按有效 target token 取平均，matched 与 shuffled 使用相同
图片、相同 teacher-forced target，只替换 Planner memory。两遍低显存反传使用
`sigmoid(-matched+shuffled+margin)` 的固定一阶系数，其梯度与 softplus margin 的一阶
梯度一致。因此没有发现 gap 公式或反传符号错误。

但 v2 指标存在三项解释边界：

1. hard negative 来自另一 image cluster，而不是同图内只改变某一 channel 的严格
   counterfactual；因此它测的是跨图 plan identity sensitivity；
2. `0.02` 是预注册的工程阈值，尚未由 control distribution 或置信区间校准；
3. validation retrieval 曾将同 cluster 的重复 caption 当作 negatives，原始
   `0.190476` 无效，修复后必须每个 cluster 只保留一个 representation。

所以 v2 的严谨结论限定为：**当前 all-latent v2 未通过操作性语义门，caption stage
保持 No-Go；它不是对 latent communication 这一方法类别的否定。** 支持 No-Go 的有效
证据是 validation gap 仅 `0.002664`、没有样本超过 `0.2` margin、margin loss 接近
无区分基线 `softplus(0.2)`，以及 conflict router mass 降至 `0.0289`。原 retrieval
数值不参与该判断。

### Phase A3：通道平衡语义恢复（已完成；旧 outer baseline 与 A4 协议不同）

本轮使用 `64 train / 24 validation`、只训练 bridge、冻结两个 7B。不得直接进入
caption bridge。训练和选择规则为：

```text
L_rec = (L_conflict + L_local + L_global) / 3
L_NCE = (NCE_conflict + NCE_local + NCE_global) / 3
L_cf  = (L_swap_conflict + L_swap_local + L_swap_global) / 3
L_A3  = lambda_rec * L_rec + lambda_NCE * L_NCE
      + lambda_cf * L_cf + lambda_var * L_var
```

- 每个 `L_channel` 先按本 channel 的有效 token 数归一化，再对三个 channel 等权平均，
  防止较长 local/global association 在 token-weighted CE 中控制优化；
- Phase A3 首个 control 固定三路 cross-channel mixing 为等权，禁止可学习 router 通过
  把 conflict 权重压到零来绕过任务。只有固定门通过后，才比较带 minimum-usage/load-
  balancing 正则的 learned gate；
- 分别只替换 conflict、local、global，保存每张图的
  `delta_conflict/delta_local/delta_global`，不再只交换整个 memory；
- InfoNCE 必须真正进入 loss，并按 channel 计算。当前 A4 虽使用冻结 Generator 的
  contextual hidden，但仍经过初始化时复制的 query projection，不能称为完整的
  receiver-native semantic projector；它只能作 auxiliary identity/alignment diagnostic。
  下一版正式 outer 前必须在 train-only contextual states 上拟合并冻结 projector（或
  改为 target-span/output-distribution alignment）。v2 的固定随机投影只可能证明
  trace identity/词汇区分，不能证明 Receiver 学到可用语义；
- 所有 contrastive 统计以 image cluster 为单位，一 cluster 一个样本；保存逐图数值，
  使用 image-clustered bootstrap 95% CI；
- 同时保留跨图 TF-IDF hard negative 作为次要 stress test，但主 gate 使用单通道
  counterfactual；阈值由 identity/shuffled/text-teacher controls 校准后冻结；
- 三个 channel 均须达到高于 control 的正向 gap；只看总体平均不算 `strong_go`。但
  `64/24` 的任一通道点阈值未达，不能直接写作方法级 No-Go：由于 pilot 的 Type-II
  风险，gate 输出 `pilot_inconclusive`，转入预注册的 sealed outer semantic validation。
  只有真实技术失败，或后续已预注册的明显负向效应，才允许硬停止；若 outer validation
  仍显示 conflict 不可恢复，再进入 `C-text + A-latent` 混合消融。

本轮完成后，只有通过剩余 40 个未参与 early stopping 的 outer semantic confirmation，
才允许训练 caption bridge，并按 `Text-HOMER / C-text+A-latent / C-latent+A-text /
All-latent` 顺序做低成本比较。

### Phase A3 正式结果（job 6707953）

作业在 b-batch 单张完整 H100 上正常退出，exit code 为 0，耗时 25 分 06 秒；正式
preflight、CUDA allocator 检查和训练后 validator 均通过。运行 provenance 固定在
`outputs/pilot/cross_attention_semantic_phase_a3/run_manifest.json`，代码 commit 为
`28d530769d3af8051696b21d55bf257cdcaa35dc`，配置、数据 manifest、666 条 trace index
的 SHA-256 均被记录。

| 项目 | 结果 |
|---|---:|
| 训练/验证 cluster | 64 / 24 |
| epoch / optimizer steps | 5 / 80 |
| bridge trainable parameters | 2,820,612 |
| policy trainable parameters | 0 |
| validation total（epoch 1 → 5） | 4.12281 → 3.00137 |
| validation caption NLL（epoch 1 → 5） | 1.80835 → 1.12946 |
| validation InfoNCE retrieval@1（epoch 1 → 5） | 0.125 → 0.535 |
| 最终整体 matched−shuffled gap | 0.000216 |
| 最终 conflict/local/global gap | −0.000040 / −0.000317 / 0.001006 |
| 最终 conflict/local/global retrieval@1 | 0.396 / 0.563 / 0.646 |
| 最大 relative update norm | 0.0430 |
| fixed channel mass | 1/3, 1/3, 1/3 |

`complete.json` 为 `status=complete`、`epochs_completed=5`。`semantic_gate.json` 为
`status=pilot_inconclusive`，但 `engineering_gate_pass=true`：NLL 和表示层 retrieval
均改善，更新有限且无 NaN/OOM；然而 24-cluster 的 channel-wise matched/shuffled
bootstrap CI 仍跨 0（conflict `[-0.002806, 0.002740]`、local `[-0.001935,
0.001354]`、global `[-0.000262, 0.002396]`）。这不是技术失败，也不是 latent 方法的
最终 No-Go；它只说明低功效 pilot 尚未证明稳定的逐通道 Receiver 使用。A3 的旧 outer
baseline 可以在 clean commit 后作为历史对照重提，但不能代替新的 A4 protocol；当前
仍继续禁止 caption quality 结论、DPO 和 preference learning。

### Phase A4：channel-isolated functional semantic pilot（已完成）

A4 使用 `configs/pilot/cross_attention_semantic_phase_a4.yaml`，仍只训练 bridge、冻结
7B receiver，且关闭 semantic-recovery 图像输入。每次只启用目标 channel，并加入 donor-side
reconstruction；这些是本项目的因果诊断扩展，不是 HOMER 原方法。作业 `6711109` 完成
5 epochs/80 optimizer steps，`complete.json` 为 `status=complete`，工程 gate 通过，
但 semantic gate 为 `pilot_inconclusive`。

| 指标 | A4 validation |
|---|---:|
| overall matched−counterfactual log-probability gap | 0.0064997 |
| conflict / local / global gap | 0.0088532 / 0.0051597 / 0.0054863 |
| conflict / local / global bootstrap 95% CI | [-0.0078013, 0.0270501] / [0.0014195, 0.0093984] / [0.0003536, 0.0114575] |
| InfoNCE retrieval@1（overall） | 0.5625 |
| mean relative update norm | 0.02503 |

因此 A4 只能证明当前实现完成了可运行的 target-only/donor-reconstruction 机制 pilot；
它没有证明稳定的 channel-causal use，更没有产生 caption 质量、latent-vs-text 或
偏好学习结果。A4 checkpoint 不得直接进入 caption；下一步必须先重写独立的 A4 outer
evaluator，并补上按 channel token length 匹配 donor、zero-bridge control 和预注册的
cluster-level 统计规则。旧的 A3 outer 脚本不能冒充 A4 evaluator。

注意：`outputs/pilot_validation/` 中已有的 `text_homer`、`learned_kl`、`typed_kl` 等
caption 文件是在旧的 visual-caption pilot（commit `cdcfbbd...`）下生成的，不是 A4
修订方法的结果。它们没有通过 A4 semantic gate，也缺少当前要求的完整 checkpoint/
evaluator provenance；其中 `token_embedding`、`statebridge` 和 `typed_quantized` 还出现
了空输出或乱码。故本轮不计算其 `good-caption rate`，不把这些文件用于比较，也不把
已有 880 个 Group-of-3 packet 写成新方法的盲评证据。必须在 A4 outer 通过后，使用新
checkpoint、新 output 目录和完整 manifest 重新生成 caption，再交给盲评。

### Phase A3 的验证协议与样本量定位

验证 latent 是否被 Receiver 真正使用，采用三层证据，而不是单一 reconstruction loss：

1. **表示层**：按 conflict/local/global 分别做 contextual-teacher retrieval/InfoNCE；
2. **行为层**：保持图片、prompt、target 和另两路 memory 不变，每次只替换一路，测
   `delta_conflict/delta_local/delta_global`；这对应 Interlat 的 task-mismatched latent
   和 information-scrambling control 思路；
3. **下游层**：机制门通过后，在未参与训练/early stopping 的图片上比较 Text-HOMER、
   matched latent 与 counterfactual latent 的 Group-of-k caption 质量。

统计单位始终是 image cluster。连续 gap 用 paired/image-clustered bootstrap 95% CI，
比例用 binomial interval；报告 effect size，不以单个 p-value 代替效果量。24 个 validation
cluster 在最坏比例 `p=0.5` 附近的近似 95% 半宽约为 `1.96*sqrt(.25/24)=0.20`，因此：

- `64 train / 24 validation` **只用于工程与机制方向 pilot**：发现 loss 接线错误、
  channel collapse、明显无依赖或明显强依赖；
- 它不能支持“latent 优于 text”或“某方法失败”的论文结论，也不应用于微小架构排序；
- point gate 通过但任一 channel 的 bootstrap 下界未超过 0 时，状态只能是
  `go_to_outer_semantic_validation`，不得进入 caption bridge；
- point gate 未通过但运行和指标均有效时，状态为 `pilot_inconclusive`，同样不得进入
  caption bridge，也不得写作 latent 方法失败；它必须转入更大的 sealed outer semantic
  validation。仅技术异常可以在此阶段硬停；
- `bounded_residual_update` 或 `fixed_equal_channel_mass` 等工程不变量失败时，状态为
  `hard_no_go`，必须先修复实现；这不是 latent 语义效果结论。
- 下一步先在剩余 40 个未用于 early stopping 的 validation cluster 上做 sealed semantic
  confirmation。仍不确定时，用 pilot 的逐图标准差执行预先功效分析，再扩大到 64/97/121
  clusters；不得事后反复查看同一批数据并改阈值；
- v1/v2 将用修复后的 cluster-level、逐通道 evaluator 重新评估。旧 checkpoint 没有
  fixed-equal/channel-balanced 训练，因此重评只能比较其既有表示与因果敏感性，不能伪装
  成 Phase A3 训练结果。

### 旧 semantic bridge 修正重评的冻结协议

详细审查记录见 [`docs/SEMANTIC_REEVALUATION_PROTOCOL_ZH.md`](SEMANTIC_REEVALUATION_PROTOCOL_ZH.md)。
审查发现旧 donor 选择没有控制 channel 长度；对 v1 的单一 concatenated-memory
softmax，这会改变归一化分母，不能把 gap 直接解释为语义依赖。修正后的
`scripts/re_evaluate_failed_semantic_bridges.py` 使用：

- 与 checkpoint 严格匹配的 v1/v2 bridge 类和 state dict；
- 固定 hash 抽取的 24 个 validation image clusters，每 cluster 一个代表 row；
- 来自 train split、且排除该 checkpoint 拟合过的 clusters 的 donor pool；
- 不同 conflict signature，先按被替换 channel 的 token-length 差最小化，再在同长度
  候选内优先同源并最大化 description TF-IDF 相似度；
- image、semantic target、prompt 和另外两个 channel 全部固定的单 channel swap；
- image-cluster bootstrap 95% CI、Wilson proportion interval、残余 length-gap
  correlation 与完整 provenance。

该重评只测 semantic receiver sensitivity，不产生 caption，也不使用 sealed test。
`pilot_inconclusive` 仍表示 24-cluster 统计功效不足，不能写成方法级 No-Go；只有
预注册 outer semantic validation 的稳定负向结果才可否定旧方法。

重评作业 `6695787` 已完成：v1/v2 均为 `pilot_inconclusive`。v1 只有 global channel
出现极小正向 gap（均值 `0.001170`，95% CI `[0.000056, 0.002230]`），conflict/local
不稳定；v2 的 conflict/global CI 均跨 0，local 均值为负。两者三个 channel 的
`fraction(gap > 0.02)` 都是 `0/24`，所以不能进入 caption bridge。完整数值见
`docs/SEMANTIC_REEVALUATION_RESULTS_ZH.md`。

A3 smoke 作业 `6689653` 独立因 global target `492 > max_target_tokens=384` 的配置错误
退出，已按 engineering failure 记录。配置改为 `768` 后，replacement smoke `6706516`
通过 validator；formal A3 `6707953` 随后完成。formal 的 engineering gate 通过，但
24-cluster semantic gate 为 `pilot_inconclusive`，因此必须先做 outer confirmation，
不能把该结果当作 caption 质量证据。

这一协议依据 NLP 功效分析与配对显著性测试规范；Interlat 的错配/结构破坏实验用于证明
latent 的任务特异性，而不是仅凭 latent 可解码就宣称 Receiver 使用了它。

## 7. Successive filtering，而不是一次性矩阵

### Gate E：v3.5 engineering smoke

- 真实 image + 真实 Planner trace；
- hidden/token/semantics/hash 全部通过；
- image-conditioned SFT receiver；
- Learned/Typed 参数和 24-slot 预算一致；
- policy trainable params=0；
- loss/gradient/update finite；
- 记录峰值显存。

Gate E 通过前禁止正式训练。replacement smoke `6706516` 已通过 Gate E；formal A3
`6707953` 已完成并正常退出。两类结果都不能被改写成 caption 质量证据。

### Pilot P（A3 semantic gate 通过后才解锁）

这三个 caption-level latent pilot 仍不是当前阶段；它们必须等待 Phase A3 和 outer semantic
confirmation 通过后才可提交。每个仅 64 train clusters、24 validation clusters、1 seed：

1. Learned + KL；
2. Typed + KL；
3. Typed + no-KL。

64/24 clusters 通过 `SHA256(seed, split, cluster_id)` 固定抽样，不按编号截断。24 张只用于 early stopping；真实 pilot 生成在剩余 40 张 outer-validation 图片上进行，避免用模型选择图片重复证明模型收益。三个优化作业串行完成后，使用 3 个共同 seeds 生成 `Text-HOMER / StateBridge / full-plan text / budget text / token embedding / Learned+KL / Typed+KL / Typed-quantized / Typed-no-KL`，并构造匿名、双向 Group-of-3 packet。流程随后停止等待独立评审；不得仅凭 validation loss 自动扩展。Group-of-3 只承担低成本筛选；最终主结论必须使用 Group-of-10。优胜条件同时要求 validation NLL、matched-vs-hard-negative margin和 outer-validation 真实生成不退化。

### Confirmatory C：只扩展 pilot 优胜者

- 602 train / 64 validation；
- 至少 3 seeds；
- SFT receiver 为主；
- Base receiver 只为优胜架构补充训练，回答 receiver-specificity，不再做四乘四矩阵；
- early stopping 只看 validation，test 只在配置冻结后运行一次。

## 8. 生成与盲评

主质量评测：121 张 adapter-unseen 图片、10 个共同 generation seeds、每条件 Group-of-10。候选数量参考 Humor in AI 官方仓库中公开的 10-caption 生成/排序资产，但 Group-of-10、镜像 A/B、绝对标签并非该论文的逐行复现；Group-of-3 仅作为历史敏感性分析，不承担主结论。每个比较生成 A/B 镜像方向，组内候选顺序独立随机化。评审必须支持看图，记录 provider/model/version/temperature/prompt hash。用 `build_judge_calibration.py` 从非 test 的官方 crowd ranking 构造五个清晰偏好示例；它复现官方 5-shot 校准思想，但不是论文每个测试项随机配五对的逐样本实现。本项目的 `Tie` 与绝对标签也属于额外扩展，因此应表述为“paper-aligned adaptation”，不能声称逐行复现官方 judge。

A/B 镜像只用于诊断位置偏差，不是两个独立观测。统计前必须先在每个 `rater × image × comparison` 内折叠镜像方向；否则会人为扩大样本量并污染 rater agreement。

同时汇总：

- `overall` group win rate；
- `best_pick` win rate；
- 每条 caption 的 `good/weak/bad`；
- generation-seed variance；
- image→rater 两层 bootstrap 95% CI；
- Krippendorff nominal alpha；
- 主比较族 Holm correction；
- internal 与 official-unseen 分来源结果；
- 23 张 SFT-seen official 图只作 diagnostic，不混入主 CI。

预注册主比较：

1. full-plan text vs Typed；
2. budget text vs Typed；
3. token embedding vs Typed；
4. Learned vs Typed。

机制性次比较：Text-HOMER vs full-plan、budget text vs token embedding、token embedding vs StateBridge。

## 9. 多样性实验

主质量 Group-of-10 与多样性使用同一批固定 plan、10-seed generations，但分别回答质量和多样性问题。对 121 张主测试图，每条件共 1,210 条 caption。报告：

- Distinct-1/2；
- self-BLEU-2；
- pairwise TF-IDF semantic distance；
- Vendi score；
- Humor in AI 官方代码的 Average EAD（n=1..5，V=32,000）；
- `all-mpnet-base-v2` SBERT diversity；
- 人工/独立 judge 的 angle label coverage；
- 只在 `good` captions 上重算 diversity。

只有“质量不降且 good-only angle coverage 上升”才能支持 latent 增强幽默角度多样性的主张。纯 lexical diversity 不能替代这个结论。后续可另做 multi-plan sampling（5 plans × 2 captions），但不得与 fixed-plan 10-caption 结果混在一起。

## 10. Go/No-Go

进入 preference learning 前必须同时满足：

1. 至少一个 learned bridge 对预算匹配 control 有同方向的 3-seed 增益；
2. image-cluster CI/校正后统计不支持退化；
3. absolute good rate 与 grounding/hallucination 不退化；
4. hard-negative semantic margin 明显优于随机/错配；
5. 多样性增益在 good-only subset 和人工 angle coverage 上仍存在。

否则停止在 bridge 结论，不通过 DPO“补救”一个尚未证明有效的通信机制。

## 11. 当前可复现状态

- Python 3.12 独立环境与 locked dependencies：通过；
- v2.5/v3.0 executable isolation：通过；
- frozen adapters SHA-256：通过；
- image-clustered split/hash/leakage：通过；
- CPU tests：完整 suite 75/75 通过；
- 正式 v3.5 GPU trace/bridge smoke：作业 6649172 已通过；
- 数据质量修复：Electronic Sheep 的标量 `UNKNOWN` 曾被错误迭代成 `U/N/K` caption；已删除 133 个无效训练行。各 split 数量保持 602/64/97/24/23，但具体 cluster 成员和部分 standard-description 来源发生变化，不能据“数量相同”复用全部 trace；
- 正式 trace 生成：666/666，缺失、重复与 failure 均为 0；
- Cross-attention Phase A v1 已完成但为方法级 No-Go：epoch 5 validation NLL=0.632975，matched-minus-shuffled gap=0.004843，低于 0.02 gate，caption stage 未启动；
- 审计确认 v1 的 InfoNCE 未进入训练调用路径，且三通道拼接后的统一 softmax 存在长度竞争。v2 已改为通道内独立 softmax、通道间门控，并强制真实 gradient-window InfoNCE；
- v2 第一轮真实 GPU engineering smoke：作业 6688553 已通过。随后代码审计修复了 teacher projection 跨 step 漂移风险；
- v2 post-fix GPU smoke：作业 6688566 已通过；冻结 policy trainable params=0，bridge params=2,820,804，InfoNCE=0.7612，smoke retrieval@1=0.5，gradient/update finite，峰值显存约 11.82 GB。该数值只证明训练路径执行，不能作为泛化结果；
- Hierarchical Phase A v2：作业 6688689 已完成并判定为**当前配置的操作性 No-Go**。validation NLL 从 1.1196 降至 0.6326，但 matched-minus-shuffled gap 仅 0.002664（工程 gate 0.02），`gap>0.2` 的比例为 0；conflict channel 权重从约 0.315 降至 0.0289。它说明当前 loss/router 没有形成足够的 plan 条件依赖，不得外推为“latent 方法失败”；
- v2 报告的 validation retrieval@1=0.190476 不可作为正式结论：实现错误地把同一 cluster 的 3/6 条 caption 行当作互为 negatives。未来已修正为每个 image cluster 只取一条 representation。该数值既不能支持也不能反对 v2；
- caption bridge 继续禁止。不得通过增加 epoch 或扩为 602 条来绕过语义门。A3 已完成，A4
  也已完成但为 `pilot_inconclusive`；下一项只允许修正 A4 outer evaluator、channel-length
  donor 与 zero-bridge control，不能直接进入 `C-text + A-latent`；
- Phase A3 已实现并通过 CPU suite；配置为 `configs/pilot/cross_attention_semantic_phase_a3.yaml`。真实双样本 GPU smoke 首次作业 `6689653` 因 `electronic_sheep:325:0` 的 `492 > 384` token 上限配置错误退出；配置提高到 `768` 后，replacement smoke `6706516` 已在完整 H100 上通过 validator，未截断完整 HOMER chain；
- A3 replacement smoke 已通过冻结参数、真实两 cluster、逐通道 counterfactual、contextual InfoNCE、有限梯度与实际 update 门禁；随后 formal A3 job `6707953` 已在 b-batch 单张完整 H100 上以 exit code 0 完成 5 epoch/80 steps。其工程 gate 通过但 semantic gate 为 `pilot_inconclusive`，不得把 validation NLL/retrieval 的改善写成 caption 质量收益；
- Formal A3 与 A4 的逐轮 checkpoint、validation JSONL、`complete.json`、`semantic_gate.json`、
  preflight 和 job stats 均已保留。A3 的旧 outer 结果尚不存在，A4 的新 outer evaluator
  尚未提交；在该 A4 gate 之前不得启动 caption bridge、DPO 或任何 preference job；
- pilot 真实生成评估：训练后自动生成 packet，但必须由独立评审完成才允许放大；
- preference learning/DPO：属于旧方案，在 v3.5 latent gate 通过前禁用。

## 12. Text/latent 混合消融（语义 Gate 后）

不默认三个 channel 都适合 latent。为控制实验数量，第一轮只比较：

1. `Text-HOMER`：conflict/local/global 全文本；
2. `C-text + A-latent`：conflict 保留文本，local/global 使用 latent；
3. `C-latent + A-text`：conflict 使用 latent，local/global 保留文本；
4. `All-latent`：三通道均使用 latent。

四个条件必须共享图片、plan、caption prompt、generation seeds 和信息来源。只有某个
association 组合显示收益后，才继续区分 local 与 global；避免直接展开全部 2^3 组合。
主判断同时看 absolute good rate、grounding、matched/shuffled sensitivity 和参数/延迟。

## 13. 失败记录纪律

所有失败必须同步写入 `docs/EXPERIMENT_FAILURES.jsonl`，阅读版规则在
`docs/EXPERIMENT_FAILURE_LOG_ZH.md`。必须区分 environment/data/engineering/method/
evaluation 五类；禁止把排队、NVML、OOM、依赖或代码异常写成方法失败。每次修复必须
使用新输出目录，保留旧日志、checkpoint、配置和 job ID，并及时更新本计划的“当前可复现状态”。

### 13.1 当前外层语义确认资源请求（2026-09-03）

PJM 明确禁止把 `node=1` 与 `gpu=1` 同时指定（`GENKAI1006`），也禁止把
`gpu=1` 与 `-P exec-policy=simplex` 同时指定（`GENKAI0029`）。在 GENKAI 上，
`node=1` 的 node-allocated 作业就是 simplex/node-exclusive；GPU-capable 的
`b-batch` 节点配置会自动给该作业分配 GPU。因此外层 semantic confirmation 的
smoke/formal 脚本统一使用 `b-batch + node=1`，不再写显式 `gpu` 或
`exec-policy`，并在进程内固定 `CUDA_VISIBLE_DEVICES=0`，使 PyTorch 仍只看到一张
稳定设备。旧的共享 GPU 探针 `6708044` 已取消；修正后的节点独占 smoke 为
`6708113`，调度统计已确认 `NODE NUM=1`、`gpu=4`、`simplex=true`，并以 exit code 0
在 2 分 01 秒完成。其真实 trace、forward、counterfactual、validator 均通过；2-cluster
语义结果仍只标记 `outer_semantic_inconclusive`，不作方法结论。这次变更只修复调度
资源类型冲突，不改变模型、数据或语义实验设计；smoke 通过后才允许提交 40-cluster
outer confirmation。

随后提交的旧 A3 40-cluster outer confirmation `6708118` 在实际 forward 前因
tracked worktree dirty 被 preflight 拒绝，因此没有 outer 结果；它不能被当作 A4
confirmation。A4 `6711109` 已完成但 gate 仍为 `pilot_inconclusive`。下一次 outer
confirmation 必须使用独立的 A4 evaluator、新输出目录和 clean provenance。不得为
缩短等待改回 MIG（已有 allocator 故障记录）或未经确认改用共享 GPU；正式作业仍须
由自身的 `check_cuda_resource.py` 确认设备后才解释数值。

### A4 outer evaluator smoke（2026-09-04）

新的 A4 专用 outer evaluator 已在 clean commit `6fb5c32` 后通过全量 preflight（全量
测试、编译、数据 `2846/2846`、trace `666/666`、frozen-artifact 校验均通过）。曾
提交 2-cluster × 1-seed 的工程 smoke `6711814`（`b-batch + node=1`），但因检查到
`b-inter` 空闲而在启动前取消；`b-inter` 的 batch 覆盖尝试被 PJM 拒绝，未执行任何
forward。当前重新提交同一唯一 smoke 时仍使用已验证的 `b-batch + node=1`；该
请求不显式申请 `gpu`、不使用 MIG，也没有与其他资源组并行 race；walltime 收紧为
10 分钟，依据已通过的 A4 real-trace smoke 实测约 4 分钟。它只检查
target-only channel isolation、image-free semantic prompt、length-matched donor、
donor reconstruction、zero-bridge control 和 artifact validator，不能产生 humorous
caption 或 `good-caption rate`。在 smoke/outer gate 结束前，历史
`outputs/pilot_validation/` 的 120-row caption 文件仍保持 quarantine，不能与新协议
混合统计。若 smoke 通过，再提交 40-cluster × 3-seed sealed outer evaluator；只有
其 `outer_semantic_go` 才解锁新的 caption generation 和盲评。

## 14. 权威参考

1. Shang et al. HOMER. ICLR 2026. https://openreview.net/pdf?id=SzaRhPom4o
2. HOMER official implementation. https://github.com/Shang-hub/HOMER-Official-Implementation
3. Du et al. InterLat. ACL 2026. https://aclanthology.org/2026.acl-long.1248/
4. Peng et al. StateBridge. COLM 2026. https://arxiv.org/abs/2608.13317
5. Zhang et al. Humor in AI. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf
6. Hessel et al. Electronic Sheep. ACL 2023. https://aclanthology.org/2023.acl-long.41/
7. Tevet & Berant. Evaluating the Evaluation of Diversity in NLG. EACL 2021. https://aclanthology.org/2021.eacl-main.25/
8. Friedman & Dieng. The Vendi Score. TMLR 2023. https://arxiv.org/abs/2210.02410
9. Yang et al. Hierarchical Attention Networks. NAACL 2016. https://aclanthology.org/N16-1174/
10. Libovicky & Helcl. Attention Strategies for Multi-Source Sequence-to-Sequence Learning. ACL 2017. https://aclanthology.org/P17-2031/
11. He et al. Momentum Contrast. CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html
12. van den Oord et al. Contrastive Predictive Coding. 2018. https://arxiv.org/abs/1807.03748
13. Card et al. With Little Power Comes Great Responsibility. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.745/
14. Dror et al. The Hitchhiker's Guide to Testing Statistical Significance in NLP. ACL 2018. https://aclanthology.org/P18-1128/
15. Graham et al. Statistical Power and Translationese in Machine Translation Evaluation. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.6/
16. Howcroft and Rieser. What happens if you treat ordinal ratings as interval data? EMNLP 2021. https://aclanthology.org/2021.emnlp-main.703/
17. Koehn. Statistical Significance Tests for Machine Translation Evaluation. EMNLP 2004. https://aclanthology.org/W04-3250/
