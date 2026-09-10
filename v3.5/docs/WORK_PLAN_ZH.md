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

## 0B. A4 未达标后的 A5 修复候选（2026-09-05）

A4 outer 的三 seed、40 image-cluster 结果为：conflict `0.0115005`，local
`0.004940757`，global `0.003208954`。local/global 的点估计没有达到预注册的
`0.01`，但 bootstrap lower bound 为正；因此结论仍是“因果通道使用未证实”，不是
“latent 类方法无效”。A4 的 immutable artifact 不覆盖，证据仍保留在
`outputs/outer_semantic_confirmation/a4_outer40/summary.json`。

本次只实现一个有文献依据、可归因的低风险修复候选，不声称它必然提高 gap：

1. `cross_attention.py` 增加 `projection_mode=per_channel`，为 conflict/local/global
   分别学习 K/V/O 投影；shared projection 仍保留为 A4 基线。多源 attention 的 flat 与
   hierarchical 组合有 ACL 2017 的架构依据，但本项目的三类幽默字段和该实现是本项目
   的适配，不是论文原样复现。
2. A5 保持 `target_only` channel mask、无图像 semantic-recovery prompt、donor
   reconstruction、matched/shuffled margin、三路 InfoNCE 和 fixed-equal fusion，避免
   用其它通道或图像 shortcut 解释目标。
3. 增加 `semantic_target_alignment=0.5`：冻结 Receiver 在原生文字字段条件下产生
   target-span final hidden 的 teacher，bridge student 对其做 cosine distillation；这
   是 BLIP-2 的“冻结模型 + 轻量 bridge、表示对齐后再做生成可解释性”原则，以及
   HistAlign 对 memory/current hidden misalignment 的针对性借鉴。该项只提供 receiver-
   native 表征对齐，不能单独证明 causal channel use，最终仍由 matched/counterfactual
   和 caption 结果决定。
4. A5 训练使用全部 602 train clusters、64 validation clusters（不再用 24-cluster
   development subset），仍只训练 bridge，两个 7B policy 完全冻结。因为 per-channel
   K/V/O 使参数量增加约三倍，A5 是“修复候选”而非公平 placement ablation；若有效，后
   续必须单独做 projection-only 与 alignment-only 消融。

协议更正（2026-09-05）：A5 初次作业 `6712005` 的训练入口误用了未按 channel 长度匹配
的 `hard_negative_cluster_map`，第 1 epoch 的 global gap 出现明显 outlier；该作业已
取消，输出标记为工程协议错误，绝不用于科学结论。修复后的训练入口使用
`length_matched_channel_donors(..., allow_target_donor_overlap=True)`，并把完整 donor
map 与长度诊断写入 `channel_donors.json`。新作业输出固定为
`outputs/pilot/cross_attention_semantic_phase_a5_lengthmatched/`，必须重新 smoke 后才
能训练；详见 `docs/EXPERIMENT_FAILURE_LOG_ZH.md` 的 V35-ENG-006。

执行前审计又发现 A5 outer 的 sealed target trace index 不包含 validation donor trace。
该问题已记录为 V35-ENG-007 并修复：outer 命令必须同时传入
`--trace-index ..._test/index.jsonl` 与 `--donor-trace-index .../index.jsonl`，脚本合并两者
并保存 donor index hash；禁止使用只含 held-out target 的旧命令。
随后发现旧重评 helper 还硬编码 `split=train`，与 A5 outer 的 validation donor 预案冲突；
已改用 formal bridge 的通用 length-matched donor 实现，并记录为 V35-ENG-008。outer
提交前必须在合并 index 上完成 validation-donor mapping smoke。

训练/评测顺序固定为：

```text
clean commit + CPU tests
→ A5 real-trace bridge smoke (≤2 samples, no scientific training)
→ A5 bridge-only training (all 602/64, frozen 7B)
→ sealed held-out Planner traces (internal_test 97 + official unseen 24)
→ A5 outer semantic confirmation (121 clusters × 3 seeds)
→ only if outer semantic_go: Text-HOMER vs A5 latent caption generation
→ 10 candidates/condition/image, mirrored Group-of-10 packet
→ independent multi-rater blind aggregation + image-cluster bootstrap
```

测试 Planner traces 单独存于
`data/cache/planner_traces_homer_strict_v35_test`，输入 manifest 为
`data/processed/latent_bridge_v35/test_trace_inputs.jsonl`；不得把 test trace 追加到
训练/validation 的 666 条 index。`cache_planner_traces.py` 会校验 manifest 中每个
cluster 的 image hash/description，并把 input-manifest hash、prompt hash、adapter hash
写入每条 trace。缺 trace、repair failure、hash 不一致或 outer semantic gate 不通过时，
caption 作业 fail-closed，不生成“提升”结论。

资源策略：共享 `c-batch + gpu=1` 的 A5 作业曾被排到 13:00；为优先最快且不增加
GPU 数量，已取消该 queued copy，改用内容完全相同的
`jobs/cross_attention_phase_a5_csimplex.pjm`（`c-batch + node=1`，一张原生整卡，
不叠加 `gpu=1`/`exec-policy=simplex`）。它必须等待 retry2 释放当前 node 后再启动；该
切换只改变调度契约，不改变模型、数据、seed、loss 或输出目录。
若首次 sealed cache 的 `failures.json` 非空，只能提交
`jobs/cache_test_planner_traces_retry.pjm` 对失败 cluster 做有界重试；不得删除或覆盖已
成功 trace，也不得把失败 cluster 当作完整 121 条使用。第一轮重试达到 `117/121` 后
仍有 4 条严格 schema 失败（`nycc_236/323/394/682`），因此新增
`jobs/cache_test_planner_traces_retry2.pjm`：仍使用同一 image manifest、Qwen revision、
HOMER prompt、validator-feedback repair 和 provenance，只把 residual cluster 的随机
尝试上限从 8 提到 16；成功记录由脚本跳过，失败记录仍会使 validator fail-closed。直到
`test_planner_trace_validation_retry2.json` 报告 `119/121`（`nycc_323/394` 仍失败），故
又注册 `jobs/cache_test_planner_traces_retry3.pjm`：只对这两个 residual cluster 进行最多
32 次同协议采样。若 retry3 仍不能得到严格、可 replay 的 JSON，正式结论必须保留为
`data gate blocked`，不得人工补写或放宽 schema；只有 `test_planner_trace_validation_retry3.json`
报告 `121/121` 且 `status=pass`，A5 outer 和 caption 作业才可使用 sealed test。

当前适用文献依据：BLIP-2 (Li et al., 2023)、HistAlign (Wan et al., EMNLP 2023)、
Flamingo (Alayrac et al., NeurIPS 2022)、Multi-Source Attention (Libovický & Helcl,
ACL 2017)、CPC/InfoNCE (van den Oord et al., 2018)。这些文献支持冻结 receiver、轻量
bridge、receiver-native alignment、gated/cross attention 和 contrastive negatives；没有
任何一篇文献保证本项目的 `0.01` gap 阈值，因此 A5 结果必须实测、不得预写。

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
`b-batch` 节点配置会自动给该作业分配 GPU。因此短 smoke 使用
`b-batch + node=1`，不再写显式 `gpu` 或 `exec-policy`，并在进程内固定
`CUDA_VISIBLE_DEVICES=0`，使 PyTorch 仍只看到一张稳定设备；正式 outer 在检查到
完整 GPU 空闲且队列更短时使用已验证的 `c-batch + gpu=1`，不写 `node=1` 或
`exec-policy`。旧的共享 GPU 探针 `6708044` 已取消；修正后的节点独占 smoke 为
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
forward。重新提交的唯一 smoke `6711856` 使用已验证的 `b-batch + node=1`，于 208
秒以 exit code 0 完成，A4 validator 通过；2-cluster 结果为
`outer_semantic_inconclusive`，不作方法结论。该
请求不显式申请 `gpu`、不使用 MIG，也没有与其他资源组并行 race；walltime 收紧为
10 分钟，依据已通过的 A4 real-trace smoke 实测约 4 分钟。它只检查
target-only channel isolation、image-free semantic prompt、length-matched donor、
donor reconstruction、zero-bridge control 和 artifact validator，不能产生 humorous
caption 或 `good-caption rate`。在 smoke/outer gate 结束前，历史
`outputs/pilot_validation/` 的 120-row caption 文件仍保持 quarantine，不能与新协议
混合统计。若 smoke 通过，再提交 40-cluster × 3-seed sealed outer evaluator；只有
其 `outer_semantic_go` 才解锁新的 caption generation 和盲评。

2-cluster smoke `6711856` 已通过工程 validator；其 semantic status 为
`outer_semantic_inconclusive`，符合小样本 smoke 的预期，不能外推为方法结论。随后
原提交的 40-cluster × 3-seed sealed outer job `6711870` 因 b-batch 预测排到
2026-09-10 而在启动前取消；没有 forward 或科学输出。当前正式请求已切换到
已验证且当时有空闲完整 GPU 的 `c-batch + gpu=1`（2 小时 walltime），只保留一个
新 job。该 job 的输出目录为
`outputs/outer_semantic_confirmation/a4_outer40`；完成后必须先通过
`validate_outer_semantic_confirmation_a4.py`，再按预注册 gate 决定是否解锁 caption。
运行结果和完整 hash/统计见 [`docs/A4_OUTER_VALIDATION_RESULTS_ZH.md`](A4_OUTER_VALIDATION_RESULTS_ZH.md)：
validator 通过，但 gate 为 `outer_semantic_inconclusive`；因此当前仍不能报告
`good-caption rate`。

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

## 15. 当前执行状态（2026-09-05）

- A4 的 outer 结果仍只说明 local/grounding gap=`0.0049407572`、global/association
  gap=`0.0032089539` 未达到预注册 `0.01`，不能作为 latent 方法已失败的充分证据。
- A5 修正版采用逐 channel 长度优先 donor、per-channel K/V/O、target-only mask、
  receiver-native contextual alignment、实际调用 InfoNCE 和 fixed-equal fusion；两套
  7B policy 冻结，只更新 bridge。精确方法与引用见 `PHASE_A5_REPAIR_METHOD_ZH.md`。
- sealed held-out Planner trace 的 validator-repair 已完成第三轮有界重试：
  `121/121` records、`missing=0`、`failure_records=0`，验证文件为
  `outputs/preflight/test_planner_trace_validation_retry3.json`。所有输出均保持
  原始 hash/provenance，未人工改写语义。
- 当前 A5 canonical 输出目录曾被取消作业 `6712281` 留下空目录，重跑 `6712296` 因
  provenance guard 正确拒绝；空目录已可恢复地隔离为
  `outputs/pilot/cross_attention_semantic_phase_a5_lengthmatched_canceled_6712281_empty`，
  事件记录为 `V35-ENG-011`。下一次提交不得复用部分/空输出，也不得删除历史证据。
- `a5_pre_submit_20260905_0207.json` 在 clean commit `d275122` 上通过完整 preflight。
  canonical 目录清空后，才允许按已验证的 c-batch node-exclusive 资源契约重新提交
  A5；A5 semantic gate 通过后才可运行 sealed outer，outer gate 通过后才可生成 caption。
- 禁止提前把 bridge 的 semantic gap 当作好笑 caption 提升。最终需要共同 seeds、
  image-clustered bootstrap 和独立 Group-of-10 盲评；`good / weak / bad` 绝对标签与
  相对 win-rate 必须分开报告。没有实际 judge packet 结果时，不得填写 good-caption rate。

### 15.1 本轮执行顺序（A5 已通过语义 gate）

```text
121/121 trace gate (pass)
→ clean preflight (pass)
→ A5 bridge-only training (job 6712327, complete)
→ A5 semantic gate (strong_go; epoch 5)
→ sealed outer semantic confirmation (job 6712454, outer_semantic_go)
→ Text-HOMER / full-plan-text / A5 latent caption generation (job 6712460, complete)
→ generation integrity + Caption-judgement audit (pass)
→ Group-of-10 mirrored multi-rater evaluation (complete; downstream A5 gain not supported)

本轮 A5 的可引用结果、精确 gap/CI 和方法证据边界见
[`PHASE_A5_RESULTS_ZH.md`](PHASE_A5_RESULTS_ZH.md)。A5 的语义 gate 只证明
channel-specific causal sensitivity；在盲评完成前，不得填写 good-caption rate、
win rate 或“更幽默”的结论。当前 Caption-judgement 运行目录为
`outputs/caption_judgement/a5_joint_group10_20260905_rubric_v11/`，其中 3,630 条候选已通过
`adapt/validate/audit`，484 个 packet（242 个镜像对）通过匿名性审计；prompt 已把
真实图片输入、逐 packet 独立评审、temperature=0、禁止私密映射/密钥和 rating 文件
归档责任写成硬性操作约束；三份独立评分已完成并聚合到 `aggregate.json`/`REPORT.md`。
A5 对 Text-HOMER 的 Overall win rate=`0.2982`，对 full-plan text=`0.5131` 且 CI
跨越 0.5；详见 [`A5_CAPTION_EVAL_RESULTS_ZH.md`](A5_CAPTION_EVAL_RESULTS_ZH.md)。
当前不启动 DPO；若继续，只做低成本 hybrid interface 修复和 caption-level causal audit。
```

### 15.2 预训练 Planner traces 续跑状态（2026-09-09，已完成）

原 trace 作业 `6719504` 使用 `b-batch + node=1`，调度器自动分配了 4 个 GPU，
但 `elapse=01:00:00` 到期（PJM code 11 / signal 24），并非 OOM 或方法错误。它
保留了 341/362 个唯一 trace；缺失集合为 21 个 cluster（`nycc_563`、`nycc_748`、
`nycc_762`、`nycc_877`–`nycc_889`、`nycc_891`–`nycc_895`）。统计文件显示最大显存
使用约 5.8 GiB，说明之前的独占四 GPU 请求对推理任务过度申请。

已在 clean commit `71f77ef` 增加 `jobs/cache_pretrained_homer_traces_repair_cgpu.pjm`：
只请求 `c-batch + gpu=1`、30 分钟、`retry-round=1`，只续跑上述 21 个缺失 cluster；
提交前 CPU preflight 已通过（`pretrained_homer_traces_repair_pre_submit_20260909.json`）。
续跑作业 `6738841` 于 2026-09-09 15:31:07 立即进入运行状态，成功补齐 20 条；唯一
残留 `nycc_563` 的失败原因为生成文本无法逐 token replay。随后在 clean commit
`0731f25` 提交 `jobs/cache_pretrained_homer_traces_retry_nycc563_cgpu.pjm`，仅请求
`c-batch + gpu=1`、20 分钟、`retry-round=2`、32 次有界尝试；作业 `6738906` 于
15:42:47 立即启动，15:45:06 以 exit code 0 完成。独立
`verify_pretrained_homer_traces.py` 现已报告 `362/362`、`missing=0`、`extra=0`、
`invalid=0`、`failure=0`。缓存保持 append-only；三组 Git commit provenance 均使用相同
的模型 revision、官方 prompt hash、输入 manifest hash，故不丢失可追溯性。

因此 **pretrained Planner trace gate 已通过**。下一步按顺序执行
`jobs/cache_pretrained_homer_context.pjm`：先在同一 GPU 作业中运行 1-cluster 的真实
context smoke，再生成完整的 `summary/retrieval/selection` context，最后运行
bridge-input validator；在这两项通过前不得启动 bridge 训练。`progress.json` 曾因单样本
续跑没有触发“每十条写回”条件而暂时显示 360，已在后续提交中修复为每次运行结束必写回，
以 `index.jsonl` 和独立 verifier 为权威结果。

### 15.3 context cache 的速度修正（2026-09-09）

真实 GPU smoke（1 cluster）通过，但完整单 GPU context 作业 `6739559` 的实测吞吐约
为每分钟 0.5–1 条；在 4 小时 walltime 内无法可靠完成 362 条。因此该作业在生成 8
条有效 context 后以 signal 15 取消，目录已原样隔离为
`data/cache/homer_pretrained_7b_homer_context_partial_6739559`，不得删除或覆盖。

后续不改变 HOMER 的三次调用协议（`summary → offline retrieval → select_conflict /
select_entities → seeded path`），而是将剩余 354 个 cluster 按全局 trace-input 顺序
轮转为 3 个各 118 条的分片，每个分片只请求 `c-batch + gpu=1`。新增
`jobs/cache_pretrained_homer_context_shard_cgpu.pjm` 和
`scripts/merge_pretrained_homer_context.py`：分片成功后必须由 merger 检查无重复/外部
cluster、trace/context/input/prompt/model hash 和 schema，再按 sealed 顺序生成 canonical
362-record context。`cache_pretrained_homer_context.py` 已改为使用全局 row offset 计算
seed，分片不会改变确定性 seed schedule。分片完成并通过 merger、bridge-input validator
前，仍不得启动 bridge 训练或 caption 评测。

#### 15.3.1 并发 GPU 可见性修正（2026-09-09）

复查三个并发 c-batch 分片后发现，旧脚本把 node-exclusive 路线使用的
`CUDA_VISIBLE_DEVICES=0` 复制到了 shared `gpu=1` 路线。三个作业均位于同一节点时，这
可能让进程都选择物理 ordinal 0，因而既不能证明使用了三个独立 GPU，也会降低吞吐。九州
大学 Genkai 官方 GPU-sharing 指南明确说明：仅申请单 GPU 的资源组不需要手动设置
`CUDA_VISIBLE_DEVICES`；只有多 GPU/sub-GPU 资源组才需要显式指定设备。因此已取消
`6739599/6739600/6739601`，保留各自已写入的 12/9/13 条有效记录，移除硬编码并在模型
加载前加入 `check_cuda_resource.py`（要求恰好一张 native、至少 40 GiB 的设备）。新的
提交必须先通过这个可见性门禁；缓存脚本会跳过已保留记录，不改变全局 seed schedule。
该资源修正不改变 HOMER 数据、prompt、模型 revision 或科学协议。

当前账号在 `c-batch` 的硬限制为同时运行 2 个 job、2 个 custom GPU、28 个 CPU core；
因此三分片实际采用“两路并发 + 第三路自动接续”，而不是申请更多 GPU。2026-09-09
重提交的两个作业在模型加载前均通过可见性门禁：`device_count=1`、设备为 H100 80GB、
`CUDA_VISIBLE_DEVICES` 保持调度器值（未被脚本覆盖）。第 3 个作业保持 accepted/queued，
不应因预测时间暂时较晚而重复提交；前两路释放资源后由 PJM 自动调度。

其中 `6739766` 在模型加载前的并发 pytest 临时文件竞态门禁中退出，已由
`test_public_bridge_route.py` 的 `tmp_path` 修复；它没有写入新的科学 context。修复后的
预提交门禁 commit 为 `8e5173d`（status=pass），当前有效作业为 `6739767`（shard 1）、
`6739768`（shard 2），`6739775`（shard 0，等待第二个运行槽位）。

#### 15.3.2 分片完成后的严格校验失败（2026-09-10）

三个分片作业均正常加载同一 adapter-free `Qwen2.5-VL-7B-Instruct`
（revision `cc594898...cfb5`），CUDA 门禁通过，且没有 OOM/NVML/资源错误；但作业的
最终 shard validator 按设计以 exit code 2 退出，因为 13 条输入无法形成合法的 HOMER
context。当前有效记录为：shard 0=`117/118`（1 failure）、shard 1=`113/118`
（5 failures）、shard 2=`111/118`（7 failures），加上已隔离 partial 作业的 8 条，
合计 `349/362`，canonical context 尚未生成。

失败均是 Planner 生成内容经过严格解析后的 schema/语义一致性错误，不是数据缺失：
`nycc_700` 的实体数不足两项；`nycc_575/815/579/717/892` 的 summary 不是 JSON
object；`nycc_655/613/622` 的 summary value 不是非空字符串列表；`nycc_668/678/723`
选择了不在 summary key 中的实体；`nycc_821` 选择了重复实体。原始分片
`index.jsonl`、`failures.json`、作业输出和 stats 全部保留，不得手工改写或把这些行
当成成功数据。

该结果表明当前 context 生成器已有严格 fail-closed 门禁，但还缺少统一的
validator-feedback repair。下一步只允许：保留原始 JSON，向同一 Planner 回传原始输出和
具体校验错误，仅要求 schema/引用修复且禁止新增或人工改写语义；修复输出必须带原始输出
hash、错误 hash、模型/prompt/seed/commit provenance，并只重试这 13 个 cluster。修复重跑
通过 `failures.json=[]` 且三 shard 的精确 118 条校验后，才可运行 strict merger；在此之前
禁止 bridge 训练、caption 生成和任何科学结论。详细事件见
`docs/EXPERIMENT_FAILURES.jsonl` 的 `V35-ENG-017`。

#### 15.3.3 有界 validator-feedback repair（2026-09-10）

本轮只恢复 `V35-ENG-017` 列出的 13 个 residual cluster，不重新遍历 362 条：
`manifests/pretrained_homer_context_repair_20260910_r2.json` 固定 cluster、来源 shard、
原始 error 和三个 `failures.json` 的 SHA-256。`scripts/cache_pretrained_homer_context.py`
在同一 Qwen2.5-VL-7B revision、同一 HOMER 原始 prompt 后追加一轮 validator feedback；
summary 只允许严格 JSON 序列化修复，entity 只允许映射到唯一已有 summary key。修复前后
语义字符串必须完全守恒，任何新增、删除、合并、重排、改写或伪造都会使该 cluster 失败。
原始响应、修复响应、SHA-256、seed、模型/adapter/prompt hash、repair policy 和 Git
commit 都保存于 context record/failures evidence。

作业 `jobs/repair_pretrained_homer_context_cgpu.pjm` 按 source shard 分三次提交，每次
只申请 `c-batch` 的一张 scheduler-visible native GPU，不设置 `CUDA_VISIBLE_DEVICES`，并
在模型加载前运行 CUDA、人口、trace 和 route gate。每个 shard 必须回到完整 `118/118`
且 `failures.json=[]`；否则保留 residual failure 并建立新的修复 manifest，禁止扩大范围。
只有三个 shard 全通过，才运行严格 merger 生成 canonical `362/362` context；随后重新
构建 bridge training view 并通过 `--require-bridge-data-ready`，才允许 bridge/caption
训练和 pure-text HOMER 对比。

本次提交前的只读资源检查显示 c-batch 当前无空闲节点，而 b-batch 有大量空闲 GPU
units；因此优先采用 `jobs/repair_pretrained_homer_context_bbatch.pjm`：仅申请一张
scheduler-visible GPU、串行修复三个 shard，设置 45 分钟短上限以便 backfill。该选择只
改变调度合同，不改变模型、prompt、seed 或数据范围；若调度器无法提供单卡可见性，
`check_cuda_resource.py` 会在模型加载前 fail-closed，不得改成多 GPU 或绕过 manifest
门禁。shared 作业显式声明 `#PJM -P exec-policy=share`，并在提交后用 `pjstat` 核对
预计启动时间；任何被排到未来时段的重复副本均在模型加载前取消并记录。

随后只读检查发现 `b-reserve` 有 1 个空闲节点（100/112 GPU units free），但提交立即被
`GENKAI2009` 拒绝：该组必须提供外部 reservation ID，本账户当前没有该授权。因此
`jobs/repair_pretrained_homer_context_breserve.pjm` 只作为 guarded fallback 保存，不得
再次尝试或伪造 reservation ID；在 b-batch/c-batch 给出可运行的授权 GPU 前，不提交新的
重复副本。

13:26 的再次验证提交 `6749312` 使用了 `b-batch + gpu=1 + exec-policy=share`，但
`pjstat -v` 预计启动时间为 09/12 21:00；`pjshowrsc` 同时显示当前空闲 b-batch 节点仅
暴露 `simplex/true`，没有可立即调度的 `shared/true` 节点。该作业在模型启动前以
`faster_start_not_available` 删除，未产生 stdout、模型加载或数据写入（详见
`V35-SCHED-014`）。因此当前没有活动 repair job；后续只在只读审计同时证明
`shared/true`/node-exclusive 合同可用且预计启动时间足够早时提交一个副本。空闲 GPU
总量不能替代 per-node capability 和 start estimate 检查；禁止继续堆叠队列作业。

随后以 `c-batch + gpu=1`、30 分钟上限提交单个 shard-0（`6749354`）验证共享路径；
该节点虽然有 40 个空闲 GPU 单元，但两个节点 CPU 均已分配满，`pjstat -v` 预计
09/11 10:00 才能启动。作业在模型启动前以 `c_batch_cpu_full_delayed` 删除，未产生
科学输出（`V35-SCHED-015`）。因此当前仍是“等待同时具备 shared GPU 与 CPU 容量的
授权槽位”，不是代码或数据失败；不得因 GPU 数字看似空闲而绕过 CPU/节点能力检查。

新增 guarded simplex 后备脚本 `jobs/repair_pretrained_homer_context_bsimplex_short.pjm`：
它只提交一个 shard、申请 `b-batch + node=1`、30 分钟，并在进程内暴露 CUDA 0。
该路由会占用整节点，只有在只读审计证明其启动时间早于 shared 路由时才允许使用；
否则必须保持不提交。脚本不会改变 repair manifest、模型、prompt、seed 或输出范围。

#### 15.3.4 simplex repair 的真实终态与 parser 修复（2026-09-10）

`6749375` 是一次真实获得 `b-batch` 独占节点的受控 repair 作业，不是排队或资源
失败。它在模型加载前通过 CUDA、population、adapter-free trace 三个 preflight，随后
加载固定 revision 的 Qwen2.5-VL-7B；但在唯一残余 `nycc_700` 上以 exit code 2 结束，
没有追加任何 context record。`.stats` 显示运行 5 分 01 秒、最大内存 6198 MiB，
没有 OOM、NVML、CUDA 或数据写入异常。

根因已经定位为工程 parser 缺陷：该 Planner summary 是合法 JSON payload，但生成边界
只留下了开头的 ` ```json ` fence、缺失结尾 fence。`_jsonish` 原先只接受完整成对的
fence，因而无法把原始 payload 交给 lossless semantic comparison。第一次 repair 保留
了全部五个 value，第二次响应删除了两个 value；后者被语义守恒校验正确拒绝。官方
HOMER summary prompt 是合并两路 imagination 后去重，并未规定 summary 每个 key 必须
恰好三个 value；因此禁止为满足旧的固定长度假设而截断或改写这些 value（官方实现的
retrieval/tree 也接受合并后的可变列表）。

本次只做格式层修复：`_jsonish` 现在接受未闭合的 opening JSON fence，并新增了来自
`nycc_700` 的回归测试；它不会放宽 summary 的对象/字符串列表 schema，也不会改变
HOMER prompt、模型、seed 或 semantic policy。事件和原始证据登记为
`V35-ENG-019`（见 `docs/EXPERIMENT_FAILURES.jsonl`）。完整 pytest、compile 和
preflight 通过后，才允许再次提交**唯一一个** manifest-scoped repair 副本；在三个
shard 都精确达到 `118/118`、`failures.json=[]` 并通过 strict merger 之前，context gate
仍为 `349/362 blocked`，bridge/caption 训练和科学评测继续禁止。

`6749447` 随后在模型加载前被 provenance 门禁拒绝（exit code 2）：`6749375` 已经把
新的 repair 尝试写回 shard-0 的 `failures.json`，旧 manifest 的源文件哈希和
`nycc_700` 当前错误字段随之过期。该终态不是数据或 GPU 故障。旧 manifest 保留作为
历史记录；新增 `pretrained_homer_context_repair_20260910_r2.json`，明确标注
`supersedes`、作业 ID、更新后的三份 failure hash 和当前 13 个错误，所有 repair
入口已切换到 r2。以后每次失败若使 `failures.json` 变化，必须先生成新的版本化 manifest
并完成 hash/target 校验，禁止原地改写 manifest 或绕过 provenance gate。

#### 15.3.5 r2 分片结果与 r3 有界重试（2026-09-10）

`6749861` 使用 r2 manifest 在 b-batch simplex 节点完成了 shard-1 的 5 个 residual
目标处理。作业通过 CUDA、population、adapter-free trace 和模型加载门禁，但最终为
`117/118`、`failures.json` 保留 1 条 `nycc_821`，因此以 exit code 2 结束。该错误不是
资源或模型故障：selector 输出了 summary key 中不存在且无法唯一引用已有 key 的
`engagement`。由于当前 policy 是 validator-feedback **format/reference-only**，把它
改成 `eagle`、`perch` 或 `tree` 会改变语义，故不能手工兜底；完整原始输出、修复输出、
错误和 hash 均保留，事件见 `V35-ENG-022`。

本次成功重试使可复用 context 达到 `354/362`（含 partial 的 8 条；shard-0=`118/118`、shard-1=`117/118`、
shard-2=`111/118`）。由于每次作业都会刷新 `failures.json`，r2 的源 hash 已过期；已新建
`manifests/pretrained_homer_context_repair_20260910_r3.json`，只锁定当前 8 条 residual
（shard-1 一条、shard-2 七条），并将 loader 的旧“必须 13 条”检查改为 `1..362` 的有界
检查。来源文件 hash 与 target/error 集合仍必须完全相等，所以不会扩大重试范围；所有
repair entrypoint 已切换到 r3。下一步只允许按资源策略提交 shard-1 的单条 retry，随后
再提交 shard-2；三 shard 精确达到 `118/118` 且 `failures.json=[]` 前，strict merger、
bridge 训练、caption 生成和科学评测继续保持 fail-closed。

`6750308` 使用新的 `retry-round=2` 于 `14:36:42` 启动并于 `14:39:34` 通过，shard-1
达到 `118/118` 且 `failures.json=[]`；`nycc_821` 以新的合法实体组合完成，无任何语义
手工映射，详见 `V35-ENG-024`。当前总量为 `355/362`（partial=8、shard-0=118、
shard-1=118、shard-2=111）。r3 的 shard-1 failure ledger 已改变，故新建 r4，仅锁定
shard-2 的 7 条 residual，并把下一轮设为 `retry-round=3`；只有 shard-2 通过后才合并。

`6750287` 的 r3 shared-GPU 副本经提交后预计到 `09/12 21:00`，在模型启动前按最快启动
策略取消（`V35-SCHED-016`）；b-batch 的 aggregate GPU FREE 数不能替代 shared 节点的
实际 backfill 能力。随后提交唯一的 simplex 后备 `6750288`（b-batch、node=1、CUDA 0、
30 分钟，预计 `09/10 15:00`），只处理 shard-1 的 `nycc_821`，不得与其他副本并行。

`6750288` 于 `14:30:30` 启动并在 `14:33:27` 结束，CUDA、population、trace 和模型加载均
通过，但仍返回 `nycc_821`。复核发现它沿用了 `--retry-round 1`，与 `6749861` 使用完全相同
的 seed namespace，故这不是独立随机重试；事件记录为 `V35-ENG-023`。r3 manifest 的
failure-ledger hash 未变化，因此保留 r3，只把全部 repair entrypoint 的 retry-round 改为
`2`，并再次限制为该单条 residual。未通过严格 shard validator 前，不得合并或进入
bridge/caption 评测。

`6750335` 使用 r4/`retry-round=3` 处理 shard-2 的 7 条 residual，其中 5 条通过、2 条
失败（`nycc_579` 的 `Royal demeanor` 不能唯一映射，`nycc_678` 会产生重复实体）。因此
当前 context 为 `360/362`（partial=8、shard-0=118、shard-1=118、shard-2=116），
详见 `V35-ENG-025`。已建立 r5 manifest，仅锁定这 2 条，并将下一轮提升为
`retry-round=4`；不得放宽 lossless validator 或手工改写实体，只有两条均通过才运行
strict merger。

`6750419` 已于 `15:05:09` 在 b-batch simplex 节点启动，使用 r5、retry-round=4，
仅处理 `nycc_579` 与 `nycc_678`。该作业是当前唯一活动副本；完成后先执行严格 shard
post-check，再决定是否合并 362 条 context。作业运行期间不得启动 bridge、caption 或
任何 v2.5/DPO 流程。

`6750419` 于 `15:09:38` 结束：`nycc_579` 修复成功，`nycc_678` 仍因重复实体被严格
拒绝，当前 context 为 `361/362`。事件记录为 `V35-ENG-026`；已创建 r6 manifest，
仅包含 `nycc_678`，下一轮使用 retry-round=5。成功前继续保持 bridge/caption 门禁关闭。

`6750455` 使用 r6/retry-round=5 仅处理 `nycc_678`，于 `15:19:04` 结束，仍为
`117/118`。这次暴露了共享解析器缺陷：validator 已能无损接受缺失 closing fence，
但 public-code `_json_object` 尚未同步处理该格式。该工程问题已记录为 `V35-ENG-027`；
已修复并更新 `homer_official_assets.json` 的源文件 hash，r7 只保留当前 residual，
retry-round=6。模型、GPU 和数据门禁均无异常，bridge/caption 继续锁定。
`6750482` 于 `15:28:34` 在模型加载前因 r7 manifest 的 `previous_error` 与当前
`failures.json` 不逐字一致而退出（`V35-ENG-028`）；无模型权重加载、无数据写入。
已保留 r7 证据并创建精确匹配的 r8，下一轮为 retry-round=7，仅处理 `nycc_678`。

资源策略复核：b-batch simplex `6750513` 的预计启动为当日 `15:50`；c-batch 探测作业
`6750522` 的预计启动为次日 `09/11 10:00`，因此已在执行前取消 c-batch，仅保留 b-batch
这一份副本（`V35-SCHED-017`）。

`6750513` 于 `15:44:16` 结束，模型/数据门禁均通过，但 `nycc_678` 的
validator-feedback 仍返回三项候选，违反“严格保留两个实体”的 reference-only 合约，
因此 shard-2 仍为 `117/118`；事件记录为 `V35-ENG-029`。当前 failure ledger 已变化，
已建立不可变的 r9 manifest（精确 hash 与错误串、retry-round=8），并完成本地脚本编译与
manifest gate。只允许提交 `jobs/repair_pretrained_homer_context_bsimplex_r9.pjm` 的一份
b-batch simplex 副本；成功前不得合并 context、训练 bridge、生成 caption 或运行评测。
