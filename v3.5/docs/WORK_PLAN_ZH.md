# v3.5 修正版实验计划

## 1. 研究问题与边界

核心问题不是“latent 能否替代文字”，而是：在 Planner 和 Generator 都冻结时，连续通信能否比等信息量文本/离散 token 更准确地传递 `Conflict + Local Association + Global Association`，并提高图片相关幽默 caption 的质量和角度覆盖。

本阶段只训练 bridge。Preference Learning、DPO、联合反传和两个 7B 的参数更新全部禁用。只有 latent 在未见图片上显示稳定收益后，才重新讨论 preference objective。

## 2. 两条不能混淆的系统线

### A. 严格 HOMER 文本线

`standard description -> conflict -> local/global imagination -> retrieval -> selected conflict/path -> caption`

Caption 阶段按 HOMER 公开 prompt 只接收 description/conflict/path。由于论文未公开精确 Qwen-VL revision，本项目只能声明为“固定 Qwen2.5-VL 替代模型的 method/data reproduction”。

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

语义恢复 Phase A 从 v2 起还必须实际优化跨样本表示判别：

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

## 7. Successive filtering，而不是一次性矩阵

### Gate E：v3.5 engineering smoke

- 真实 image + 真实 Planner trace；
- hidden/token/semantics/hash 全部通过；
- image-conditioned SFT receiver；
- Learned/Typed 参数和 24-slot 预算一致；
- policy trainable params=0；
- loss/gradient/update finite；
- 记录峰值显存。

Gate E 通过前禁止正式训练。

### Pilot P：只做 3 个 SFT-receiver pilot

每个仅 64 train clusters、24 validation clusters、1 seed：

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

主质量评测：121 张 adapter-unseen 图片、10 个共同 generation seeds、每条件 Group-of-10。它对齐 Humor in AI 的候选组规模；Group-of-3 仅作为历史敏感性分析，不承担主结论。每个比较生成 A/B 镜像方向，组内候选顺序独立随机化。评审必须支持看图，记录 provider/model/version/temperature/prompt hash。用 `build_judge_calibration.py` 从非 test 的官方 crowd ranking 构造五个清晰偏好示例；它复现官方 5-shot 校准思想，但不是论文每个测试项随机配五对的逐样本实现。本项目的 `Tie` 与绝对标签也属于额外扩展，因此应表述为“paper-aligned adaptation”，不能声称逐行复现官方 judge。

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
- 下一次正式训练：新的 64/24 Phase A v2 已满足工程门禁，待 clean commit 后提交；输出目录 `hierarchical_cross_attention_semantic_v2` 不覆盖 v1；
- pilot 真实生成评估：训练后自动生成 packet，但必须由独立评审完成才允许放大；
- preference learning：禁用。

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
