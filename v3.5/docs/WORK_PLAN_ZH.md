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
- 每条 trace 还必须固定唯一的 Git commit、dataset manifest、HOMER prompt 源文件和 frozen-adapter manifest SHA-256。未提交或 v3.5 工作树不干净时，正式 trace 生成直接失败。

正式 train/validation trace 路径固定为 `data/cache/planner_traces_homer_strict_v35`。旧 v3.0 trace 禁止复制或引用。test trace 在模型/bridge 选择冻结后另行生成，避免测试集参与开发。

## 5. 数据切分与泄漏处理

| split | image clusters | caption rows | 用途 |
|---|---:|---:|---|
| train | 602 | 2295 | bridge fitting |
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

64/24 clusters 通过 `SHA256(seed, split, cluster_id)` 固定抽样，不按编号截断。Pilot 只判断架构/蒸馏是否值得扩展，不作论文最终结论。三个优化作业串行完成后，自动在 24 张 validation 图片上用 3 个共同 seeds 生成 `full-plan text / token embedding / Learned+KL / Typed+KL / Typed-no-KL`，并构造匿名、双向 Group-of-3 packet。流程随后停止等待独立评审；不得仅凭 validation loss 自动扩展。优胜条件同时要求 validation NLL、matched-vs-hard-negative margin和真实生成不退化。

### Confirmatory C：只扩展 pilot 优胜者

- 602 train / 64 validation；
- 至少 3 seeds；
- SFT receiver 为主；
- Base receiver 只为优胜架构补充训练，回答 receiver-specificity，不再做四乘四矩阵；
- early stopping 只看 validation，test 只在配置冻结后运行一次。

## 8. 生成与盲评

主质量评测：121 张 adapter-unseen 图片、10 个共同 generation seeds、每条件 Group-of-10。它对齐 Humor in AI 的候选组规模；Group-of-3 仅作为历史敏感性分析，不承担主结论。每个比较生成 A/B 镜像方向，组内候选顺序独立随机化。评审必须支持看图，记录 provider/model/version/temperature/prompt hash。用 `build_judge_calibration.py` 从非 test 的官方 crowd ranking 构造五个清晰偏好示例；它复现官方 5-shot 校准思想，但不是论文每个测试项随机配五对的逐样本实现。本项目的 `Tie` 与绝对标签也属于额外扩展，因此应表述为“paper-aligned adaptation”，不能声称逐行复现官方 judge。

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
- CPU tests：50/50 通过（以后以最新 test 输出为准）；
- 正式 v3.5 GPU trace/bridge smoke：作业 6649172 已通过；
- 正式 trace 生成：首次作业因补充 Git/prompt/data/adapter provenance 而主动取消，旧缓存已隔离；代码提交后从干净目录重新生成；
- 正式训练：未启动，后续由 trace gate 串行触发三个 pilot；每个 pilot 只申请 1 GPU、4 小时上限；
- pilot 真实生成评估：训练后自动生成 packet，但必须由独立评审完成才允许放大；
- preference learning：禁用。

## 12. 权威参考

1. Shang et al. HOMER. ICLR 2026. https://openreview.net/pdf?id=SzaRhPom4o
2. HOMER official implementation. https://github.com/Shang-hub/HOMER-Official-Implementation
3. Du et al. InterLat. ACL 2026. https://aclanthology.org/2026.acl-long.1248/
4. Peng et al. StateBridge. COLM 2026. https://arxiv.org/abs/2608.13317
5. Zhang et al. Humor in AI. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf
6. Hessel et al. Electronic Sheep. ACL 2023. https://aclanthology.org/2023.acl-long.41/
7. Tevet & Berant. Evaluating the Evaluation of Diversity in NLG. EACL 2021. https://aclanthology.org/2021.eacl-main.25/
8. Friedman & Dieng. The Vendi Score. TMLR 2023. https://arxiv.org/abs/2210.02410
