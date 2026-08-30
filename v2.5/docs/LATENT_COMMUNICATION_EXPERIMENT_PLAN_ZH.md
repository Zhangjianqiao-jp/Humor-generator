# 7B Planner → 7B Generator Latent Communication 严格实验计划

更新日期：2026-08-28

## 1. 决策与研究边界

本实验重新授权一种新的双模型路线，但不恢复已废弃的 7B→3B 系统：

- Planner：`Qwen/Qwen2.5-VL-7B-Instruct` + `outputs/newyorker_caption_aware_viewpoint_v3_7b_qlora/final_lora`；
- Generator：同一基座 + `outputs/7b-generator/best_val_loss`；
- 两个 SFT adapter 全程冻结；
- 第一阶段唯一可训练参数为 Latent Bridge；
- 已确认的“Planner 指导对 Caption Generator 有正向作用”不再通过 no-hint、random-hint、wrong-hint 等实验重复验证；
- 现阶段不训练 Planner，不联合更新两个 7B，不启动 joint RL。

核心研究问题不是 Hint 是否有用，而是：在同一个在线 Planner 和同一个冻结 Generator 下，连续通信是否比文本序列化传递更准确、更充分地表达图像矛盾、幽默机制和角度。

## 2. 三种通信条件

所有条件先使用同一个冻结 Planner 对每张图片做一次确定性在线推理。三种条件共享同一 Planner trace，避免 Planner sampling 成为混杂变量。

### T — Pure Text（SFT 联合推理基线）

```text
image → frozen 7B Planner → decoded compact plan text
image + plan text → frozen 7B Generator → caption group
```

Generator 只接收 plan 文本，不接收 Planner hidden states。这就是本实验中的“纯 SFT 联合推理”。

### L — Pure Latent（主实验）

```text
image → frozen 7B Planner → generated-token final-layer states
states → trainable Bridge → 16 continuous prefix slots
image + latent slots → frozen 7B Generator → caption group
```

Generator prompt 不包含 plan 文本。Planner 可以内部解码，因为逐 token hidden state 需要在解码过程中产生；但跨模型通信内容只有连续向量。因此这里的 pure latent 是“通信通道纯 latent”，不是“Planner 完全无 token 解码”。

### H — Hybrid Text + Latent

```text
image → frozen 7B Planner → {plan text, hidden states}
image + plan text + bridged latent slots → frozen 7B Generator
```

Hybrid 用于判断 latent 是替代文本，还是补充文本未能序列化的信息。

## 3. Bridge 方法

### 3.1 已实现的 Learned Query-Resampler

输入为 Planner 最后一个 Transformer block 在 plan 解码阶段的 hidden states：

\[
H_p\in\mathbb{R}^{T\times d_p}.
\]

Bridge 使用 16 个 learned queries，通过窄维度 cross-attention 将可变长度状态压缩为：

\[
Z=B_\psi(H_p)\in\mathbb{R}^{16\times d_g}.
\]

默认 bottleneck 为 512、8 heads。输出按 Generator embedding 的平均范数校准，然后作为被 mask 的连续 prefix 插入 Generator assistant answer 之前。占位 token 只用于维持 Qwen2.5-VL 的 multimodal RoPE、attention mask 和视觉 token 计数；其 embedding 会被连续向量替换，且不参与 caption loss。

代码：

- `src/latent_communication/bridge.py`
- `src/latent_communication/state_capture.py`
- `src/latent_communication/qwen_pipeline.py`
- `src/models/qwen_vl_dual_adapter.py`

### 3.2 与 StateBridge / Interlat 的关系

StateBridge 使用 whitening + Orthogonal Procrustes + norm calibration + vocabulary anchoring，并且是 training-free 方法。本项目采用它的两个关键接口思想：只捕获生成阶段末层状态，以及将对齐后的连续状态作为 receiver prefix；但当前主方法是有监督 learned query-resampler，不是 StateBridge 的逐行复现。

Interlat 使用 learned adapter/compression 使 agent 完全通过 latent state 通信，更直接支持本项目“冻结两端、先训练小型 bridge”的设计。

Pure Latent 主实验关闭 vocabulary anchoring，因为 anchoring 到 decoded plan token embeddings 会把离散词汇身份重新带入通信通道。StateBridge 的闭式对齐可以后续作为零训练 control，但不加入首轮三系统主比较，避免扩大实验矩阵。

## 4. Bridge-only 训练

### 冻结门禁

加载时只保留一份 7B base，并加载两个命名 adapter：`planner` 与 `generator`。每次计算显式切换 adapter。训练开始前程序检查：

```text
policy_trainable == 0
bridge_trainable > 0
```

任何 Planner/Generator 参数可训练都会立即失败。

### 数据与目标

- Train：`newyorker_compact_sft_v2/caption_train.jsonl`，79 个训练图片 cluster；
- Validation：`caption_validation.jsonl`，24 个完全不同的图片 cluster；
- 每个 epoch 每张训练图片随机选择一个高分 caption，防止把同一图片的上百条 caption 错当成独立视觉样本；
- Planner 对每个新图片在线推理一次。因为 Planner 冻结且 greedy decoding 确定，训练中可以缓存该图片的 trace；缓存记录 adapter hash、prompt hash、解码配置；
- Generator 输入不含文本 plan，Bridge 通过冻结 Generator 的 caption teacher-forcing NLL 获得监督：

\[
\mathcal{L}_{bridge}
=-\log G_{\theta_{SFT}}(y\mid x,B_\psi(H_p)).
\]

这不是端到端更新：梯度只经过冻结 Generator 回到 Bridge，Planner hidden state 在边界处 detach。

入口与配置：

- `scripts/train_latent_bridge.py`
- `configs/latent_communication/bridge_sft.yaml`
- `jobs/genkai_train_7b_latent_bridge.pjm`

当前默认 20 epochs × 79 image-cluster samples，约 1,580 examples；有效 optimizer batch 为 8。第一轮只使用一种 bridge 宽度和 slot 数，不同时搜索 layer、rank、slot、objective。

## 5. 第一阶段评价：DPO 前先隔离通信通道

### 5.1 生成协议

- 开发集：24 张 validation 图片；正式确认前不读取保留的 47 张 test 图片；
- Planner greedy 在线调用一次/图片，三种通信条件复用完全相同的 text/state trace；
- Generator seeds：`20260828/20260829/20260830`；
- 每个 system × image × seed 生成 3 条 caption；
- temperature 0.8、top-p 0.9、top-k 50、max-new-tokens 64；
- 在每个 image/seed/mode 前重置相同 RNG seed，避免某一路系统消耗随机数改变另一路结果。

入口与配置：

- `scripts/evaluate_latent_communication.py`
- `configs/latent_communication/eval_validation_group3.yaml`
- `jobs/genkai_eval_7b_latent_group3.pjm`

### 5.2 匿名 Group-of-3

每个 seed 独立构造以下位置平衡比较：

1. `Latent vs Text`：预注册 primary comparison；
2. `Hybrid vs Text`：latent 是否能补充文本；
3. `Hybrid vs Latent`：文本是否仍提供额外价值。

每组报告：

- Overall group win rate（Tie=0.5）；
- Best-pick win rate；
- `good / weak / bad` 绝对标签；
- image grounding、specificity、hallucination；
- Distinct-1/2、semantic diversity、组内重复率和 generic-template rate；
- 以图片为 cluster 的 bootstrap 95% CI；
- 三个 generation seed 的均值、标准差与最差 seed；
- 多评委一致性。评委不知道 system、seed、checkpoint 和 plan 文本。

统计单位必须是图片，不能把 24×3×3 条 caption 当成独立样本。Primary 检验为 Latent > Text；Hybrid 的两项为 secondary，使用层级检验或 Holm 校正，避免三次比较带来的偶然阳性。

### 5.3 Bridge Go/No-Go

进入通信条件 DPO 前至少满足：

1. Latent 或 Hybrid 对 Text 的 image-clustered mean win rate > 50%；
2. 95% CI 不显示明显负效应；
3. 至少 2/3 seeds 方向一致；
4. `good` 比例不下降，grounding/hallucination 不恶化；
5. 没有空 caption、模板坍塌或 latent prefix 数值异常。

若只有点估计略高但 CI 很宽，先扩充 validation 图片或增加人工评审，不直接用 DPO 把不确定差异放大。

## 6. 第二阶段：三种通信条件分别进行受控 DPO

Bridge-only gate 通过后，建立三个互相隔离的 Generator preference run：

| Run | Generator condition | Planner | Bridge | DPO 更新对象 |
|---|---|---|---|---|
| T-DPO | image + online text plan | frozen | none | 新 Generator preference LoRA |
| L-DPO | image + latent prefix | frozen | frozen best Bridge | 新 Generator preference LoRA |
| H-DPO | image + text + latent | frozen | frozen best Bridge | 新 Generator preference LoRA |

公平性约束：

- 三路都从同一个 `outputs/7b-generator/best_val_loss` 开始；
- 使用完全相同的 preference pairs、optimizer steps、effective batch、学习率、LoRA placement/rank 和 seed；
- 每个 pair 的 chosen/rejected 必须共享相同图片和同一个在线 Planner trace；
- 每种 condition 单独计算 SFT reference log-probabilities，禁止复用 Text route 的 reference logp；
- 首轮 DPO 固定 Bridge，不同时更新 Bridge 与 Generator，以免无法归因；
- 根据已经完成的完整 DPO 诊断，首选实验 objective 为 `conditional image preference + chosen anchor` 的明确实现；vanilla DPO 只作为已有基线，不继续无约束放大；
- 中间 checkpoint、validation early stopping、chosen/rejected absolute logp、image-clustered statistics 全部保留。

DPO 后形成六系统表：`T-SFT/L-SFT/H-SFT` 与 `T-DPO/L-DPO/H-DPO`。既比较每种通道内 DPO 前后，也比较同一训练预算下三种通道的最终质量。

只有 validation 上预注册 primary comparison 改善，才在 47-image test 上做一次三 seed 确认性 Group-of-3。Test 不用于选 bridge、checkpoint、DPO step 或 objective。

## 7. 结果解释边界

- `Latent > Text` 才支持连续通道保留了对下游 caption 有用、而文本未充分表达的信息；不能仅凭 Bridge train loss 更低得出该结论。
- `Hybrid > Text` 且 `Latent ≤ Text` 表示 latent 更适合作为补充而不是替代文本。
- `Latent/Hybrid` 只提高相对 win rate但 `good` 不提高，不能写成“生成了真正更好笑的 caption”。
- 本实验检验 downstream caption utility，不直接声称 latent 向量可解释为特定幽默机制。
- 两个 agent 同构、共享基座是当前有利条件；结论不能自动推广到异构模型。

## 8. 当前实现状态

- Learned bridge、latent slot 注入、generated-token state hook：已实现；
- 双冻结 adapter 共享基座 loader：已实现；
- Bridge-only 训练入口及冻结门禁：已实现；
- Text/Latent/Hybrid 三路在线生成：已实现；
- 24-image × 3-seed × Group-of-3 job：已实现；
- CPU tensor/unit tests：23 项相关测试通过；
- 首次 full-GPU smoke `6632091` 在冻结门禁通过后暴露 latent prefix 未同步扩展 `mm_token_type_ids` 的 multimodal RoPE 维度错误；full training 未启动。修复后增加回归测试；
- MIG GPU end-to-end smoke `6632109` 已通过：`policy_trainable=0`、`bridge_trainable=6,841,857`，在线 Planner、hidden capture、latent injection、Generator forward/backward 与 checkpoint 保存均成功；
- Bridge full training `6632115` 已在单张 `c-batch` H100 上启动；
- 三路 DPO：明确位于 Bridge Go/No-Go 之后，尚未开始。

## 8.1 五小时自动审计（2026-08-29）

服务器端使用独立 tmux watchdog，在启动五小时后自动执行以下流程：

1. 检查正式 Bridge 作业及 `best.pt`、`run_manifest.json`；
2. 等待既有依赖监控器提交并完成 Text/Latent/Hybrid Group-of-3 生成；
3. 校验 24 张 validation 图片、3 个 seeds、每组 3 条 caption，且确认未读取保留 Test47；
4. 输出空生成、组内重复、模板化率、Distinct-1/2 和长度的可视化诊断；
5. 为每个 seed 生成相互独立、隐藏系统身份的 LLM 盲评 packet；
6. 状态转为 `AWAITING_BLIND_RATINGS`，只有收到独立盲评且满足第 5.3 节门槛后，才允许进入通信条件 DPO。

自动化入口：

- `scripts/five_hour_latent_pipeline_review.sh`
- `scripts/analyze_latent_group3_outputs.py`
- `scripts/supervise_latent_pipeline.sh`：每两分钟检查两个 tmux 句柄，异常退出时幂等重启；五小时目标时间写入持久文件，重启不会重新计时。

## 8.2 冻结 Bridge 的三路盲评结果（2026-08-29）

作业 `6632173` 已完成 24 张 validation 图片、三个 generation seeds（`20260828/20260829/20260830`）的 Text/Latent/Hybrid Group-of-3 生成。Codex 在不读取 system key 的条件下先对匿名 group 固定 overall、best-pick、best caption index 与 `good/weak/bad`；ratings SHA-256 为 `9e0da129263b315dde27303a9bec18fd9409481ac859b5e97282dc8a34f8ad12`。随后才解盲。Tie 计 0.5，95% CI 使用以 24 张图片为 cluster 的 bootstrap。

| Comparison（前者相对后者） | Overall W-L-T | Overall tie-adjusted [95% CI] | Seed SD | Best-pick [95% CI] |
|---|---:|---:|---:|---:|
| Latent vs Text | 27-41-4 | 40.3% [29.2%, 51.4%] | 14.2% | 40.3% [29.9%, 50.7%] |
| Hybrid vs Text | 30-36-6 | 45.8% [38.2%, 52.8%] | 13.7% | 45.8% [38.9%, 52.8%] |
| Hybrid vs Latent | 43-25-4 | 62.5% [50.7%, 73.6%] | 14.6% | 59.0% [47.9%, 69.4%] |

绝对 `good` group 比例为 Text `27/72=37.5%`、Hybrid `22/72=30.6%`、Latent `13/72=18.1%`；以三个 seed 中至少两个为 `good` 定义 image-level majority-good 时，分别为 Text `8/24`、Hybrid `7/24`、Latent `2/24`。

预注册的 Bridge Go/No-Go 未通过：Latent 与 Hybrid 对 Text 的均值均未超过 50%，二者都没有至少 2/3 seeds 稳定胜过 Text，且 `good` 比例均下降。Hybrid 稳定优于 Latent 说明文本通道补回了纯 latent 丢失的信息，但不能据此声称 latent 为 Text 提供了净增益。

因此当前决定为 **No-Go：不启动 T-DPO/L-DPO/H-DPO 三路通信条件偏好训练，也不读取 Test47**。保留 bridge 与全部生成作为失败分析。若继续 latent communication，下一轮应先改变 bridge supervision/容量或增加信息保持目标，再在 validation 重新通过同一门禁；不能对当前 bridge 直接用 DPO 放大。当前结果来自一个独立 judge，论文级结论仍应补第二位盲评者并报告一致性，但单评委结果已经不足以授权后续 DPO。

完整产物：

- `results/latent_communication/validation_group3/CODEX_BLIND_SUMMARY.md`
- `results/latent_communication/validation_group3/codex_multiseed_summary.json`
- `results/latent_communication/validation_group3/codex_multiseed_summary.png`

代理指标只用于发现空输出、坍塌、模板化等回归，不能替代幽默盲评，也不能单独触发 DPO。

## 9. 权威论文依据

1. Peng et al. (2026), *StateBridge: Training-free Hidden-state Alignment for Latent Communication in LLM Multi-Agent Systems*, COLM 2026. https://arxiv.org/abs/2608.13317
2. Du et al. (2026), *Enabling Agents to Communicate Entirely in Latent Space*, ACL 2026. https://aclanthology.org/2026.acl-long.1248/
3. Hao et al. (2024/2026), *Training Large Language Models to Reason in a Continuous Latent Space* (Coconut). https://arxiv.org/abs/2412.06769
4. Aneja et al. (2019), *Sequential Latent Spaces for Modeling the Intention During Diverse Image Caption Generation*, ICCV 2019. https://openaccess.thecvf.com/content_ICCV_2019/html/Aneja_Sequential_Latent_Spaces_for_Modeling_the_Intention_During_Diverse_Image_ICCV_2019_paper.html
5. Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023. https://arxiv.org/abs/2305.18290
6. Wang et al. (2024), *mDPO: Conditional Preference Optimization for Multimodal Large Language Models*. https://arxiv.org/abs/2406.11839
7. Zhang et al. (2024), *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*, NeurIPS 2024. https://arxiv.org/abs/2406.10522
