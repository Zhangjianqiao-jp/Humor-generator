# 7B Planner × 3B Captioner：方法、实验判断与下一阶段方案

> **历史文档 / 已废弃路线（2026-08-25）：项目已停止 3B Captioner 及 7B×3B 联合路线。自此之后所有新 SFT、Preference Learning、生成评测和模块实验只在独立 7B Generator（`Qwen/Qwen2.5-VL-7B-Instruct` + `outputs/7b-generator/best_val_loss`）上进行。下文仅保存历史实验依据，不得据此提交新的 3B 或联合训练作业。**

原始更新日期：2026-08-21；路线废弃日期：2026-08-25

## 1. 研究目标

本项目将幽默 caption 生成拆成两个可解释模块：

```text
cartoon image
    ↓
7B visual humor planner
    ↓ compact-viewpoint JSON plan
3B caption generator
    ↓
New Yorker-style caption
```

7B 的职责不是直接写 caption，而是识别图像的异常关系，并将其映射到可用于构造 punchline 的社会脚本、双义概念或语义桥。3B 同时读取图片和 plan，生成最终 caption。

今天的核心问题不是“两个模型是否都能运行”，而是：

1. 7B plan 是否给 3B 带来可测量的增量价值？
2. 3B 是否真正使用了 plan，而不是忽略它？
3. plan 中的 `type/target` 是否具有因果作用？
4. 错误 plan 是否会伤害生成？
5. 当前证据是否足以进入偏好优化？

## 2. 数据与模型状态

### 2.1 7B planner

- Base：`Qwen/Qwen2.5-VL-7B-Instruct`
- 方法：4-bit NF4 QLoRA
- LoRA：rank 8、alpha 16、dropout 0.05
- 模块：`q_proj/k_proj/v_proj/o_proj`
- SFT 训练数据：78 张图片、78 条人工/人工审定的 caption-aware compact-viewpoint 标签
- Validation proxy：24 张独立图片
- 当前使用 checkpoint：`outputs/newyorker_caption_aware_viewpoint_v3_7b_qlora/final_lora`

### 2.2 3B captioner

- Base：`Qwen/Qwen2.5-VL-3B-Instruct`
- 当前 LoRA：`outputs/newyorker_compact_v2_captioner_3b_qlora/best_val_loss`
- 输入：图片，可选 compact-viewpoint JSON
- 输出：短英文 caption

### 2.3 数据隔离原则

- SFT、偏好训练和最终评测必须按图片/contest 划分，不能按 caption 随机划分。
- 测试图片、测试 gold captions 和今天的盲评判决不能进入后续 DPO 数据。
- 推理 JSONL 中保存的 `gold_captions` 只是评测元数据，没有进入模型 prompt。

## 3. 7B SFT 方法与训练判断

### 3.1 配置

| 参数 | 数值 |
|---|---:|
| Epoch | 30 |
| Optimizer steps | 300 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| 名义有效 batch | 8 |
| Peak learning rate | `2e-5` |
| Warmup | 5%，约 15 steps |
| Scheduler | 线性衰减至接近 0 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Optimizer | paged AdamW 8-bit |

### 3.2 Loss

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 约 2.5 | 约 2.00 | 0.8898 |
| 5 | 1.771 | 0.7935 |
| 约 7.5 | 约 1.67 | **0.7818** |
| 10 | 1.585 | 0.7964 |
| 15 | 1.469 | 0.8404 |
| 20 | 1.401 | 0.8825 |
| 25 | 1.365 | 0.8979 |
| 30 | **1.352** | **0.9045** |

Token-level validation loss 在 step 75、约 7.5 epoch 最低，此后持续上升，最终较最低点恶化约 15.7%。训练 loss 则持续下降。因此，从标签 token 拟合角度，模型在 7–10 epoch 后开始过拟合。

但联合推理的实际盲评出现不同结论：final checkpoint 比 best-validation-loss checkpoint 更好。这说明当前 validation proxy 很小，而且 JSON token likelihood 与“这个 plan 能否帮助 3B 产生幽默 caption”不是同一个目标。以后不能只根据 validation loss 选择 planner checkpoint。

### 3.3 梯度

- 记录点梯度范数中位数约 0.838，平均约 0.795。
- 范围约 0.475–1.654。
- 只有四个记录点超过 1.0，且配置启用了 `max_grad_norm=1.0`。
- 无 NaN、Inf、持续上升或梯度消失。

判断：学习率和优化过程稳定；主要问题不是梯度，而是小数据上的目标错配和后期 token-level 过拟合。不继续用同一批 78 条数据追加 SFT epoch。

## 4. 第一阶段：绝对好结果率盲评

### 4.1 协议

- 24 张独立 New Yorker 测试图片。
- 三个系统：`joint_best`、`joint_final`、`direct`。
- 每系统、每图生成三条，共 216 条匿名 caption。
- 相同 3B LoRA、生成参数和随机种子。
- 评分先固定，之后解盲。

评分：

- 1：无关、泛化、不成立或没有可辨认笑点。
- 2：图像落点明确，至少有轻度笑点。
- 3：图像准确且有清晰双关、反转或跨域映射。
- `score >= 2` 计为 good。

### 4.2 结果

| 系统 | Good | Strong | 至少命中一条的图片 |
|---|---:|---:|---:|
| 7B final + 3B | **24/72 = 33.3%** | 3/72 | 16/24 = 66.7% |
| 7B best + 3B | 19/72 = 26.4% | 0/72 | 14/24 = 58.3% |
| 3B-only | 16/72 = 22.2% | 1/72 | 13/24 = 54.2% |

final 联合系统相对 direct 提升 11.1 个百分点，但图片级 paired bootstrap 95% CI 为 `[-6.9, +27.8]` 个百分点，跨 0。

### 4.3 该指标的作用与局限

绝对 good rate 能识别“两组都很差”的情况，也有利于失败归因；但评分阈值具有主观性，不能直接回答两个系统谁更好。因此它保留为辅助指标，不再作为模型比较的唯一主指标。

## 5. 采用 NeurIPS 2024 风格的 Group Win Rate

Zhang et al. 的 HumorousAI benchmark 将每个系统生成的一组 caption 做两种比较：

1. **Group Overall**：哪一组整体更幽默？
2. **Group Best Pick**：先从每组各选最佳 caption，再比较两个最佳项。

论文使用每组 10 条、91 张留出图片和 5-shot judge prompt。今天的实验每组只有三条，因此准确名称为：

> NeurIPS-2024-style Group-of-3 evaluation

这是一种适合当前反事实诊断的缩小版，不应声称是论文 benchmark 的严格复现。

## 6. 反事实设计

### 6.1 四个条件

| 条件 | 输入给 3B 的信息 | 研究作用 |
|---|---|---|
| `correct` | 图片 + 本图 7B final plan | 实际联合系统 |
| `direct` | 仅图片 | 无 planner 基线 |
| `swapped` | 图片 + 另一张图的完整 plan | 判断图像-plan 匹配的因果作用 |
| `target_corrupted` | 保留本图 scene/views/anchors，但换入 donor 的 type/target/external_knowledge | 隔离 humor bridge 的作用 |

完整 swapped 使用确定性一一错配：24 张图中 donor 不重复，且没有图片拿到自己的 plan。

### 6.2 生成控制

- 四路使用同一个 3B LoRA。
- 生成配置、候选数和随机种子一致。
- correct/direct 复用此前同 seed 输出，防止重新采样引入额外方差。
- swapped/target-corrupted 各生成 24×3 条。
- 四路合计 288 条，schema 全部合法，prompt leakage 为 0。

### 6.3 预注册比较

1. `correct vs direct`：planner 的净增量。
2. `correct vs swapped`：匹配 plan 的作用。
3. `correct vs target_corrupted`：`type/target` 的作用。
4. `swapped vs direct`：完整错误 plan 的伤害。
5. `target_corrupted vs direct`：错误 humor bridge 的伤害。

### 6.4 盲评纪律

- 120 个 trial；每个 trial 展示图片、匿名 Group A/B，各三条 caption。
- 不展示系统名、plan、gold caption、训练信息或旧分数。
- 每个比较中目标系统在 A/B 两侧严格各 12 次。
- 同一匿名 trial 分别记录 Overall 和 Best-Pick。
- 120/120 判决完整固定并校验后，才读取逐 trial 系统映射。
- Overall 全局选 A/B 为 63/57；Best-Pick 为 61/59，未见明显总体位置偏差。

限制：只有一个 GPT/Codex judge，而且 judge 在早期绝对评分中见过部分 correct/direct 文本，因此这是位置盲评，不是完全陌生的独立评审。

## 7. 反事实 Win-Rate 结果

| 左侧系统胜率 | Overall | Wilson 95% CI | Best Pick | Wilson 95% CI |
|---|---:|---:|---:|---:|
| correct vs direct | **58.3%** | [38.8%, 75.5%] | **54.2%** | [35.1%, 72.1%] |
| correct vs swapped | **79.2%** | [59.5%, 90.8%] | **70.8%** | [50.8%, 85.1%] |
| correct vs target-corrupted | **75.0%** | [55.1%, 88.0%] | **75.0%** | [55.1%, 88.0%] |
| swapped vs direct | 29.2% | [14.9%, 49.2%] | 33.3% | [18.0%, 53.3%] |
| target-corrupted vs direct | 20.8% | [9.2%, 40.5%] | 20.8% | [9.2%, 40.5%] |

## 8. 今天形成的专业判断

### 8.1 已被证据支持

1. **3B 会读取 plan。** 如果 3B 完全忽略 plan，correct、swapped 和 corrupted 应接近；实际 correct 对 swapped 为 79.2%。
2. **plan 必须与图片匹配。** 错图 plan 相对 direct 的 Overall win rate 只有 29.2%。
3. **`type/target` 是有效中间变量。** 在保留正确视觉 scene/anchors 后，只破坏 humor bridge，correct 仍以 75% 胜出；corrupted 相对 direct 只有 20.8%。
4. **错误 plan 比无 plan 更危险。** 当前 3B 具有明显的条件服从性，因此推理端需要 planner grounding/confidence gate。
5. **联合方向值得继续。** correct 相对 direct 点估计为正，但样本不足以证明稳定胜出。

### 8.2 尚未被证明

1. 不能声称联合模型已经显著优于 3B-only；correct-vs-direct 区间仍跨 50%。
2. 不能根据 24 张图估计真实生产胜率。
3. 不能认为 token-level validation loss 最低的 checkpoint 下游最好。
4. 不能认为 DPO 自动解决视觉误识别；错误视觉前提可能被偏好优化进一步强化。
5. 不能直接同时更新两个模型；目前无法可靠分配“差结果是 planner 还是 captioner 导致”的信用。

## 9. 下一步：固定 3B，构造 7B Planner 偏好数据

这是当前优先级最高的工作。不是立即运行已有的 3B DPO 配置，也不是把两套 LoRA 同时更新。

### 9.1 冻结基线

- Planner policy/reference 起点：7B `final_lora`。
- Captioner：固定当前 3B `best_val_loss`，整个 planner-DPO 阶段不更新。
- 冻结今天的 24 张测试图，只用于最终比较。
- 保存配置、adapter hash、seed 和生成参数。

### 9.2 在非测试图片上生成 planner 候选

每张训练图片建立同输入下的多个 plan：

- 7B final 随机采样 4–8 个 plan；
- 7B base/早期 checkpoint plan，增加能力层次差异；
- 人工 caption-aware plan，作为高质量候选而不是自动强制 chosen；
- 同图 `target_corrupted`；
- 一一 swapped plan；
- 视觉事实错误、只复述场景、过度泛化等 hard negatives。

DPO 的 chosen/rejected 必须对应同一张输入图片。swapped plan 可以作为该图片输入下的 rejected，因为它代表 planner 在该图片上不应输出的响应。

### 9.3 视觉硬门槛

在考虑幽默之前先检查：

1. JSON schema 合法；
2. scene 和 anchors 可在图中找到；
3. 关键对象、数量、角色和关系无误；
4. 不复制 gold caption；
5. target 不是单纯复述 scene；
6. 没有使用只在 gold caption 中出现、图片无法支持的实体。

视觉事实错误的 plan 不能因为偶然生成一条好 caption 而成为 chosen。

### 9.4 用固定 3B 测量 plan utility

每个 plan 让固定 3B 生成至少三条 caption。比较 plan 时使用 caption group，而不是单条最大分：

```text
image + plan_i → 3 captions → Group Overall / Best-Pick result
```

主信号采用同图 pairwise group preference；辅助记录：

- 绝对 good rate；
- 视觉错误率；
- plan-caption faithfulness；
- caption diversity；
- 相对 direct 的 causal gain。

不要只用 `max(score)`，否则一个偶然样本会错误抬高 plan。优先使用 group preference、平均表现或保守置信下界。

### 9.5 偏好对准入

只有下列情况才进入 DPO：

- chosen 通过全部视觉硬门槛；
- chosen 在顺序翻转后的盲比较中仍优于 rejected，或有多人一致判断；
- chosen/rejected 不是只有措辞差异，而是 utility 有明确差异；
- 每张图限制 pair 数，避免少数图片支配数据；
- train/validation 按 image/contest 隔离。

建议 schema：

```json
{
  "image": "...jpg",
  "image_id": "nycc_xxx",
  "prompt": "7B compact-viewpoint planner prompt",
  "chosen": "{...better compact plan...}",
  "rejected": "{...worse compact plan...}",
  "meta": {
    "pair_type": "downstream_group_preference",
    "overall_wins": 2,
    "best_pick_wins": 2,
    "visual_gate": "pass",
    "captioner_checkpoint": "frozen-3b-id"
  }
}
```

## 10. 7B Planner DPO 首轮建议

首轮目标是验证流程，不追求大幅更新：

| 参数 | 建议首轮 |
|---|---:|
| Policy init | 7B final SFT LoRA |
| Reference | 冻结的同一 final SFT checkpoint |
| 更新模型 | 仅 7B planner LoRA |
| Learning rate | `1e-6` 起步 |
| Beta | `0.1` 起步；后续比较 0.05/0.1/0.2 |
| Epoch | 1–2；最多 3 |
| Effective batch | 约 8 |
| Max grad norm | 1.0 |
| Checkpoint | 每 0.5–1 epoch |

这些是小数据、已有 SFT 起点下的保守首轮参数，不是通用最优值。模型选择不能只看 DPO loss，应按 held-out planner utility 和下游 group win rate 决定。

### 10.1 冒烟标准

- CPU 校验 schema、chosen/rejected 同图、无 test ID、无重复 pair；
- adapter/reference 完整且有限；
- 单 batch forward/backward；
- 单图生成 plan → 3B caption 的端到端冒烟；
- 通过后只申请 1 个 MIG；
- 不覆盖 SFT final adapter。

### 10.2 停止条件

立即停止或回滚，如果：

- 视觉事实错误率上升；
- correct-vs-swapped 下降；
- correct-vs-direct 没有改善；
- plan 变长但 causal gain 不变；
- DPO margin 增长而外部 win rate 下降；
- 输出开始复制训练 caption 或固定模板。

## 11. 验收设计

### 11.1 开发阶段

使用独立 validation images 比较：

- SFT-final planner；
- DPO planner；
- direct 3B；
- swapped；
- target-corrupted。

保持 3B checkpoint、seed、候选数、prompt 完全一致。

### 11.2 正式评测

为了接近 NeurIPS 2024：

- 至少 91 张严格留出图片；
- 每系统、每图生成 10 条；
- 报告 Group Overall 与 Group Best-Pick win rate；
- A/B 随机且位置平衡；
- 使用独立 judge，最好再加入人类评审；
- 以图片为统计单位报告置信区间；
- 对比 human median、#1000–1009、#200–209、top-10；
- 报告 caption diversity。

预注册的主要成功标准：

1. DPO-correct 对 direct 的 win-rate 95% CI 下界高于 50%；
2. DPO planner 显著优于 SFT-final planner；
3. correct-vs-swapped 不下降；
4. 视觉事实错误率不升高；
5. 增益在 Overall 和 Best Pick 中至少一个稳定存在，另一个不显著退化。

## 12. 后续顺序

建议执行顺序：

```text
冻结24图测试集与现有checkpoint
        ↓
非测试图生成多种planner候选
        ↓
视觉硬门槛 + 固定3B rollout
        ↓
Group preference / 人工复核
        ↓
高置信7B plan DPO pairs
        ↓
7B-only DPO smoke与训练
        ↓
反事实 + direct group win-rate复测
        ↓
训练(image, plan, caption) reranker
        ↓
只有在信用分配明确后，考虑3B DPO或交替优化
```

不建议当前直接做 joint trajectory-DPO。若 7B-only DPO 已稳定改善，下一阶段可以固定更新后的 7B，再对 3B 做 plan-conditioned caption DPO；最后才考虑交替优化，而不是同时更新。

## 13. 可复现资产

### SFT

- `configs/lora_sft_caption_aware_viewpoint_v3_7b_qlora.yaml`
- `genkai_train_caption_aware_viewpoint_v3_7b_qlora.pjm.6546345.out`
- `outputs/newyorker_caption_aware_viewpoint_v3_7b_qlora/`

### 初次联合评测

- `docs/CAPTION_AWARE_V3_BLIND_EVAL.md`
- `outputs/newyorker_caption_aware_v3_joint_vs_direct_eval/blind_evaluation_report.json`
- `scripts/report_multisystem_blind_caption_comparison.py`

### 反事实评测

- `scripts/build_counterfactual_plan_inputs.py`
- `jobs/genkai_eval_plan_counterfactual_v1.pjm`
- `scripts/build_group_winrate_eval.py`
- `scripts/report_group_winrate_eval.py`
- `docs/COUNTERFACTUAL_GROUP_WINRATE_EVAL.md`
- `outputs/newyorker_caption_aware_v3_counterfactual_eval/group_eval_blind.jsonl`
- `outputs/newyorker_caption_aware_v3_counterfactual_eval/group_eval_decisions.json`
- `outputs/newyorker_caption_aware_v3_counterfactual_eval/group_winrate_report.json`

## 14. 权威论文依据

1. Zhang, J. et al. (2024). *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*. NeurIPS 2024 Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
2. Hessel, J. et al. (2023). *Do Androids Laugh at Electric Sheep? Humor Understanding Benchmarks from The New Yorker Caption Contest*. ACL 2023 Best Paper. https://aclanthology.org/2023.acl-long.41/
3. Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
4. Shi, W. et al. (2024). *Direct Multi-Turn Preference Optimization for Language Agents*. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.138/
5. Song, Y. et al. (2024). *Trial and Error: Exploration-Based Trajectory Optimization of LLM Agents*. ACL 2024. https://aclanthology.org/2024.acl-long.409/
6. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html
7. Hu, E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. https://openreview.net/forum?id=nZeVKeeFYf9
