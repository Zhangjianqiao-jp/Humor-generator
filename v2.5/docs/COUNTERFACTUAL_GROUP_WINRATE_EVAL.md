# 7B→3B 反事实 Group Win-Rate 评测

评测日期：2026-08-21

## 结论

反事实实验确认：3B captioner 确实使用了 7B 的 compact plan；`type/target` 幽默桥具有可测量的因果作用，而不是一段无效冗余提示。但当前 7B 的正确 plan 相对 3B-only 只有弱优势，尚未在 24 张图上达到统计确定性。

| 比较（左侧系统胜率） | Overall | 95% Wilson CI | Best Pick | 95% Wilson CI |
|---|---:|---:|---:|---:|
| correct vs direct | 14/24 = **58.3%** | [38.8%, 75.5%] | 13/24 = **54.2%** | [35.1%, 72.1%] |
| correct vs swapped | 19/24 = **79.2%** | [59.5%, 90.8%] | 17/24 = **70.8%** | [50.8%, 85.1%] |
| correct vs target-corrupted | 18/24 = **75.0%** | [55.1%, 88.0%] | 18/24 = **75.0%** | [55.1%, 88.0%] |
| swapped vs direct | 7/24 = **29.2%** | [14.9%, 49.2%] | 8/24 = **33.3%** | [18.0%, 53.3%] |
| target-corrupted vs direct | 5/24 = **20.8%** | [9.2%, 40.5%] | 5/24 = **20.8%** | [9.2%, 40.5%] |

关键解释：

1. `correct > swapped`：图像与 plan 的匹配关系很重要，3B 不是单纯把 compact JSON 当作风格前缀。
2. `correct > target-corrupted`：即使保留正确的 scene、views 和 anchors，只替换 `type/target` 也会显著损害 caption，证明 humor bridge 是有效中间变量。
3. `swapped < direct`、`target-corrupted < direct`：3B 会服从错误提示；错误 plan 比没有 plan 更危险。
4. `correct > direct` 只有 58.3%/54.2%，且区间跨 50%：当前 planner 的净增益方向为正，但强度有限，不能宣称已稳定优于 3B-only。

## 系统条件

- `correct`：测试图片 + 7B final checkpoint 为本图生成的 compact plan。
- `swapped`：测试图片 + 由确定性一一错配产生的另一张图片的完整 plan。
- `target_corrupted`：保留本图正确的 scene、views、anchors，只从 donor plan 替换 type、target、external_knowledge。
- `direct`：仅输入测试图片，不注入 plan。

四路使用同一个 3B captioner LoRA、同一生成配置和同一随机种子。每个系统、每张图生成三条 caption。

## Judge 协议

- 参考 NeurIPS 2024 HumorousAI 的 Group Overall 与 Group Best-Pick 方法。
- 24 张留出图片，五个预注册比较，共 120 个匿名 trial。
- 每个 trial 只展示图片、匿名 Group A 和 Group B；不展示 plan、系统名称或 gold caption。
- 每组包含三条 caption，因此准确名称是 **NeurIPS-2024-style Group-of-3**，不是论文 Group-of-10 的严格复现。
- 每个比较中目标系统放在 A/B 各 12 次。
- 评分固定并通过 120/120 ID 完整性校验后才解盲。
- Overall 全局选择 A 63 次、B 57 次；Best-Pick 选择 A 61 次、B 59 次，未见明显总体位置偏差。
- 局限：单一 GPT/Codex judge；评审者在此前绝对评分中见过部分 correct/direct 文本，因此是位置盲评，但不是完全陌生的独立评审者。

## 对下一阶段训练的含义

本结果支持训练 7B planner 的偏好目标，但暂不支持同时更新两个模型。

推荐先固定 3B，以同图的计划偏好对训练 7B：

```text
chosen   = image-matched、视觉事实正确、能使3B产生高胜率caption的plan
rejected = swapped/target-corrupted/视觉误认/只复述画面的plan
```

偏好标签应由 3B 下游 Group Win Rate 或可靠 reranker 产生，并设置视觉正确性硬门槛。不能只奖励 plan 与 caption 的表面一致，否则系统会学习服从错误但自洽的 plan。

正式论文级结论还需要：

- 扩大到论文相近的至少 91 张留出图片；
- 每个系统每图生成 10 条；
- 使用独立 judge，最好增加人工评审；
- 同时报告 human median/top-10 校准和 caption diversity。

## 可复现文件

- 反事实输入构造：`scripts/build_counterfactual_plan_inputs.py`
- 匿名 group packet：`outputs/newyorker_caption_aware_v3_counterfactual_eval/group_eval_blind.jsonl`
- 固定盲选：`outputs/newyorker_caption_aware_v3_counterfactual_eval/group_eval_decisions.json`
- 私有映射：`outputs/newyorker_caption_aware_v3_counterfactual_eval/group_eval_key.json`
- 完整逐图报告：`outputs/newyorker_caption_aware_v3_counterfactual_eval/group_winrate_report.json`
- packet 构造脚本：`scripts/build_group_winrate_eval.py`
- win-rate 报告脚本：`scripts/report_group_winrate_eval.py`

## 论文依据

- Zhang et al. (2024), *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*, NeurIPS 2024 Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
- Hessel et al. (2023), *Do Androids Laugh at Electric Sheep? Humor Understanding Benchmarks from The New Yorker Caption Contest*, ACL 2023 Best Paper. https://aclanthology.org/2023.acl-long.41/
- Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
