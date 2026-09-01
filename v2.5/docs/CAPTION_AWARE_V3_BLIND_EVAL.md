# 7B Planner + 3B Captioner 盲评报告

评测日期：2026-08-21

## 结论

本轮测试支持继续研究联合推理，但证据还不足以直接开始联合 DPO。

- 7B final + 3B：24/72 个好结果，**33.3%**。
- 7B best-val-loss + 3B：19/72，**26.4%**。
- 3B-only：16/72，**22.2%**。
- final 联合方案相对 3B-only 提升 **11.1 个百分点**，相对提升 50%。
- 以图像为 bootstrap 单位，差值的 95% CI 为 **[-6.9, +27.8] 个百分点**，跨过 0；24 张图不足以证明稳定提升。

因此，当前判断是：**联合路径有明确的正向信号，但还没有达到可以宣称有效或直接投入高成本联合 DPO 的证据强度。**

## 评测协议

- 测试集：New Yorker 留出测试集中的 24 张唯一图片。
- 系统：`joint_best`、`joint_final`、`direct`。
- 每个系统、每张图片生成 3 条 caption，共 216 条。
- 三套系统使用同一个 3B captioner LoRA、相同生成参数和相同 3B 随机种子。
- 联合系统仅更换 7B planner checkpoint；direct 不提供 planner 输出。
- caption 被随机匿名化；评分文件完整固定并校验 216/216 后，才读取系统映射。
- `gold_captions` 只作为评测记录元数据保存，没有进入推理 prompt。

评分标准：

- 1：不合格；无关、泛化、逻辑不通，或没有可辨认笑点。
- 2：好结果；明确落在图像关键关系上，并至少形成一个轻度笑点。
- 3：强结果；图像落点准确、简洁，有清晰双关、反转或跨域映射。

“好结果比例”采用 `score >= 2`。这是单一 GPT/Codex 评审的严格盲评，不等价于人群笑感。

## 完整结果

| 系统 | 好结果 | 好结果率 | 强结果率 | 平均分 | 24 张图中至少命中 1 条 |
|---|---:|---:|---:|---:|---:|
| 7B final + 3B | 24/72 | **33.3%** | 4.2% | 1.375 | 16/24（66.7%） |
| 7B best + 3B | 19/72 | 26.4% | 0.0% | 1.264 | 14/24（58.3%） |
| 3B-only | 16/72 | 22.2% | 1.4% | 1.236 | 13/24（54.2%） |

按每张图的三条候选取最高分，final 联合系统取得 7 次独占胜利，direct 为 5 次，best 联合系统为 2 次；其余为并列。

## 盲评中的代表样例

final 联合系统的强结果：

- 洞穴壁画与猛犸象：`I knew you were going to be big when I saw your cave paintings.`
- 戴拳击手套的钢琴家：`I think I should give him the win on points.`
- 客厅内铺设铁轨：`I think it's just a bit off the beaten track.`

3B-only 的强结果：

- 甜食形状的飞碟接近荒岛：`I was trying to land a job in your field but kept landing in space.`

失败仍然很多：联合 final 的 72 条中仍有 48 条不合格。典型原因是 planner 错认视觉对象，或只输出“异常描述”而没有可供 3B 使用的双义映射。例如图片 548 的 final plan 把火箭车错误解释成“像鱼一样被切开”，随后 3B 的三条 caption 全部没有形成可用笑点；direct 在同一张图反而生成了三条合格的超速梗。

## 是否进入 DPO

现在不建议立刻做“两个模型一起更新”的联合 DPO，原因有三点：

1. 样本只有 24 张，统计区间跨 0，尚不能排除本轮提升来自抽样波动。
2. planner 的视觉误识别仍是硬瓶颈；偏好优化不能可靠修复错误的视觉前提。
3. 一条 plan 对三条 caption 的影响是耦合的，直接联合更新会难以判断奖励究竟来自 planner 还是 captioner。

更稳妥的下一步：

1. 先扩展到至少 100–200 张严格留出图片，每图仍生成 3 条，并由 2–3 名盲评者评分。
2. 固定 3B，做 7B plan 的反事实一致性测试：正确 plan、换图 plan、无 plan 三路对比，确认 3B 确实使用了 plan。
3. 先训练轻量 reranker，输入图片、plan、caption，选出同图多个候选中的最佳项；这直接利用当前“每图至少一条好结果 66.7%”与“单条好结果 33.3%”之间的空间。
4. 若扩大评测后 final 联合方案仍稳定高于 direct，再为 7B 构造 plan 偏好对：`chosen` 是能导出高分 caption 且视觉事实正确的 plan，`rejected` 是错图、只复述画面或导出低分 caption 的 plan。先只做 planner DPO，保持 3B 冻结。

## 可复现文件

- 匿名候选：`outputs/newyorker_caption_aware_v3_joint_vs_direct_eval/blind_candidates.jsonl`
- 盲评分：`outputs/newyorker_caption_aware_v3_joint_vs_direct_eval/blind_scores.json`
- 解盲映射：`outputs/newyorker_caption_aware_v3_joint_vs_direct_eval/blind_key.json`
- 完整统计及逐条结果：`outputs/newyorker_caption_aware_v3_joint_vs_direct_eval/blind_evaluation_report.json`
- 统计脚本：`scripts/report_multisystem_blind_caption_comparison.py`

## 论文依据

- Hessel et al. (2023), *Do Androids Laugh at Electric Sheep? Humor Understanding Benchmarks from The New Yorker Caption Contest*, ACL 2023 Best Paper. https://aclanthology.org/2023.acl-long.41/
- Zhang et al. (2024), *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*, NeurIPS 2024 Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
- Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
