# 7B→3B 联合推理与单独 3B 的 GPT 盲评报告

更新时间：2026-08-12 JST。

> **作废说明（2026-08-12）：** 本报告的联合组误用了新建的三行
> `ANCHOR/CONTRAST/ANGLE` prompt，而没有使用旧 v2.5 中效果较好的
> gold-caption-conditioned `hic-compact-json` 路线。因此这些数字只描述三行
> planner 消融，不能作为目标 compact-json 联合系统的结论。修正实验必须先用
> `gold-caption-minimal-viewpoint-v2` 生成 viewpoint JSON，再确定性渲染为
> `hic-compact-json` 后交给 3B；该修正实验属于使用 gold caption 的上限实验。

## 结论

在当前 24 张未见 test 漫画上，**没有证据表明 7B planner 提升了幽默 caption 质量**。点估计反而是单独 3B 更好：

| 指标 | 7B→3B 联合 | 单独 3B | 联合减单独 |
| --- | ---: | ---: | ---: |
| 好 caption（分数 ≥2） | 10/72 = **13.9%** | 15/72 = **20.8%** | -6.9 pp |
| 每图 3 个候选中至少一个好结果 | 10/24 = **41.7%** | 12/24 = **50.0%** | -8.3 pp |
| 平均分（0–3） | 0.764 | 0.819 | -0.056 |
| 至少相关（分数 ≥1） | 45/72 = 62.5% | 43/72 = 59.7% | +2.8 pp |
| 强结果（3 分） | 0/72 | 1/72 | -1 |

以图片为重采样单位做 20,000 次配对 bootstrap：

- caption 好结果率差的 95% CI：**[-19.4, +4.2] pp**；
- 图片命中率差的 95% CI：**[-37.5, +20.8] pp**。

两个区间都跨过 0。因此不能用这 24 张图断言单独 3B 在总体上显著更强；可以确定的是，当前实验**没有观察到联合方案的增益**。逐图最高分比较为联合胜 7、单独胜 7、平局 10。

## 公平对照设计

- test：clean-v2 的全部 24 张唯一漫画，训练/验证阶段均未见；不是从 4,415 条 caption 中重复采样图片。
- 两组使用同一份 step-400 3B best QLoRA adapter、同样的 3 candidates/image、temperature、top-p 和 seed。
- 联合组：图片 → step-300 7B best planner → 生成 plan → step-400 3B captioner。
- 单独组：图片 → step-400 3B captioner；prompt 只有生成一句短而图像相关的幽默 caption，不包含 plan。
- 每张图内部随机交换 A/B；GPT 只查看图片、匿名 A/B caption 和 candidate index。先固化 `blind_id → score`，之后才读取 private key 解盲。
- 推理作业 `6463755` 仅使用一个 MIG，运行 5:01，exit code 0。两组各 72 条输出全部非空且 prompt 泄漏为 0。

评分标尺：0=错图、明显不通；1=相关或流畅，但基本没有笑点；2=形成清楚、图像特异的轻度笑点；3=转折意外、简洁且明显好笑。“好结果”严格定义为分数 ≥2，不把流畅或提到图中物体直接算作好笑。

## 中肯的定性判断

单独 3B 唯一的 3 分结果是金字塔跳水台图片的：

> “Don't worry, he's only the first step of your pyramid scheme.”

它同时利用 `first step`、金字塔和 `pyramid scheme`，是本次最完整的视觉双关。联合组较好的例子包括末日废墟中的 “I wish we'd made a better plan.”、外星人观察狗的 “Your human friend is a good observer.”、地狱看电视的 “The whole thing is the result of global warming, of course.”。这些能结合图片，但大多只是轻度反转，尚未达到强笑点。

联合组的问题不是完全离题。它的 ≥1 分比例略高，说明 plan 能提供主题约束；问题是 planner 多数输出 `dry literal escalation`，容易诱导 3B 复述异常而不是完成 setup→punchline。7B 还明显误读了若干图：

- 把钢琴家及拳击手套看成“在桌前演讲”；
- 把海岛旁的外星海怪看成糖果船；
- 把山顶餐厅看成室内桌子与僧人；
- 把冰封猛犸看成穴居人画的大象；
- 把被洪水淹没的客厅简化为码头；
- 漏掉餐厅门口的 paparazzi。

错误 plan 会比无 plan 更危险，因为 3B 会服从这段显式错误信息。当前架构需要 planner confidence/grounding gate：高置信且图像一致的 plan 才注入 3B，否则退回直接生成。下一步优先做 `correct generated plan / swapped plan / no plan` 三组配对人评，而不是立刻 DPO；否则偏好优化可能只强化 planner 的错误条件。

## 限制

- 只有 24 张图片，置信区间很宽，本报告属于探索性工程评价。
- 只有一个 GPT 评委；幽默高度主观，正式结论需要至少 3 位盲评者，并报告多数票或平均分及评委一致性。
- “单独 3B”使用的是原本按 plan-conditioned 数据训练的同一 adapter，不是专门以 image→caption 重新训练的 direct baseline。这一设计隔离了推理时 plan 的边际贡献，但不是两种最优训练方案的最终对决。
- 每图三候选属于同一模型，caption 级样本不独立；因此主要看图片级命中率及按图片 bootstrap，而不能把 72 条当 72 个独立 test 样本。

## 可复现产物

- 联合输出：`outputs/newyorker_joint_vs_direct_3b/joint_captions.jsonl`
- 单独输出：`outputs/newyorker_joint_vs_direct_3b/direct_3b_captions.jsonl`
- 匿名候选：`outputs/newyorker_joint_vs_direct_3b/blind_candidates.jsonl`
- 解盲前固定评分：`outputs/newyorker_joint_vs_direct_3b/blind_scores.json`
- 完整逐条评分及统计：`outputs/newyorker_joint_vs_direct_3b/gpt_blind_evaluation.json`

## 权威依据

- Hessel et al., *Do Androids Laugh at Electric Sheep? Humor “Understanding” Benchmarks from The New Yorker Caption Contest*, ACL 2023 Best Paper: https://aclanthology.org/2023.acl-long.41/
- Hessel et al., *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
- Tanaka et al., *Content-Specific Humorous Image Captioning Using Incongruity Resolution Chain-of-Thought*, Findings of NAACL 2024: https://aclanthology.org/2024.findings-naacl.152/
