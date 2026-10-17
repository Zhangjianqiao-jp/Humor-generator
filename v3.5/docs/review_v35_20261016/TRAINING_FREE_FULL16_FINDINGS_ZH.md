# 五组接口实验：完整 16 样本结果

四批串行 GPU 作业 7102101、7102269、7102275、7102615 均退出 0。16 个固定训练 cluster、每图五组、每组一次恢复和两次 caption，共 240 条。严格分析检查通过；原始数据哈希见 `training_free_full16_analysis.json`。这是探索性训练集诊断，不是独立测试或 HOMER 官方复现。

| 条件 | 实体/路径恢复 | 完整计划恢复 | caption 格式通过 |
|---|---:|---:|---:|
| text_plan | 16/16，16/16 | 15/16 | 32/32 |
| exact_embeddings | 16/16，16/16 | 15/16 | 32/32 |
| soft_top64_replayed_plan | 0/16，0/16 | 0/16 | 32/32 |
| norm_hidden_replayed_plan | 0/16，0/16 | 0/16 | 31/32 |
| no_memory | 不适用 | 不适用；空对象正确 16/16 | 31/32 |

恢复表使用事后次要指标：仅去掉完整 JSON 围栏。原始严格 JSON 指标全部为零并保留在 JSON 报告。文本唯一不完全一致仍为 nycc_562 的 conflict 字符串转列表；不能简单判作语义丢失。两种 latent 的 16 条恢复结果都确实是空对象，而非其他 JSON 内容。

text_plan 与 exact_embeddings 的同 seed caption 32/32 相同；soft 与文本 2/32 相同；norm hidden 与文本在双方格式有效时 1/31 相同。相同率既不是幽默质量，也不能单独判定模型是否使用计划。

## 决策与执行

现有证据支持输入 embedding 通路可用、当前两种连续消息在恢复任务上失败。无法据此否定全部 latent 通信，也没有证据确认文本计划改善幽默。

已在第四批退出后提交单样本六组 StateBridge-inspired replay 小试：作业 **7103017**，20 分钟，一个 GPU，默认第一个固定训练 cluster，预期 18 条生成。输出 `results/engineering_smoke/v35_statebridge_pilot_20261018`，当前排队，未获得结果。比较原始 hidden、全词表范数校准和完整对齐，并保留无计划、文本和原词嵌入控制。先验证资源和可读性，再决定是否扩大；单图不能证明普遍效果。

输入脚本 SHA256 `6adfa9eea189805420e7a2e90e75da445b1cd34bc41816f58fb75e926eafaed9`；对齐模块 SHA256 `79420bae6ca3a1f935bc04ba5536c8bbc326c7c66b3e94eecb2976c916f0889a`。保持冻结。

完整盲评包已在本地 `results/preflight/v35_effectiveness_20261017/blind_full16/` 创建，96 对、交换顺序 192 条，评分全部为空。12 样本包保留作历史产物，后续完整评价使用 full16，不混入重复条目。

## 参考

[CIPHER, ICLR 2024](https://arxiv.org/abs/2310.06272)：概率加权消息先例，当前 top64 replay 是诊断变体。
[StateBridge](https://arxiv.org/abs/2608.13317)：无训练隐藏状态对齐候选；论文结果不替代本任务验证。
[Zheng et al., NeurIPS 2023](https://arxiv.org/abs/2306.05685)：评审顺序和其他偏差的依据，支持独立质量评价。
