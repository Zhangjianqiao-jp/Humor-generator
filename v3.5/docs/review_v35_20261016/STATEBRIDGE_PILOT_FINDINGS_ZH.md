# 对齐 GPU 小试：首次恢复计划，仍非质量通过

作业 7103017 已退出 0，单个固定训练 cluster nycc_864，六组共 18 条生成完整。哈希和指标见 `statebridge_pilot_analysis.json`。

去除完整 JSON 围栏的次要内容指标：text_plan、exact_embeddings、statebridge_replayed_plan 均完整恢复 entities/conflict/paths；raw_hidden_replayed_plan、global_norm_hidden_replayed_plan 返回空对象。no_memory 正确返回空对象。原严格 JSON 指标不被覆盖。

这是同样本上完整对齐相对 raw/仅范数校准改善可读性的初步证据，不是 16 图成功率，也不是幽默质量提升。完整对齐包括白化、Procrustes、重着色、范数校准和词表锚定，不能把效果全归因于其中一步。对齐以该计划的 token 嵌入为参照，不能解释成恢复文本中不存在的信息。

## 资源和实现限制

82 个计划 token、维度 3584；发送端前向约 5.57 秒，对齐约 6.79 秒。本数值仅单次单图，未构成吞吐基准。两条 aligned caption 均格式有效，其中一条与文本 caption 相同；无独立质量结论。

GPU 记录 rotation determinant≈6.3613、max|RᵀR−I|≈0.00124955，而精确正交矩阵应满足 det=±1、RᵀR=I。因此不能把实际 GPU 变换描述为严格保距或官方实现逐元素等价。小维度 CPU 比对通过不能覆盖此高维 GPU 性质。保留原始证据，当前不扩展精度支线；后续若主张几何保持，必须另验证。此处主张仅限已观察到的计划恢复。

## 下一步

固定脚本与参数，在小试退出后串行提交下四个固定 cluster（offset=1、count=4），作业 **7106405**，输出 `results/engineering_smoke/v35_statebridge_expand_20261018_batch1`。预期 72 条生成，20 分钟上限；若失败保留部分输出，不假定完成。通过后再继续剩余 11 个，不能只报告这个成功案例。质量评审目前校准不充分，仍需原图条件或人工复核。

参考：[StateBridge](https://arxiv.org/abs/2608.13317)，无训练对齐候选；本实验为完整计划回放变体，非其完整多智能体流程复现。
