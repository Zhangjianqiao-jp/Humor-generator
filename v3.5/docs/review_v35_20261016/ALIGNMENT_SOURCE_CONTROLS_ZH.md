# 对齐源状态贡献：九组控制实验入口

目的：区分真实 hidden 的贡献与同消息词嵌入参照 E(P) 的贡献。上一轮小矩阵反例已证明“能重建参照”不充分说明源状态具有有效语义；本轮把反例落实为 Qwen 生成控制，尚未进行 GPU 验证。

独立入口 `scripts/compare_alignment_source_controls.py`，基于冻结六组入口派生，记录父脚本 SHA256，不改动作业 7106405 使用的文件。

保留六组：no_memory、text_plan、exact_embeddings、raw_hidden_replayed_plan、global_norm_hidden_replayed_plan、statebridge_replayed_plan。新增三组：

- random_aligned_plan：CPU 独立 Generator，种子 20261018，标准高斯状态乘单一系数匹配真实 hidden 整体 RMS，再用相同 E(P) 对齐。它仍条件于长度和整体尺度，不能称完全不含任何输入相关信息；但不保留逐 token 内容与方向。
- permuted_aligned_plan：真实 hidden 的 token 行循环移位一格，E(P) 顺序保持原样，重新拟合对齐。所有索引都移动，但重复 hidden 可能使部分数值相同；这不是从完整排列集合随机抽样。
- global_norm_token_embeddings：直接 E(P) 按全词表平均范数缩放，不做对齐；用于检验同样范数尺度的简单嵌入基线。

默认同一个固定训练 cluster；每组一次恢复、两个 caption seed，共 27 条。生成提示、计划全长、固定模型、对齐 λ=0.001/α=0.3、proper_rotation=True 与原六组一致。对照生成使用独立随机数发生器，不改变 caption 的采样 RNG。新增对齐逐阶段落盘，分别记录耗时。

## 解释规则

如果真实/随机/移位状态都恢复计划，优先归因于参照对齐，而非源 hidden 专有语义；不能把恢复指标用于支持“保留文本之外的信息”。若真实 hidden 明显更好，也只能说明当前干预破坏了某些有用结构，仍需多图、更多控制随机种子及独立质量评价。一个随机种子不足以比较所有随机接口。

若完整对齐与范数匹配 E(P) 行为接近，应优先采用简单接口作为强基线，不能因对齐计算复杂就赋予创新优势。各组 caption 的质量未验证，不能由恢复或文本相同率代替。

## 验证与排程

CPU 检查通过：同种子重复一致、全局 RNG 不变、整体 RMS 匹配、移位索引正确。分析器的 11 项测试通过，包含控制模块/干预参数变化时拒绝合并批次。shell 语法检查通过。上述不等于真实 GPU 运行。

作业文件 `jobs/v35_alignment_source_controls.pjm`，20 分钟、单 GPU、默认 1 cluster。仅在既有 7106405 结束后串行提交，本轮未提交：

```sh
pjsub -x COMPARE_OUTPUT=results/engineering_smoke/v35_alignment_source_controls_pilot,MAX_CLUSTERS=1 jobs/v35_alignment_source_controls.pjm
```

参照：[StateBridge](https://arxiv.org/abs/2608.13317)。这些随机/移位控制是本项目追加的归因诊断，不是论文原始实验或正式复现。
