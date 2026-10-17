# StateBridge 对齐模块：实现与验证边界

## 当前状态

`scripts/statebridge_alignment.py` 已实现独立的无训练对齐计算；已有独立 Qwen 生成入口 `scripts/compare_statebridge_interfaces.py`，尚未在 GPU 执行，也没有加入正在排队的五组实验。不能称为 StateBridge 完整复现或效果通过。

参考 [StateBridge 论文](https://arxiv.org/abs/2608.13317)（arXiv 页面注明 COLM 2026 接收）及[官方实现](https://github.com/YanwenPneg/StateBridge/blob/3f6bf5442c6e8848555a6132516e6d36f35444fb/methods/state_bridge.py)。固定参考提交 `3f6bf5442c6e8848555a6132516e6d36f35444fb`。

## 计算与假设

输入 H、E 为逐 token 对应的隐藏状态与接收模型词嵌入，形状均为 N×d。分别中心化，计算 C_H=H_cᵀH_c/N+λI、C_E=E_cᵀE_c/N+λI；白化后求正交 Procrustes 映射 R，重建 H_c C_H^(-1/2) R C_E^(1/2)+μ_E。随后校准到全词表平均 embedding 范数，并以 α 混合最近余弦邻居的词嵌入。默认 λ=0.001、α=0.3。

论文正交约束允许反射；官方代码额外强制 det(R)=+1。本实现用 `proper_rotation=True` 对应官方代码，False 对应允许反射的目标。两者不能不加说明地混为一谈。整体白化、重着色映射不保证在原始空间保距。

这里使用当前消息的已知 token 嵌入作为对齐目标，因而不能将对齐后信息恢复解释成“恢复文本未包含的信息”。当 N 远小于 d，中心化样本协方差秩至多 N−1；正则化使运算可行，但不能证明未观测方向的语义对应关系。完整 d×d 分解的耗时也须在 GPU 上实测。

## 已有验证

- 六项 CPU 数学测试：反射约束、同空间方向及范数、词表分块和并列处理、α=1、退化消息有限性、长度不匹配拒绝。
- 与官方 `_align_hidden_sequence` 的三个 float32 CPU 样例比对，最大绝对差 1.1920928955078125e-7，容差 atol=rtol=2e-5。详见 `statebridge_alignment_validation.json`。
- 输入要求严格逐 token 配对；不像官方实现那样自动取最短长度。最近词表搜索分块以限制中间矩阵内存。

## 后续 GPU 接入门槛

现有五组、四批串行实验继续使用固定脚本。后续对齐实验单独版本化：固定同一计划和接收提示，对比原始 hidden、范数校准 hidden、完整对齐，并保留 text 与 exact embedding 控制。记录对齐开销、输入长度、token 对应和全部参数；先做单 cluster 资源检查，再扩展。若采用计划回放而非官方发送端生成，应命名为 StateBridge-inspired replay，不能声称官方复现。

先看实体、冲突、路径恢复及格式，再用盲评判定幽默质量。恢复率、训练样本小试和对齐数值一致性均不能替代质量证据。概率加权接口的对照依据另见 [CIPHER, ICLR 2024](https://arxiv.org/abs/2310.06272)；当前 top64 replay 也是诊断变体。

## 独立生成入口（待 GPU 验证）

`compare_statebridge_interfaces.py` 是已冻结五组脚本的独立派生版本，文件头记录来源 SHA256。六组为 no_memory、text_plan、exact_embeddings、raw_hidden_replayed_plan、global_norm_hidden_replayed_plan、statebridge_replayed_plan。相同模型、计划全长、提示、两个 caption seeds 和一次 greedy 恢复。每个 cluster 共 18 次生成；默认只运行一个 cluster 做资源与接口验证。

相邻比较含义：raw→global_norm 测试范数校准；global_norm→statebridge 测试白化、旋转、重着色和词表锚定的组合，不能把组合效果全归于旋转。当前不做 K=64 截断，避免截断与对齐同时改变。与原五组归一化方案也不同：原方案使用每个对应 token 的范数，此处使用全词表平均范数。

启动文件 `jobs/v35_statebridge_compare.pjm`，20 分钟、一个 GPU。须在既有四批结束后再提交，当前未提交。工作目录为 v3.5：

```sh
pjsub -x COMPARE_OUTPUT=results/engineering_smoke/v35_statebridge_pilot_20261018,MAX_CLUSTERS=1 jobs/v35_statebridge_compare.pjm
```

`alignment_events.jsonl` 在分解前后落盘，用于区别分解未完成与生成失败。manifest 记录对齐模块哈希和固定参数；分析器拒绝合并参数或模块哈希不同的批次。准备时间是各组共享开销，不作为某组独有耗时。资源小试完成后再决定是否扩展到完整 16 个 cluster。
