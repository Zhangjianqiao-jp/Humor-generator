# 无训练接口串行 GPU 对比

原一小时作业 7102098 因预计排队到 10 月 20 日而取消，未运行。替换为每批 4 个 cluster、20 分钟上限的四批串行任务，首批 **7102101**，预计 2026-10-18 05:00 JST 启动（可能变化）。结果目录 `results/engineering_smoke/v35_training_free_serial_20261017_batch0` 到 `batch3`。尚无 GPU 通过结果。

## 本轮范围

固定 16 个训练 cluster（SHA256 seed `v35_effectiveness_diagnostic_20261017:`），不用 test，不训练 bridge，不调用付费 API。一个模型进程、一张 GPU，各 cluster/task/condition 顺序执行，不并发争用 GPU。

五个条件：

| 条件 | 输入 | 含义 |
|---|---|---|
| no_memory | description + 无计划提示 | 无计划对照 |
| text_plan | description + 原 selected plan 的 JSON 文本 | 相同内容文本基线 |
| exact_embeddings | 文本条件全部相同 token 的 embedding；通过 inputs_embeds 输入 | 接口一致性对照，不是新的 latent 方法 |
| soft_top64_replayed_plan | 固定 header 下 teacher-forcing 计划，在 t-1 logits 上取 top64 概率加权 token embedding，替换第 t 个计划 token | CIPHER-inspired，非原始 CIPHER；截断并重归一化的概率混合，记录选定 token 是否在 top64 中 |
| norm_hidden_replayed_plan | 同 token 的最终归一化 hidden state，范数校准到对应输入 token embedding | 直接 hidden-state 诊断，不是 StateBridge，没有训练或对齐矩阵 |

所有条件使用相同自定义 caption 指令、description、JSON plan 和采样参数；这不是 HOMER 官方提示的原样复现。计划片段保留全长，不声称压缩或提速。文本与 embedding 采用完全一致的分段 tokenize，保证 token 序列可对照；no-memory 不提供任何 selected-plan 内容。

## 两项任务

- Caption：每条件每图种子 20261017/20261018，temperature=1，top_p=1，最多 128 新 token。
- 计划恢复：每条件每图一次 greedy，最多 384 新 token，要求 JSON entities/conflict/paths。exact recovery 是严格字符串/结构相等，不代表语义恢复的完整评价；无计划时正确行为是返回 {}，不应把它恢复失败视作模型缺陷。

总计 16 × 5 × (2+1)=240 次生成。跨模型/跨条件的效果仍需人工或独立盲评；自动报告格式和严格恢复率不直接判断幽默。记录到达长度上限，不将其自动等同拒答或证实截断。

## 执行入口

- 共用实现：`scripts/compare_training_free_interfaces.py`
- 独立方案入口：`sh scripts/run_text_interface_controls.sh --output <new-dir>`、`sh scripts/run_soft_plan_interface.sh --output <new-dir>`、`sh scripts/run_normalized_hidden_interface.sh --output <new-dir>`。
- 一次串行运行全部条件：`jobs/v35_training_free_compare.pjm`，单节点，每批最多 20 分钟，默认 4 个 cluster。
- 脚本只接受 1–16 个训练 cluster；目录必须不存在。逐条持久化 `records.jsonl`；完成写 `summary.json`，完成状态不是质量通过。
- 固定模型 revision、NF4/BF16 和 source/config/data 哈希。已知低精度 cache 数值差异保持披露，不继续扩展精度支线。

## 必须保留的限制

soft 方案的每步分布来自计划前缀，既可能丢失低概率的已选 token，也不携带该位置之后的内容；它不是文本计划的无损编码，恢复实验正用于测量这点。不得把它的失败推论为所有 latent 方法失败。

直接 hidden-state 方案只校准范数，没有解决坐标对齐；失败时才进一步比较 StateBridge 风格对齐。正式 StateBridge 复现尚未实现。

计划编码用固定 header，接收者用 chat 模板；输入分布偏移是本诊断的一部分。caption 目标模板及输出长度均为探索性固定选择，不能在看到结果后只为一个条件调参。

## 参考

[CIPHER, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e444859b2a22df6b56af9381ad1e9480-Abstract-Conference.html)：概率加权 embedding 通信先例。
[StateBridge, 2026 preprint](https://arxiv.org/abs/2608.13317)：后续对齐方法参照；本轮归一化 hidden-state 不是其复现。

## 串行自动接续

`scripts/continue_training_free_batches.py` 已在 genkai0001 的 tmux `v35-recovery:latent-serial` 启动。只有前批 60 条结果齐备、源码哈希/offset 一致、调度器退出码为 0，才提交下一批。任一失败、源码改变、输出目录冲突或超过 72 小时就停止，不自动重试。此监控不会自动宣布质量通过或自动修改 PR。日志位于 `results/preflight/v35_effectiveness_20261017/serial_batches.log`。
