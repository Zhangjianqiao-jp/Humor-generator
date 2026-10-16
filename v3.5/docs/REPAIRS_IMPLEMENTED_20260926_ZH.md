# v3.5 审计修复与新方法入口

用户已授权修改代码。本次修改实现代码修复与可验证的新路线；没有启动正式训练、GPU 作业或付费评审，没有覆盖历史 checkpoint、生成结果和审计证据。

## 使用哪个配置

- 新路线：`configs/pilot/cross_attention_caption_selected_plan_v2.yaml`。
- 历史路线：`configs/pilot/cross_attention_caption_pretrained_public.yaml`，保留原始 early trace 与裸 caption 监督，用于解释/复现历史实验。
- 训练必须显式传入 `--config`；比较生成和 CPU preflight 默认使用新配置。新路线拒绝旧 checkpoint，包括训练恢复、初始化和生成入口。需从新 bridge 重新训练。

## 实际改动

| 问题 | 实现与边界 |
|---|---|
| 最终计划未传入 latent | `training/selected_plan.py` 用冻结、eval 的 receiver 编码最终选定 conflict 和两条路径；训练和生成共用该函数。编码只读取选定文字，没有图像、caption、未选候选。编码状态为各输入位置的最后 decoder 层输出，并非旧 trace 的预测 token 状态。 |
| 三通道含义不清 | 新路线保留结构槽名 conflict/local/global；后两者明确代表 selected path 0/1，不再声称对应早期 local/global 规划器。修改最终选择会修改编码输入，修改早期 trace 引用不会影响它。 |
| 训练格式冲突 | 监督 `##Caption: <human caption>`，通过生产 processor 的 `continue_final_message=True` 保持 assistant 未结束，不监督 EOS/虚构 explanation。NLL、KL、margin 使用同一目标位置和 causal shift。 |
| 格式失败混入 caption | 新增独立 `homer/caption_parsing.py`，保留固定哈希的官方提示/解析文件不变。记录 raw、严格解析状态和官方兼容解析值。 |
| 失败样本被丢弃或偶然获胜 | 新配置使用明确标记的 strict-parser 扩展；失败仍保留候选数，在聚合时按失败计分，即使评审误判 A。官方兼容路径不应用该覆盖。 |
| checkpoint 报告失真 | `configure_frozen_receiver` 明确保持 eval、禁用 checkpoint 并如实报告。未采用会破坏临时 hook 重算的直接 train 模式修复。显存仍需真实 GPU smoke 验证。 |
| 诊断条件混用 | caption 与 semantic 分支及时保存 matched 诊断。caption 另存 shuffled entropy/relative update，原 `mean_*` 明确来自 matched。 |
| margin 被当成机制证明 | 新路线验证时额外计算相同输入和目标的 no-memory NLL，报告 matched 改善值。机制 gate 独立写入日志/checkpoint，缺失指标不能通过。生成完成只表示探索性结果已生成。 |
| 评测校验漏洞 | 拒绝 bool/小数/字符串计数，检查五轮、candidate/reference 索引和重复键。建包及主/辅助 CLI 对照独立 test population，能发现整图/整轮缺失。库函数未传 expected_images 时只声明 observed_ids_only。 |
| 新旧运行混用 | 生成行记录方法、解析器、config/context/trace/checkpoint 哈希与生成长度；恢复时核对合同。建包拒绝同系统混合合同以及文本/latent 不可比合同。 |
| 历史常量冒充新 provenance | 新建包不再硬编码旧 checkpoint、96 token 长度或旧评审日期；缺失资料标记未确定。 |

## HOMER 复现口径

官方 prompt、A/B 排列、参考组和 Pass@K 公式保持原样。新路线的严格解析与失败计分标记为 `homer_metrics_strict_parser_extension`，不能叫作原样官方评测。需要官方兼容评测时，在独立配置副本中将 `protocol.caption_parser` 设为 `official_compatibility_v1`，使用新的输出目录；不可混入同一生成文件。

`no_memory` 生成条件可作独立机制对照。当前官方两系统建包器仍只接收 text 与 latent 两条件；不要把三条件文件送入它。

## 已执行的检查

- 第一组 65 项回归通过，覆盖新修复与原有 bridge、prompt、Pass@K 行为。
- 完整测试集 163 项通过（33.49 秒）；随后新增的完整训练路径集成测试单独通过（1 项，4.71 秒）。该检查覆盖最终计划编码、teacher/student 目标一致性、KL/margin、bridge 梯度、冻结 receiver 无梯度和 no-memory NLL 对照。
- `scripts/check_partial_caption_target.py`：使用真实本地固定 revision processor，经生产 encode 与 `_prompt_and_full`，文本/latent 目标均为相同 8 token，解码为 `##Caption: I work for treats.`，不含 EOS；未加载模型权重。
- 新配置的 `verify_pretrained_bridge_inputs.py` 通过：362 contexts/traces，训练/验证/测试 caption 行数 813/132/141。
- 固定 pretrained route 检查通过；官方提示文件恢复原哈希，未重建数据缓存。

可重复的 CPU 检查（工作目录 v3.5）：

```bash
.venv/bin/python -m pytest -o addopts='' -q
PYTHONPATH=src .venv/bin/python scripts/check_partial_caption_target.py
PYTHONPATH=src .venv/bin/python scripts/verify_pretrained_bridge_inputs.py --config configs/pilot/cross_attention_caption_selected_plan_v2.yaml
```

未来新训练的入口为 `scripts/train_bridge.py --config configs/pilot/cross_attention_caption_selected_plan_v2.yaml --output <新的空目录>`；本次未执行。

## 尚未获得的证据

新方法的 7B/4bit 显存、长期优化稳定性和幽默效果尚未测量。编码最终计划增加了冻结模型的前向开销；不宣称更快或通信总成本更低。cache 在单个训练 task 内复用 CPU states。

caption 与最终计划的语义匹配仍未逐项标注；本次没有编造匹配标签或 explanation。部分目标监督也不保证自由生成时一定输出 explanation。16 次拒答的历史因果解释仍未证实。本次不保证方法优于文本，也不将有限测试视为全局无误证明。

## 论文依据

- [HOMER, ICLR 2026](https://arxiv.org/abs/2602.06423)：生成与评测协议基准。
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)：冻结基础模型架构背景。
- [BLIP-2, ICML 2023](https://proceedings.mlr.press/v202/li23q.html)：冻结模型与可训练接口的技术背景，不证明本项目有效。
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)：Pass@K 公式背景。

具体修正来自项目审计；论文不替代新方法实验。
