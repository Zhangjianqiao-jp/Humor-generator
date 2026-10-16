# 修正规格：先明确可检验的主张

本文件是审计建议，不表示生产实现已修改或验证通过。

## 1. 信息契约（F01，改变方法定义）

涉及 `training/public_bridge.py`、`scripts/cache_pretrained_homer_traces.py`、`scripts/generate_pretrained_bridge_comparison.py`。

若主张是比较通信表示，定义最终计划 P=(selected_conflict, selected_paths)，文本条件输入 serialize(P)，latent 条件输入冻结编码器对同一 serialize(P) 的状态。二者共享描述、样本、最终计划和生成预算。记录实际输入序列与编码条件，不能只记录来源 hash。编码器若可读取额外早期 trace 或图像，必须披露并增设等信息对照。

验收：改变未选中的早期候选、保持 P 固定，两条条件的计划输入均不变；改变 P，两者的计划输入均改变。这里验证的是输入契约，不保证两种表示的有效信息量相等。当前早期 trace 路线可以保留为单独系统比较，不能直接归因于通信表示。

## 2. 输出监督（F03，改变训练目标）

涉及 `training/public_bridge.py::target_caption_text`、`training/formal_bridge.py::_prompt_and_full` 及 loss mask。

已有数据只有 caption，没有 gold explanation。建议构造目标前缀 `##Caption: <gold caption>`，使用未结束的 assistant 序列；只监督 caption/必要格式 token，不监督紧接 caption 的 EOS，也不编造 explanation。所有 NLL、KL、matched/shuffled score 使用相同的有效位置定义，并核对 causal shift。

这是待验证方案：mask 掉 explanation 并不会教会模型生成正确解释。必须独立报告格式合规率；若需要 explanation 监督，应明确新增数据来源与质量控制。保留原始训练版本，不能覆盖历史结果。

验收：通过生产 processor 检查实际 input_ids、labels、loss mask；caption token 恰好监督一次，prompt/padding/不存在的 explanation 不受监督。用已知 logits 验证各 loss 实际读取的位置一致。

## 3. 解析（F02，修正评测输入处理）

涉及 `homer/official_prompts.py` 和评测建包入口。

保留官方兼容解析结果用于复现；新增带版本的严格解析轨道，输出 raw、caption、explanation、parse_status。缺 marker、拒答、空 caption 分开标记。失败仍保留在预先定义的样本分母中，不通过删除失败样本提高分数。严格解析轨道不能冒充未经修改的官方实现。

## 4. 工程修正（F06/F07/F09）

- checkpoint：先将“配置请求启用”与“实际执行”分开报告。真正启用前验证临时 hook 在 backward 重算期间仍存在，且 bridge 梯度与非 checkpoint 基线一致。不能只调用 model.train()。
- diagnostics：在 matched/shuffled forward 后分别保存对应诊断，明确命名。train/validation 对比相同条件。
- 聚合入口：按固定实验 manifest 校验 image、trial、system、reference 全集；拒绝非整数、bool、越界计数，禁止通过 int() 静默修正非法值。验收包含缺整轮、缺整张图片、重复键、2.9 和合法完整网格。

## 5. 科学结论（F04/F08，不改公式也必须改解释）

margin gap 增大既可能来自 matched 变好，也可能来自 shuffled 变坏。分别报告两者的绝对指标，并加入 no-memory/constant-memory 对照；这些只提供机制证据，不代替幽默质量评测。训练完成、机制阈值通过、主评测获益分别记录。

HOMER 复现保留官方参考组、Pass@K 和聚合方式；更换评审模型标记为替代评审。额外置信区间与敏感性分析单列。区间跨零不证明等效，也不能通过改写主指标得到成功结论。

## 论文依据与边界

- [HOMER, ICLR 2026](https://arxiv.org/abs/2602.06423)：生成与评测协议依据。
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)：基础架构依据。
- [BLIP-2, ICML 2023](https://proceedings.mlr.press/v202/li23q.html)：冻结模型接口训练的技术背景。

上述论文不证明这些项目修正能提升幽默质量；具体修正规格来自本地源码审计，效果必须另行验证。
