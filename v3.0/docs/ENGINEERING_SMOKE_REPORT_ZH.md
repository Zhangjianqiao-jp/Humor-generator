# v3.0 Engineering Smoke 报告

日期：2026-08-30（JST）

## 结论

本轮只验证工程可运行性，不构成模型质量实验，也没有启动正式训练。

- 当前 CPU tests：`34/34 passed`。
- CPU bridge optimizer smoke：通过。
- 最终 GPU job：`6642776`，`c-batch` 单张 H100，exit code `0`，wall time `00:01:03`。
- Base 与 SFT receiver 串行执行；两个 receiver 的所有参数均冻结，只有 typed bridge 可训练。
- caption-only logits、`use_cache=false` 与 matched/shuffled 两遍重算将峰值 reserved memory 控制在约 `6.46 GB`。
- hook 捕获状态严格保持“一 token 对应一 predictive state”，不允许裁剪或补齐。
- `generate(..., output_scores=True)` 的逐步分数与实际 emitted token 在 Base/SFT 上均为 `100%` 一致。
- KV-cache decode 与 teacher-forced full replay 的 hidden-state cosine 均值分别为 `0.999872` 和 `0.999859`。

## 三次 smoke 的诊断价值

| Job | 结果 | 暴露的问题 | 处理 |
|---|---|---|---|
| `6642764` | fail | 错把 NF4/BF16 下 cached decode 与 full SDPA 要求为逐元素近似相等 | 改为 processed-score/token 因果门禁 + replay cosine；保留数值差异报告 |
| `6642775` | partial | Base 通过；SFT raw LM-head 仅 `89.47%` 等于 emitted token | 识别到 generation logits processor 位于 raw head 之后；raw accuracy 降为诊断量 |
| `6642776` | pass | 无 OOM、无错位、冻结与 optimizer update 均通过 | 工程阻塞解除 |

这里没有“放宽到通过”：最终硬门禁直接比较 `generate()` 实际用于 greedy selection 的每步 scores 与 emitted token，要求 `100%`；teacher replay 另行验证同一因果位置。

## 最终数值

| Receiver | Generated states | Replay mean/min cosine | Processed token accuracy | Peak allocated/reserved | Bridge update |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-7B | 23 | 0.999872 / 0.999753 | 1.000 | 6.23 / 6.46 GB | 0.01036 |
| SFT 7B adapter | 38 | 0.999859 / 0.999478 | 1.000 | 6.26 / 6.36 GB | 0.01035 |

结果文件：

- `results/engineering_smoke/base_receiver.json`
- `results/engineering_smoke/sft_receiver.json`

## 当时尚未通过的科学门禁（已由后续台账与真实 trace smoke 更新）

1. GPU smoke 的 sender states 是合成状态，只验证 backward-through-frozen-receiver，不证明 latent semantic quality。
2. 最新增加的 receiver embedding norm calibration 已通过 CPU tests；由于 formal training 本身仍被 HOMER gate 阻断，本轮不额外占用 GPU 重跑。
3. HOMER 论文没有公开不可变 Qwen-VL revision、standard-description 来源清单、335,570-joke corpus/hash、精确 tokenizer/lemmatizer、模糊 entity merge 与主实验 embedding backend。
4. 因此只能称为“已实现论文公开算法”，不能称为“完整复现实验结果”。
5. 在这些门禁解除前，不提交 learned bridge 正式训练，也不进行 latent Group-of-3 质量比较。

## 真实 Planner trace 补充 smoke

后续 job `6643918` 用两张真实训练图片替代合成 sender states：

- 2/2 Conflict/Local/Global traces 通过严格 schema；
- multimodal global trace 使用完整 conversation 重新编码做 causal replay，禁止手工拼接 image token；
- frozen SFT receiver 的 policy trainable parameters 为 0；
- typed bridge trainable parameters 为 9,106,944；
- total loss 5.9792，gradient norm 403.68（更新前 clip），参数 update norm 0.01037；
- peak allocated/reserved 为 6.69/6.75 GB。

因此 synthetic-state 限制已解除，正式 bridge 训练器的工程门禁通过。它仍不证明 latent 比 text 更好；这一结论必须由 81 个 held-out image clusters 的 Group-of-3 比较给出。

## 正式生成路径补充 smoke

最终 job `6644022` 在一张 12 GB `b-batch-mig` 上执行 matched-text、StateBridge、Learned、Typed 四条真实推理路径并通过：

- receiver policy trainable parameters：0；
- 统一 sampling：`temperature=1.0, top_p=1.0, repetition_penalty=1.0`；
- peak allocated/reserved：10.88/10.98 GB；
- StateBridge 原始 322 tokens，按上限传输 causal-tail 64 tokens；sender/receiver rank 为 63/37，因此使用忠实 dense rank-mismatch fallback；
- 随机未训练 Learned/Typed bridge 分别产生 `user` 与空输出。空输出保留为 `[EMPTY OUTPUT]`，没有通过重采样隐藏。

这只证明四条生成路径可运行，不证明 latent 有效。随机 bridge 的无意义输出是预期的负面 sanity check；科学结果必须来自训练后 checkpoint 和 sealed test。

前两次生成 smoke 的失败也保留：job `6643987` 暴露空输出处理与 checkpoint 自带 repetition penalty；job `6644004` 暴露重复 token 导致 StateBridge sender/receiver rank 不等。修复没有更换固定 seed，也没有用不忠实的低秩伪逆替代 rank-mismatch 正交补。

## 方法依据

1. Shang et al. *On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation*. ICLR 2026. arXiv:2602.06423v2.
2. Du et al. *Enabling Agents to Communicate Entirely in Latent Space*. ACL 2026. ACL Anthology 2026.acl-long.1248.
3. Peng et al. *StateBridge*. COLM 2026. arXiv:2608.13317.
4. Hao et al. *Training Large Language Models to Reason in a Continuous Latent Space (Coconut)*. COLM 2025. arXiv:2412.06769.
