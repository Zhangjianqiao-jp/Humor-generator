# v3.0 当前工作计划

## 决策

- `v2.5` 的正式 latent bridge 自动化已经停止。
- `v3.0` 在科学门禁通过前只允许 engineering smoke。
- Planner/Generator 两个 7B checkpoint 始终冻结；bridge 独立训练。
- 先完成 HOMER 文本系统复现，再引入 latent；两个阶段不得混称。

## Phase H0：HOMER 公开协议复现

1. 使用数据集自带 standard cartoon description，不自创 description prompt。
2. 使用论文附录逐字公开的 Conflict、Local/Global Imagination、Caption prompts。
3. Conflict 至少两对；Local/Global 每个 root 严格三步 chain；错误输出拒绝或重试。
4. 构建论文描述的 335,570 条 text-only joke corpus：评分低于 3 过滤、清洗、精确去重、80% word-overlap 近重复去重。
5. 实现 top-k joke retrieval、WordNet Wu-Palmer、conceptual-opposition、humor-frequency 和 POS-diversity；`k=5, delta=5`。
6. 随机选择 conflict、相关 target、DFS path；caption temperature=1。
7. 生成 5 candidates，报告 pass@1/3/5，重复 5 次。

只有 `scripts/check_reproduction_gate.py` 通过后，才允许把结果称为 HOMER reproduction。

## Phase L0：latent engineering

四个固定 baseline：

1. Text-HOMER；
2. StateBridge training-free alignment；
3. Learned latent bridge；
4. Typed learned bridge（Conflict / Local / Global 三通道）。

Learned bridge 的训练目标：

```text
L = 1.0 * caption NLL
  + 0.5 * text-teacher forward KL
  + 0.1 * matched-vs-shuffled margin
```

Base receiver 与 SFT receiver 分别训练、分别比较各自的 latent-vs-text，禁止把通过 SFT receiver 训练的 bridge 直接拿来对 Base 作公平性结论。

## OOM 修复协议

- Receiver 不计算 prompt/image prefix 的 vocabulary logits，只保留 caption causal logits。
- 开启 gradient checkpointing，关闭 KV cache。
- HOMER paper-aligned Generator 默认只消费 description/scripts/path，不重复输入图片，因此 bridge smoke 使用 text-only receiver context。
- smoke 最多 2 examples、1 optimizer step、1024 sequence tokens。
- 必须记录 peak allocated/reserved memory、finite loss、finite bridge gradients、policy trainable params=0。
- matched/shuffled 两个 7B receiver 图不得同时驻留：先无梯度计算当前 margin 系数，再对 matched 与 shuffled 分别重算、分别反传。该 surrogate 在当前参数点与原 softplus margin 具有相同的一阶梯度，但显著降低峰值显存。
- hidden state 采用“用于预测对应生成 token 的状态”语义；生成 hook 与无 cache teacher-forced replay 必须数值一致，数量不一致时禁止裁剪补齐。

## 当前 engineering 状态（2026-08-30）

- CPU/schema/loss/alignment/retrieval/baseline contract：`20/20` tests passed。
- CPU bridge optimizer smoke：通过，loss 与 bridge gradient 均有限。
- HOMER reproduction gate：按设计仍为 `BLOCKED`；缺少论文未披露的固定 Qwen-VL revision、benchmark standard-description provenance 与 335,570-joke corpus manifest/hash。
- GPU smoke：最终 job `6642776` exit code 0；只使用一个样本、一个 optimizer step，Base receiver 与 SFT receiver 串行使用同一张 GPU，避免并发占卡。完整诊断见 `docs/ENGINEERING_SMOKE_REPORT_ZH.md`。
- GPU 选择规则继承执行手册的“预计最短完成时间优先”：以 `show_rsc` 的真实 GPU free/total 为准，不使用 CPU `a-batch`，不以 `pjshowrsc` node free 代替 GPU 可用性。当前 12GB MIG 曾在 frozen-7B backward-through-input 路径连续 OOM，因此真实 bridge smoke 使用一张空闲 `c-batch` H100，而非重复无信息价值的 MIG 失败。

## Phase L1 的进入条件

- HOMER gate 通过；
- hook states 与 teacher-forced causal states 对齐测试通过；
- 单步 GPU smoke exit 0；
- StateBridge、Learned、Typed 三个 latent 通道均通过 matched > shuffled 的最小语义门禁；
- sealed validation/test 与历史 SFT 数据无 image-cluster leakage。

## 依据

- HOMER, ICLR 2026, arXiv:2602.06423v2。
- InterLat, ACL 2026, Anthology 2026.acl-long.1248。
- StateBridge, COLM 2026, arXiv:2608.13317。
- Coconut, COLM 2025, arXiv:2412.06769。
