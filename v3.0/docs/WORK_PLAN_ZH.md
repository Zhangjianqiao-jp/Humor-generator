# v3.0 当前工作计划

## 决策

- 旧系统已冻结在 tag `v2.5-legacy-freeze-20260830` / commit `3927f18`，此后只读。
- v3 可执行目录由 `scripts/check_v3_isolation.py` 检查，禁止 import 或执行旧脚本。
- 两个 7B adapter 已复制到 git-ignored artifact 目录，并由逐文件 SHA-256 manifest 固定。
- Planner/Generator 两个 7B checkpoint 始终冻结；bridge 独立训练。
- 先完成 HOMER 文本系统复现，再引入 latent；两个阶段不得混称。
- latent 稳定优于 Text-HOMER 前，不启动 preference learning。

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

- 独立环境：Python 3.12.11；依赖锁定在 `requirements.lock`。
- CPU/schema/loss/alignment/retrieval/baseline contract：`29/29` tests passed。
- CPU bridge optimizer smoke：通过，loss 与 bridge gradient 均有限。
- HOMER 官方 standard descriptions 与 joke corpus 已固定到官方 commit；Qwen 原权重 revision 未公开，因此准确声明为“固定 Qwen2.5-VL 替代的 method/data reproduction”。
- GPU smoke：最终 job `6642776` exit code 0；只使用一个样本、一个 optimizer step，Base receiver 与 SFT receiver 串行使用同一张 GPU，避免并发占卡。完整诊断见 `docs/ENGINEERING_SMOKE_REPORT_ZH.md`。
- GPU 选择规则继承执行手册的“预计最短完成时间优先”：以 `show_rsc` 的真实 GPU free/total 为准，不使用 CPU `a-batch`，不以 `pjshowrsc` node free 代替 GPU 可用性。当前 12GB MIG 曾在 frozen-7B backward-through-input 路径连续 OOM，因此真实 bridge smoke 使用一张空闲 `c-batch` H100，而非重复无信息价值的 MIG 失败。
- 资源选择必须在提交后用实际 `START_DATE`/预计启动时间二次确认；本轮仅看瞬时 free 曾误判 b 组，已记录并纠正。

## Image-clustered 数据

- 唯一图片 cluster：810。
- train / validation / test：648 / 81 / 81 clusters；2535 / 294 / 291 caption rows。
- 三个 split 的 cluster overlap 均为 0。
- 冻结 Planner/Generator SFT 曾见过的 145 个 cluster 全部强制进入 train；validation/test 泄漏均为 0。
- Electronic Sheep contest 749 没有官方 finalist，显式记录为缺失，不伪造训练标签。
- 可追溯 manifest：`manifests/image_clustered_dataset.json`。

## 正式 bridge trainer

- Planner trace 先以固定模型、adapter、prompt 和 seed 生成，经严格 schema 与 causal hidden-state/token replay 校验，再保存 SHA-256。
- Qwen 的 fenced JSON 只做无损去 fence；Global 的 typed record 格式只接受精确 `entity + associations` schema，每条 chain 仍严格三步，多链全部保留。
- 每个 epoch 每个 image cluster 只抽一条 caption，避免多 caption 图片获得更大权重。
- loss：caption NLL + Text-HOMER teacher forward-KL + matched/shuffled caption-likelihood margin。
- matched 与 shuffled receiver graph 分两次反传，避免两份 7B graph 同时驻留。
- 每 epoch 保存 checkpoint，以 image-clustered validation total loss early-stop；训练过程写 `progress.json` 与 `metrics.jsonl`。
- Base receiver 与 SFT receiver 分别训练 Learned/Typed bridge，不交叉复用 bridge 做公平性结论。

真实 trace smoke job `6643918`：2/2 traces 合格；policy trainable params=0；bridge params=9,106,944；峰值 allocated 6.69 GB；有限 gradient 且 optimizer update norm=0.01037。

正式 trace 缓存 job `6643920`：仅 1 张 c-batch GPU，train+validation 729 clusters，可从 index 断点续跑。

## Phase L1 的进入条件

- HOMER gate 通过；
- hook states 与 teacher-forced causal states 对齐测试通过；
- 单步 GPU smoke exit 0；
- StateBridge、Learned、Typed 三个 latent 通道均通过 matched > shuffled 的最小语义门禁；
- sealed validation/test 与历史 SFT 数据无 image-cluster leakage。

## 正式比较与盲评协议

- Base receiver 与 SFT receiver 分开报告，不跨 receiver 比较 bridge 优劣。
- 四个主系统为 Text-HOMER、StateBridge、Learned bridge、Typed bridge；另保留 matched-text 作为分析对照。
- `Text-HOMER vs latent` 回答完整系统效用；`matched-text vs latent` 才回答在输入信息匹配后的 communication-modality 效应。不得把前者直接解释成 latent 编码优势。
- sealed test 共 81 个从未被两个冻结 adapter 见过的 image clusters；每个系统使用相同的 3 个 generation seeds。
- 文本与 latent 生成统一使用 `temperature=1.0, top_p=1.0, repetition_penalty=1.0`。后两项是中性值；必须显式覆盖 Qwen checkpoint 自带的 `repetition_penalty=1.05`。Planner trace 使用独立 greedy 路径，不受此次生成参数修正影响。
- 固定 seed 首 token 即 EOS 时记录 `[EMPTY OUTPUT]` 并作为 bad 进入盲评，禁止重采样，以免产生只保留成功输出的选择偏差。
- StateBridge communication prefix 最多 64 tokens，超长时固定保留 causal tail，并逐例记录原始/传输 token 数。
- 匿名 Group-of-3 的 A/B 映射单独保存。每位评审除整组 A/B/Tie 与整组绝对标签外，还必须给六条 caption 逐条标记 `good/weak/bad`。
- 主盲评直接展示图片，不展示 standard description；不支持图像的 text-only fallback 必须单列，不能用于主 grounding 结论。
- win rate 以图片 cluster 为统计单位，tie 计 0.5，使用 image-cluster bootstrap 95% CI；逐候选绝对标签用于计算 generation-seed 方差。
- latent 稳定收益至少要求：相对 Text-HOMER 或信息匹配的 matched-text 对照有一致方向、CI 支持，且绝对 `good` 率不退化。否则不进入 preference learning。

对应配置：`configs/evaluation/group3.yaml`；评审格式：`docs/GROUP3_JUDGE_PROMPT_ZH.md`。

正式推理前先执行真实 generation smoke。该 smoke 只验证四条代码路径，不构成科学结果，也不用于选择方法。

## 依据

- HOMER, ICLR 2026, arXiv:2602.06423v2。
- InterLat, ACL 2026, Anthology 2026.acl-long.1248。
- StateBridge, COLM 2026, arXiv:2608.13317。
- Coconut, COLM 2025, arXiv:2412.06769。
- Humor in AI, NeurIPS 2024, arXiv:2406.10522。
- Electronic Sheep, ACL 2023, ACL:2023.acl-long.41。
