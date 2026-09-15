# v3.5 Engineering Smoke 状态

## 当前结论

旧 v3.0 的 GPU job、结果 JSON、显存数字和 checkpoint 均不属于 v3.5 证据，已经从本目录删除。v3.5 的基础真实 GPU engineering smoke 已于 2026-08-31 通过。Phase A3 replacement smoke 的第一次尝试 `6689653` 于 2026-09-03 在 forward/backward 前因 `492 > 384` 的 token 上限配置错误退出；将上限改为 `768` 后，新的真实 trace smoke `6706516` 于同日以 exit code 0 通过。该结果只关闭工程执行门禁，不是语义效果或 caption 质量结论。

已通过：

- 独立 Python 3.12 环境与 dependency lock；
- 禁止 executable code import/execute v2.5 或 v3.0；
- Planner/Generator adapters 的逐文件 SHA-256；
- 810 image-cluster 数据重建、hash 与 split leakage；
- Planner schema、predictive replay 与 post-token communication-state 对齐、Qwen-VL 图像 embedding/MRoPE、bridge loss、StateBridge dense-equivalence、Group-of-10/legacy Group-of-3 统计、24-slot budget、hard-negative 与官方 EAD/扩展 diversity 单元测试。

## 真实 GPU smoke 结果

作业：`6649172`，资源：单张 `c-batch` H100，exit 0。

- 真实 Planner traces：2/2 成功，0 failure；三个 channel 均保存真实文本、post-token states 与 SHA-256；
- policy trainable parameters：0；
- bridge trainable parameters：3,036,160；
- loss：5.9499；caption NLL：4.71875；teacher KL：2.30693；matched/shuffled margin loss：0.77687；
- gradient norm：62.27595，clip 后完成一步 optimizer update；parameter update norm：0.005981；
- backward smoke peak allocated/reserved：7.506/7.634 GB；
- 六条路径全部执行；generation smoke peak allocated/reserved：10.352/10.511 GB；
- full-plan 与 budget-text 产生正常文本；未训练的 token/state paths 产生乱码，随机 Learned/Typed bridge 立即 EOS。后者是未训练 bridge 的预期质量失败，只证明代码路径可执行，不计作 scientific evaluation。

证据文件位于忽略目录：

- `results/engineering_smoke/real_trace_typed_sft.json`
- `results/engineering_smoke/formal_generation_paths.json`
- `data/cache/planner_trace_smoke/index.jsonl`

基础 Gate E 与 trace gate（`666/666`）均已通过。正式的 `64 train / 24 validation`
channel-balanced semantic-recovery bridge-only pilot（job `6707953`）也已完成并正常退出：
engineering gate 通过，但 24-cluster semantic gate 为 `pilot_inconclusive`。两个 7B policy
在该阶段保持冻结。当前下一步是剩余 40-cluster outer semantic confirmation；只有它和
Phase A3 的工程/统计门禁共同满足，才解锁 Learned/Typed caption-level pilots；
DPO/preference learning 属于旧 v2.5 方案，在 v3.5 中禁用。

## Phase A3 replacement smoke（job 6706516）

作业在 `c-batch` 的单张完整 H100（`genkai0001`）上运行，使用修正后的
`max_target_tokens=768`，实际耗时约 4 分 20 秒，exit code 为 0。它选择了两个真实且
具有不同压力来源的 image cluster：`nycc_888`（最大原始图像）和 `nycc_325`（最长
完整 Planner memory），没有截断 global HOMER chain。

| 检查项 | 结果 |
|---|---:|
| real Planner traces | 2/2 |
| policy trainable parameters | 0 |
| bridge trainable parameters | 2,820,612 |
| contextual channel InfoNCE | 已执行（overall retrieval@1 = 0.6667） |
| channel weights | conflict/local/global 均为 1/3 |
| gradient norm / update norm | 2.17996 / 0.16644，均有限且 update 非零 |
| peak CUDA allocated / reserved | 9.57 / 13.16 GiB |
| validator | `status: pass` |

证据为 `v35_a3_smoke.6706516.out`、`v35_a3_smoke.6706516.stats`、
`results/engineering_smoke/cross_attention_semantic_phase_a3.json`。其中 report 的
`scientific_training=false` 是有意的：smoke 只证明真实 forward/backward、冻结策略、
完整 token 对齐、channel-wise counterfactual/InfoNCE 路径和资源可执行；其中出现的
 matched/shuffled gap 不能用于宣称 latent 已被 Receiver 语义使用。该 smoke 之后的
formal A3 已由 job `6707953` 完成；其完整结果、逐轮 validation 和语义 gate 位于
`outputs/pilot/cross_attention_semantic_phase_a3/`，详见 `docs/WORK_PLAN_ZH.md` 的
“Phase A3 正式结果”小节。下一步是 outer semantic confirmation，不是 caption 或 DPO。

## Formal Phase A3 完成状态（job 6707953）

正式训练使用 64 个 train clusters、24 个 validation clusters，只更新 2,820,612 个
bridge 参数，冻结两个 7B policy，5 个 epoch 共 80 optimizer steps；在 b-batch 单张完整
H100 上运行 25 分 06 秒并以 exit code 0 结束。validation total 从 4.12281 降到 3.00137，
caption NLL 从 1.80835 降到 1.12946，contextual InfoNCE retrieval@1 从 0.125 升到
0.535；但最终 channel-wise matched/shuffled gap 为 conflict −0.000040、local
−0.000317、global 0.001006，24-cluster bootstrap 区间均跨 0。因此 gate 是
`pilot_inconclusive` 而不是 technical `hard_no_go`：训练路径和表示对齐在工作，但低功效
pilot 尚未证明三路被 Receiver 稳定使用。不得据此启动 caption bridge 或 preference
learning；应先运行预注册的 40-cluster outer semantic confirmation。

## 权威依据

- HOMER, ICLR 2026: https://openreview.net/pdf?id=SzaRhPom4o
- InterLat, ACL 2026: https://aclanthology.org/2026.acl-long.1248/
- StateBridge, COLM 2026: https://arxiv.org/abs/2608.13317
