# v3.5 Engineering Smoke 状态

## 当前结论

旧 v3.0 的 GPU job、结果 JSON、显存数字和 checkpoint 均不属于 v3.5 证据，已经从本目录删除。v3.5 的独立真实 GPU engineering smoke 已于 2026-08-31 通过；尚未启动正式 bridge training。

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

该 gate 已通过。下一步可生成正式 train/validation traces；trace gate 完整通过后，只串行提交三个低成本 pilot，同一时刻最多一个正式 GPU 作业，不提交完整矩阵。

## 权威依据

- HOMER, ICLR 2026: https://openreview.net/pdf?id=SzaRhPom4o
- InterLat, ACL 2026: https://aclanthology.org/2026.acl-long.1248/
- StateBridge, COLM 2026: https://arxiv.org/abs/2608.13317
