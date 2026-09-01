# v3.5 程序错误与实验执行风险评估

更新时间：2026-09-01 15:43 JST。本文只评估 v3.5；v2.5/v3.0 不属于可执行依赖。

## 结论

当前没有发现需要取消作业 `6666633` 的确定性训练逻辑错误。作业仍在 `c-batch`
队列，调度器预计 20:00 JST 启动；因此“native allocator + 单完整 GPU”修复尚未获得
真实运行证据，不能提前称为已解决。作业内部会先运行 exact-first-example resource
smoke，失败即停止，成功才训练 P1。

风险总等级：**中等，可继续，但必须保持 evidence gate**。

## 已核验事实

- 数据 gate：train/validation trace 为 `666/666`，缺失、重复、额外及 failure 均为 0。
- split gate：train/validation/internal-test/official-unseen 两两 cluster 交集均为 0；
  `official_hia_seen_diagnostic` 不进入主要结论。
- 冻结 gate：训练入口要求 receiver trainable params 为 0，bridge params 大于 0。
- 公平性：Learned 与 Typed 均为 24 slots、3,036,160 个 trainable parameters；三个
  pilot 使用相同 64 train clusters、24 validation clusters、seed 和优化预算。
- loss：caption NLL、text-conditioned forward KL 和 matched/shuffled margin 的两阶段
  backward 与原一阶梯度符号一致，并避免同时保存两个 7B receiver graph。
- 自动化：P1/P2/P3 串行；现在不再以单独存在的 `complete.json` 判定成功，而要求
  checkpoint、manifest、metrics、finite validation 和 receiver-freeze contract 同时通过。
- 评测 packet：必须与 private mapping 具有相同且唯一的 blind IDs，才可进入独立盲评。
- 回归测试：63 项通过；Python compile、全部 PJM/shell 语法、memory validation 通过。

## 风险矩阵

| ID | 风险 | 概率 | 影响 | 当前控制 | 状态 |
|---|---|---:|---:|---|---|
| R1 | PyTorch/NVML/MIG allocator 再次崩溃 | 低（完整 GPU） | 高 | 禁止正式 MIG；native allocator；固定 `cuda:0`；>=40 GiB gate | 待真实 smoke |
| R2 | 完整 GPU 路径仍发生真实 OOM | 中低 | 高 | exact P1 首样本完整 backward smoke；两阶段 graph；4-bit receiver | 待真实 smoke |
| R3 | `device_map=auto` 隐式跨设备切分 | 低 | 高 | 全部 Qwen backend 固定 `{"": 0}` | 已控制 |
| R4 | P1 新路径与评测旧 checkpoint 路径不一致 | 低 | 高 | 评测和人工 fallback 已统一为 `learned_sft_kl_native_full`；有回归测试 | 已修复 |
| R5 | 仅凭残缺 `complete.json` 自动提交下一阶段 | 低 | 高 | monitor 验证四件套与数值/冻结 contract | 已修复 |
| R6 | 4 小时 walltime 在 epoch 中途终止 | 中 | 中 | 每 epoch checkpoint；输出不删除；monitor fail-closed | 开放 |
| R7 | validation generation 4 小时不足 | 中 | 中 | 每个 condition JSONL 按 cluster×seed 可恢复；packet 不完整不会被接受 | 开放 |
| R8 | tmux/登录节点中断使流水线不继续 | 中低 | 中 | state JSON 幂等记录 job ID；可从状态恢复，不影响正在运行的 PJM | 开放 |
| R9 | 新依赖组合的上游回归 | 中低 | 中 | 独立锁定环境、版本校验、真实 smoke；不在正式运行中升级 | 开放 |
| R10 | pilot 单 seed 被误写为最终结论 | 中 | 高（论文） | pilot 只筛选；最终 Group-of-10、共同 10 seeds、多评审、cluster bootstrap | 方法 gate |
| R11 | LLM judge 文化偏差或位置偏差 | 中 | 高（论文） | 图像可见、五例校准、镜像 A/B、多评审、绝对标签、rater agreement | 方法 gate |
| R12 | hard negative 近似匹配不等于真正语义反事实 | 中 | 中 | 作为 bridge training regularizer，不单独据此宣称机制成立；held-out 生成决定 Go/No-Go | 方法限制 |

## 已发现并修复的确定性错误

1. `pilot_validation_generation.pjm` 仍启用 `expandable_segments`：改为 native allocator。
2. P1 checkpoint 已迁移，但评测仍引用旧目录：统一为
   `outputs/pilot/learned_sft_kl_native_full/best_bridge.pt`。
3. 手工 fallback 仍写旧 P1 目录：已同步。
4. smoke 与正式 bridge 初始化 seed 不完全一致：加入相同 `torch.manual_seed(seed)`。
5. monitor 只检查文件存在：升级为结构化 artifact contract。

## 启动后的强制检查顺序

1. `check_cuda_resource.py` 必须报告：一个设备、native allocator、总显存 >=40 GiB。
2. `resource_smoke.json` 必须为 pass、cluster=`nycc_415`、bridge update 非零、loss/grad finite。
3. 才允许读取训练 loss；在 smoke 前不得判断算法成败。
4. 每 epoch 检查 train/validation NLL、KL、margin、gradient norm 和 global step。
5. P1 必须通过完整 artifact validator，monitor 才能提交 P2。
6. 三个 pilot 完成后只生成 outer-validation packet；不依据 loss 自动扩大实验。

## 剩余建议

- 如果 P1 在完整 GPU 上仍因显存失败，优先加入冻结视觉特征和 teacher logits 缓存，
  不改变 bridge objective，也不减小通信通道。
- 如果 walltime 中断但至少存在一个完整 epoch checkpoint，从 checkpoint 恢复；禁止把
  walltime termination 当作方法失败。
- 最终论文结果至少使用三个独立评审和预注册的 Group-of-10 family；pilot Group-of-3
  只作筛选。

## 参考依据

- HOMER, ICLR 2026: https://openreview.net/pdf?id=SzaRhPom4o
- InterLat, ACL 2026: https://aclanthology.org/2026.acl-long.1248/
- StateBridge, 2026: https://arxiv.org/abs/2608.13317
- Humor in AI, NeurIPS 2024 Datasets and Benchmarks:
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
- QLoRA, NeurIPS 2023:
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html
- Hidden Technical Debt in Machine Learning Systems, NeurIPS 2015:
  https://proceedings.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
