# v3.5 实验失败与计划变更准则

本文件规定所有 v3.5 实验失败的记录方式。机器可读记录位于
`docs/EXPERIMENT_FAILURES.jsonl`，使用
`scripts/record_experiment_failure.py` 追加，禁止静默重试或覆盖旧输出。

## 强制字段

每次失败必须记录：唯一 ID、时间、失败层级、状态、可观察现象、原始证据、
根因、采取的修复、对实验计划的影响、相关 artifact。失败层级只能是：

- `environment`：调度器、CUDA、NVML、依赖或文件系统；
- `data`：缺失、泄漏、schema、hash 或 provenance；
- `engineering`：shape、dtype、OOM、hidden/token 对齐或实现错误；
- `method`：代码正确运行，但预注册的机制/统计 gate 未通过；
- `evaluation`：盲化、评审、统计单位或指标错误。

只有 `method` 失败才允许被写作“方法没有通过”。环境和工程失败不得作为模型
优劣证据。每次修复后必须创建新输出目录，不允许覆盖失败实验。

## 当前结论

## V35-ENG-008（2026-09-05）：A5 outer donor helper 的 split 契约错误（已在执行前修复）

进一步的静态审计发现 outer evaluator 复用了旧失败重评脚本的 donor helper；该 helper
硬编码 donor 必须来自 `train`，而 A5 outer 预注册的是 validation donor。即使补上第二
份 trace index，原 helper 仍会在 forward 前拒绝合法的 validation donor。

已改为使用 `training/formal_bridge.py::length_matched_channel_donors`，该实现保留
per-channel length-priority、different-conflict、same-source/TF-IDF tie-break，同时
允许 outer 的 validation donor。该问题没有产生 GPU forward 或科学 artifact；旧 helper
只继续用于历史 v1/v2 重评，不得用于 A5 outer。

## V35-ENG-007（2026-09-05）：A5 outer donor trace index 缺失（已在执行前修复）

静态审计发现 outer evaluator 原命令只传入 sealed held-out test trace index，但
length-matched counterfactual donor 必须读取 validation/train trace。若直接运行，目标
trace 能加载而 donor lookup 会缺少记录，因而不能产生合法的 outer gap。该问题在任何
outer forward 之前发现，未产生科学输出。

修复为 `run_outer_semantic_confirmation_a5.py --donor-trace-index`：目标 index 与
666-record train/validation index 只读合并，并对重复 cluster 做记录一致性检查；summary
同时保存 donor index 路径和 SHA-256。旧命令禁止使用，A5 outer 仍须等待 121 条 sealed
trace 完整校验和修复后 checkpoint。

## V35-ENG-006（2026-09-05）：A5 训练 donor 未按 channel 长度匹配

A5 初次全量作业 `6712005` 已在第 2 epoch 中途主动取消。第 1 epoch 的 validation
窗口出现 global mean gap `0.205538`、单 cluster 最大 gap `1.293392`，而 A5 的
outer 协议要求按被替换 channel 的 token length 选择 donor。检查发现训练入口仍调用
旧的 `hard_negative_cluster_map`，同一个 donor 映射同时用于三个 channel，未控制
attention 候选数量/sequence-log-probability 的长度混淆。因此这批输出不能作为方法收益、
semantic gate 或 caption 生成依据；原始 checkpoint、metrics 和 stdout 保留在
`outputs/pilot/cross_attention_semantic_phase_a5_full/` 作为审计证据。

该事件分类为工程协议错误，不是 OOM、CUDA、数据缺损或 latent 方法失败。修复为
`training/formal_bridge.py::length_matched_channel_donors`：length 是首要匹配键，
随后才使用 different-conflict、same-source 和 TF-IDF 相似度；训练允许 donor pool
与 target split 重叠但排除自身 cluster，validation 仍只从 train donor pool 取样。新运行
将把完整 donor map、各 channel 的 exact-length fraction/mean absolute length delta
和 hash 写入 `channel_donors.json` / `run_manifest.json`。修复后的作业必须使用新输出
目录 `outputs/pilot/cross_attention_semantic_phase_a5_lengthmatched/`，重新通过
静态测试、real-trace smoke、完整训练和 sealed outer gate 后才可进入 caption 阶段。

## V35-ENG-005（2026-09-04）：旧 A4 正式脚本在执行前取消

旧 c-batch 作业 `6711074` 在获得资源并开始模型 forward 前被取消。审查发现该作业
调用的是通用 bridge 训练流程：没有 A4 专用的 `validate_phase_a4_smoke.py`/动态
`semantic_gate.json` 约束，且输出目录与 A4 既定目录不一致。继续运行会使“训练完成”
与 A4 gate 的 provenance 混淆，因此在无 scientific output 前主动停止。该事件是
工程/可追溯性问题，不是 CUDA、OOM、数据缺损或 latent 方法结果。

修复为独立的 `jobs/cross_attention_phase_a4.pjm`：使用 `c-batch + gpu=1`，固定
native allocator/CUDA0，先执行 locked preflight、真实 trace resource smoke 和
A4 validator，再训练 bridge；训练后只读取本次 `$output/semantic_gate.json`，并在
gate 未通过时 fail-closed。A4 output 改用新目录
`outputs/pilot/cross_attention_semantic_phase_a4_cbatch_retry1`，不得覆盖旧实验。
修复后重新运行本地回归测试和 formal preflight，再提交新作业。

## V35-ENG-003（2026-09-04）：outer preflight 因工作树未提交而停止

作业 `6708118` 已获得 b-batch 独占节点，但在实际 outer forward 前由 provenance guard
报告 `dirty_tracked_worktree` 并退出。compile、75 个测试、环境检查、2846/2846 数据行、
949/949 图片和 666/666 Planner traces 均通过；因此这不是 GPU、OOM、数据或 latent
方法结果。没有创建 `outputs/outer_semantic_confirmation/a3_outer40`。

原因是 tracked docs/job/memory 文件存在未提交修改。修复方式是审查并提交有意保留的
修改，或从 immutable clean worktree 重跑；不得用 `--allow-dirty` 绕过来源追踪门禁。
此事件使 outer A3 baseline 暂停，A4 channel-isolated pilot 也必须等 clean commit 后
再提交。

## V35-ENG-004（2026-09-04）：A4 smoke 被监控命令提前终止

A4 real-trace smoke `6711036` 在 `genkai0002` 上启动，并在 preflight 仍运行时被发出
`pjdel`，PJM 记录 signal 15、使用 62 秒；输出为空，A4 smoke 目录没有 report 或
validator。触发原因是监控时 `pjstat` 暂时没有当前行，被误判为作业已结束。这是操作/监控
工程错误，不是 CUDA、OOM、数据缺损或 latent 方法结果。

保留 `.out/.stats` 作为证据；不得把该作业视为通过或失败的科学实验。修复规则是：
`pjstat` 暂时无行时先用 `pjstat -H` 和 `.stats` 确认终态，不得直接 `pjdel`；替代 smoke
必须使用新 output 目录，在同一 clean commit 上重新执行 preflight、真实 forward/backward、
target-only one-hot mask 和 donor reconstruction validator。

1. `cross_attention_semantic_v1` 是方法级 No-Go，不是运行失败。其 reconstruction
   NLL 明显下降，但 validation matched-minus-shuffled gap 只有 `0.004843`，低于
   `0.02` gate，且 `fraction_gap_gt_margin=0`。
2. 代码审计发现 v1 把 conflict/local/global 拼接后做单一 positional softmax，长
   association channel 可能产生长度竞争；同时实现的 InfoNCE 没有进入正式 trainer。
3. v2 改为通道内独立 attention、通道间 receiver-dependent gate，并强制在梯度
   累积窗口上调用真实 InfoNCE。GPU smoke 也必须用两个真实 trace 执行同一
   contrastive backward，不能只由 CPU 单测证明。caption bridge 在新的语义 gate
   通过前仍被禁止。
4. plain-text/latent 不做一次性全 latent 假设。语义 gate 通过后按逐步消融比较：
   Text-HOMER、conflict-text+association-latent、conflict-latent+association-text、
   all-latent；只对胜出的混合方式继续拆分 local/global。
5. v2 GPU engineering smoke 作业 `6688553` 与 stationary-teacher post-fix 作业
   `6688566` 均已通过：两个真实 trace 均完成 hierarchical attention、
   InfoNCE/variance backward 和 optimizer update；policy trainable params 为 0。
   此项只关闭工程风险，不改变 v1 的方法级 No-Go。
6. Hierarchical Phase A v2 作业 `6688689` 已完成但仍为方法级 No-Go：causal
   matched/shuffled gap 为 `0.002664 < 0.02`，conflict router mass 下降到
   `0.0289`。validation retrieval 的原始 `0.190476` 受同 cluster 重复 caption
   false-negative 污染，已单独登记为评测实现错误；不能用它支持或反对方法。
7. 对第 6 项的声明范围已审计收紧：log-probability 与两遍反传公式正确，但 negative
   是跨 image cluster plan，不是同图单通道 counterfactual；`0.02` 也是工程阈值而非
   统计校准阈值。因此结论只能是“all-latent v2 未通过当前操作性 gate”，不能写成
   “latent communication 已被证伪”。
8. v2 的 reconstruction 按全部 target token 平均，较长 local/global channel 获得更多
   监督；unregularized router 随后将 conflict mass 压到 `0.0289`。v2 InfoNCE teacher
   还是固定的 receiver-embedding 随机投影，只约束 trace identity，并不充分证明
   frozen Receiver 能解释这些语义。两项均记录为设计不足。
9. 计划已改为低成本 Phase A3：三通道等权 reconstruction、三路 contextual InfoNCE、
   单通道 matched/shuffled counterfactual、固定等权 gate、cluster bootstrap CI。只有
   三个 channel 都越过校准后的 control 才能进入 caption bridge；否则转入
   conflict-text + association-latent 混合方案。
10. 样本量解释已更正：`64/24` 是开发期机制 pilot，不是方法级 confirmatory experiment。
    24 clusters 对接近 50% 的比例估计，正态近似 95% 半宽约 0.20，无法可靠检测小效果。
    因此 Phase A3 的 24-cluster point pass 只能触发 sealed outer semantic validation；
    point fail 也只能标记 `pilot_inconclusive`，不能据此排除一个可能有效的方法。只有
    工程不变量失败（报告为 `hard_no_go`）或明确、预注册的强负向证据才是 pilot hard
    stop。v1/v2 也必须用同一 cluster-level evaluator 重算后才能公平比较。
11. 旧 v1/v2 重评 evaluator 又发现一个必须单独记录的 confound：v1 对拼接后的全部
    memory 使用单一 softmax，原 donor 的 channel 长度差会改变 softmax 分母；此前
    固定 24-cluster donor 的长度差最高约为 conflict 41、local 85、global 163 tokens。
    因此原重评不能直接解释为语义 gap。已将 `scripts/re_evaluate_failed_semantic_bridges.py`
    改为逐 channel 的 length-priority donor：从非 test train split 选择不同 conflict
    signature 的 donor，优先最小长度差，再优先同源和 description 相似度；并排除该
    checkpoint 已拟合的 train clusters。逐 cluster 长度差、gap correlation 和 donor
    pool 均写入 provenance。详细协议见 `docs/SEMANTIC_REEVALUATION_PROTOCOL_ZH.md`。
12. 该修正后的重评必须使用与旧 checkpoint 严格匹配的 bridge architecture，并检查
    local model/adapter/prompt manifest；输出 summary/manifest 记录 evaluator hash、
    seeds、checkpoint/config/data/trace/prompt/model/adapter hashes。直到新的 GPU
    summary 完成前，不得把旧 `no_go` 重新解释为方法级无效。
13. A3 smoke 作业 `6689653` 在真实 trace `electronic_sheep:325:0` 处提前退出：配置
    `max_target_tokens=384`，而该 global semantic target 为 492 tokens。validator 随后
    因预期 JSON 不存在而退出。分类为 `engineering`，不是 CUDA/OOM、数据缺损或方法
    失败；没有产生 A3 语义结论，也没有提交正式训练。配置已提高到 768，保持完整
    HOMER chain。
14. 修正后的 A3 replacement smoke `6706516` 在单张完整 H100 上以 exit code 0 完成：
    两个压力样本均完成真实 forward/backward，policy trainable parameters=0，bridge
    update 非零，三路 contextual InfoNCE 和单通道 counterfactual 路径均实际执行，
    validator 返回 `status=pass`。因此第 13 项工程配置错误已关闭；它仍不提供 latent
    语义收益或 caption 质量证据，正式 A3 仍须单独运行 `64 train / 24 validation`。
15. 外层语义确认 smoke 的资源请求曾连续被 PJM 拒绝：`node=1,gpu=1` 触发
    `GENKAI1006`，`gpu=1,exec-policy=simplex` 触发 `GENKAI0029`。这是调度器资源
    类型冲突，不是模型、CUDA 或数据错误。GENKAI 的节点分配作业本身就是
    simplex/node-exclusive，并会按 b-batch 节点配置自动分配 GPU；已将脚本改为
    `b-batch + node=1`，进程内固定 `CUDA_VISIBLE_DEVICES=0`，移除显式 `gpu` 和
    `exec-policy`。旧探针 `6708044` 已取消，修正后的 smoke `6708113` 已以
    exit code 0 完成，并显示 `simplex=true,gpu=4`；其 validator 通过。以后不得把
    `node=1` 与 `gpu=1` 或
    `exec-policy=simplex` 叠加。

## V35-SCHED-003（2026-09-04）：空闲 inter 节点不接受 batch 脚本

为缩短短 smoke 的等待，曾检查到 `b-inter` 有一个空闲完整节点；在取消唯一的
queued `b-batch` 作业 `6711814` 后，尝试提交同一 smoke 的 inter 版本。PJM 返回
`PJM 0070 pjsub No execute permission: pjsub batch`，没有启动 forward、没有生成
输出，也没有占用科学计算资源。根因是 `b-inter` 只允许其交互作业模式，默认的
`pjsub` batch 脚本没有执行权限；“节点空闲”不能替代“作业模式兼容”。

该事件分类为 scheduler/engineering，不是模型、数据或方法结果。无效脚本已删除，
不再与 batch/MIG 资源组 race；A4 outer smoke 应恢复到已验证的 `b-batch + node=1`
后台模式。以后资源选择同时检查：GPU/节点是否空闲、账户限额、作业模式权限和历史
兼容性；`*-inter` 只用于显式的短交互调试，不用于正式或自动化后台实验。

## 权威依据

1. Shang et al., HOMER, ICLR 2026: https://openreview.net/pdf?id=SzaRhPom4o
2. He et al., MoCo, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html
3. Yang et al., Hierarchical Attention Networks, NAACL 2016: https://aclanthology.org/N16-1174/
4. Libovicky and Helcl, Multi-Source Attention, ACL 2017: https://aclanthology.org/P17-2031/
5. van den Oord et al., Contrastive Predictive Coding, 2018: https://arxiv.org/abs/1807.03748
6. Card et al., Statistical Power in NLP, EMNLP 2020: https://aclanthology.org/2020.emnlp-main.745/
7. Dror et al., Significance Testing in NLP, ACL 2018: https://aclanthology.org/P18-1128/
8. Graham et al., Statistical Power and Translationese in Machine Translation Evaluation,
   EMNLP 2020: https://aclanthology.org/2020.emnlp-main.6/
9. Howcroft and Rieser, What happens if you treat ordinal ratings as interval data?,
   EMNLP 2021: https://aclanthology.org/2021.emnlp-main.703/
10. Koehn, Statistical Significance Tests for Machine Translation Evaluation, EMNLP 2004:
    https://aclanthology.org/W04-3250/
