> 最新状态：job 7091538 已完成工程 smoke（见 PR_REVIEW_GUIDE_ZH.md）。下面的 QUE 状态为当时的历史记录，已被完成产物取代。之后新增 FP32 NLL 修复，尚需新的 GPU 验证。

# v3.5 selected-plan latent 推进记录（2026-10-16）

## 命名与范围

本轮始终位于 `Humor-generator/v3.5`。`cross_attention_caption_selected_plan_v2.yaml` 中的 v2 是内部配置修订号，不是项目版本。不切换到 v2 项目。

继续冻结 Qwen2.5-VL-7B，只训练 bridge；输入是最终选定 conflict 和两条 association path 的冻结编码。此路线不能声称绕过文本规划、恢复文本之前的思考，也尚未证明幽默优势。

## 本轮发现与修复

1. `real_trace_bridge_smoke.py` 旧的压力样本选择仍依赖 early planner trace 长度。现使用当前配置的真实 memory 来源，selected-plan 路线使用生产 backend chat template 统计全部 token；历史路线保留旧 trace 计数。公开 caption 路线分别选择最长文本教师完整输入和最大 memory 样本，保持两个独立训练 cluster；只有历史多模态路线仍按原始图像尺寸选样。这不是所有维度联合最坏情况的显存上界。
2. 原先只检查最后一个样本的损失。现每个样本检查有限值，优化器更新前检查梯度有限且非零，检查冻结 receiver 未积累梯度。报告两样本峰值最大值。
3. 新路线额外执行真实验证分支，必须产生 no-memory NLL 对照。该值来自随机初始化 bridge、训练样本，只证明验证计算路径能运行，不能作为机制有效性或泛化结论。
4. 旧 GPU 作业的 preflight 默认配置已变为 selected-plan，但实际 smoke 仍传旧配置。新增独立 `jobs/v35_selected_plan_smoke.pjm`，预检和运行显式使用同一配置，并核对配置哈希、memory/target policy 和源码哈希。旧作业不作为本路线入口。
5. CPU preflight 的输入检查结果现在写入该运行报告的同目录，避免独立输出目录仍引用全局共享检查报告。

## 验证和推进条件

- CPU preflight：`results/preflight/v35_selected_plan_20261016/cpu_preflight.json`。
- 完整回归测试：169 passed in 20.65s。新增计数、样本选择、非有限值失败测试。旧源字符串断言改由行为测试覆盖，不放宽科学门槛。
- 下一步仅两个真实训练样本、一次 bridge 更新的 GPU smoke。独立输出目录必须不存在；完成标志是有效 JSON。
- GPU 工程门通过后，再开展限定规模的训练/验证；使用验证集选择 checkpoint，检查 matched/shuffled/no-memory，最后才运行测试集生成与幽默评测。
- 必须补上同一最终计划的 text 对照和普通编码/投影对照，才能分辨信息来源与可训练接口的贡献。工程 smoke 不能代替这些实验。

## 论文参考与适用边界

- [HOMER, ICLR 2026](https://arxiv.org/abs/2602.06423)：幽默规划及评测协议依据。严格解析和失败计分仍是显式扩展，不标为官方原样复现。
- [BLIP-2, ICML 2023](https://proceedings.mlr.press/v202/li23q.html)：冻结骨干与可训练接口的技术先例，不证明本项目的具体 memory、损失或幽默效果成立。

## 已提交的 GPU 作业

- Job ID：`7091538`，入口 `jobs/v35_selected_plan_smoke.pjm`。
- 输出：`results/engineering_smoke/v35_selected_plan_20261016`，由作业开始时创建。
- 本轮 `pjstat 7091538` 确认 `QUE`；调度器显示预计 10/16 23:00，但非保证启动时间。
- 后续先查询此 job，不能因暂时无输出或没有目录就重复提交。尚无 GPU smoke 成功证据，未启动正式训练或外部付费评审。

## 补充数学核验

新增 `tests/test_split_backward_equivalence.py`，独立构造完整计算图（不调用生产 loss helpers）与生产 `_forward_metrics` 对比。6 组测试通过，覆盖 margin 权重 0/0.5/2、梯度累积缩放 1/0.25、batch=2、非单位蒸馏温度 1.7，并验证教师没有梯度。检查总损失和全部共享参数梯度，梯度容差 rtol=2e-5、atol=2e-7。此前 169 项完整回归与此次新增 6 项分开记录，尚未宣称 175 项完整重跑。

令 s+、s- 为目标 token 平均对数概率，z=m-s++s-，则 ranking 梯度为 sigmoid(z)(-∇s++∇s-)。在同一权重、确定性前向下，先无梯度算 sigmoid(z)，再分别反传两个分支，与完整 softplus 计算图的一阶梯度相同。这不等价于二阶导数，也不覆盖随机前向或两次前向之间更新权重的情况。当前 bridge 没有 dropout，receiver 保持 eval，两次分支反传之间没有 optimizer.step。

上述测试是数值实现核验，不证明 margin 目标能提高幽默性，也不覆盖真实 7B 的量化与 GPU 数值行为。

额外使用当前训练器的 hard_negative_cluster_map，在 271 个训练 cluster 与 44 个验证 cluster 各自内部重建配对：同一选定 conflict（strip+casefold 比较）为 0 对；完全相同 selected-plan SHA256 为 0 对。这只排除了当前数据的字面重复负例。负样本筛选仍依据旧候选冲突签名与描述 TF-IDF；不能声称已经控制最终计划的语义或长度混杂，更不能将“不同图片”直接解释为“对目标 caption 无效的计划”。

蒸馏背景参考：[Hinton et al., Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)。分步梯度等价性来自上述求导和本地测试，不由该论文替代证明。

## 输入模态更正与控制条件边界

检查 `public_text_caption_messages`、`public_latent_caption_messages` 和官方 `caption_messages` 确认：当前 public caption 路线只传文字 description，最终生成器不接收图像块。虽然使用 Qwen2.5-VL 权重、记录并校验图片路径，但这两点不等于 caption 阶段实际输入原图。此前聊天中“接收图片和任务提示”的描述不准确，以上述实现事实为准。

因此 smoke 的主提示压力轴已改为生产 processor 构建的文本教师完整 teacher-forcing 输入长度（包含实际 caption 目标）；另一个轴仍是实际 memory token 总数。报告新增 receiver_input_modality 与每样本文本完整长度；原始图片尺寸只作数据来源记录。此选样只覆盖指定训练子集中各 cluster 本轮选定 caption，仍不是所有 caption、所有前向中联合峰值的证明。

no_memory 使用与 latent 完全相同的空计划字段提示、关闭 bridge，适合测量 bridge 的增量作用。它仍保留原始 HOMER 的计划使用指令，不能冒充经过独立提示优化的最佳 direct-generation baseline；后续系统比较须单列普通直接生成基线。

文本/latent 输入信息公平性不等于 token 数或计算量相等。latent 有三次计划编码和额外 attention；编码的 chat-template token、通道划分及独立编码位置也不同。后续须记录这些差异，不能据此预先主张带宽或计算效率优势。

参考：[HOMER](https://arxiv.org/abs/2602.06423)。本节模态事实来自本地固定公开实现与消息构造核验，论文名称和模型的 VL 名称不能替代输入检查。

## 全量监督 token 检查

新增 `scripts/audit_selected_plan_targets.py`，使用固定本地 Qwen processor，经生产 `QwenBackend.encode` 和 `_prompt_and_full` 检查全部训练/验证数据；不加载基础模型权重，不使用测试集。执行：

```bash
.venv/bin/python scripts/audit_selected_plan_targets.py --output results/preflight/v35_selected_plan_20261016/target_audit.json
```

结果 pass：训练 813/813 行（271 clusters），验证 132/132 行（44 clusters）。所有行满足文本/latent 目标 token 完全相同、目标原文可无损解码、没有 EOS、完整序列保持 prompt 前缀、目标不超过配置 128-token 上限，且生成请求没有图像块。训练目标最大 44 tokens，验证最大 40；文本教师完整输入最大分别 295/283，latent 完整输入最大分别 206/185。完整输入长度不含通过 bridge 传入的 memory，不能用二者长度差直接证明总计算节省。

报告记录配置、context、模型 manifest、各 split 数据哈希。此检查不是计划-caption 语义对齐证据，也不是 GPU 行为验证。作业 7091538 已提交，此 CPU 审计是独立补充证据，没有重提作业。

基础模型背景：[Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)；上述具体 token 数来自本地 processor 实测。

## 定性对齐审计

固定抽取 8 个训练 cluster 的 24 条 caption，证据见 `docs/review_v35_20261016/alignment_sample.json`，逐案例解释及数学边界见 `docs/review_v35_20261016/ALIGNMENT_FINDINGS_ZH.md`。发现主题相关性不能替代笑点机制支持；例如冰淇淋患者的人类双关与所选 Therapist→kid、Patient→lot 路径未形成显式对应。单一审阅者的定性样本不是总体失配率，未修改数据或损失。

工程 smoke 继续等待现有 job。正式训练需明确弱对齐监督的解释边界并安排 NLL/KL/margin 消融；宣称计划执行机制之前需要独立的语义兼容性证据与同主题不同笑点的 donor 控制。

## Checkpoint 选择与损失消融入口

生产训练器此前只按 validation.total 选择 checkpoint；total 混合 caption NLL、KL 与 matched/shuffled 项。这样在不同损失权重间没有统一选择标准，也可能仅因 donor 得分下降而偏好某个 epoch。

新增 `training/checkpoint_selection.py` 与显式 `training.checkpoint_selection_metric`：支持 total（历史默认）及 caption_nll（caption 阶段）。选择分数缺失、非有限或非法类型时失败；恢复训练禁止改变选择指标。run manifest、checkpoint 和 complete.json 记录指标与最佳值。caption_nll 路线不会将 NLL 冒充 best_validation_total；历史 total 字段保留。旧 `formal_pipeline_monitor.py` 是历史 SFT pilot 专用入口，只读取旧 total 完成格式，不用于本轮新消融。

三份待运行配置位于 `configs/pilot/v35_selected_plan_ablations/`：

- nll.yaml：NLL=1，KL=0，margin=0。
- nll_kl.yaml：NLL=1，KL=0.5，margin=0。
- nll_kl_margin.yaml：NLL=1，KL=0.5，margin=0.5。

三者统一按图像 cluster 平均的验证 caption_nll 最小化选 checkpoint，固定原有 seed、数据划分、bridge、学习率、最大 epoch 与 early stopping 设置。程序化比较确认与基础配置仅实验名、两个损失权重和 checkpoint 指标不同。不同配置可能提早停止于不同 epoch，实际训练步数和成本必须报告；不能宣称训练算力严格匹配。

现有 selected-plan 基础配置及已排队 smoke 的配置未变；这三份消融未提交、未训练。机制 gate 与 checkpoint 选择分开，NLL 更低也不代表更幽默；它只是跨损失消融统一的验证代理指标。正式生成质量仍需独立盲评。

验证：选择规则+分步梯度测试 13 passed；随后完整测试集 182 passed in 21.17s。包含“total 变好但 NLL 变差时不应选择”的反例、恢复指标不一致拒绝、非有限分数拒绝。

模型选择的统计背景：[Cawley & Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)。论文讨论有限数据上的模型选择偏差，不声称 caption NLL 就是幽默质量指标；本项目仍需控制验证集反复调参及独立测试评估。

## Bridge 局部因果性与逐 token 等价性

新增 `tests/test_bridge_token_causality.py`，shared/per_channel 投影 × learned/fixed_equal 融合的 4 组测试全部通过。对固定 memory 比较完整 query 序列和逐 token 调用的输出；扰动未来 query 不改变前缀输出；对前缀输出反传时，未来 query 的梯度严格为零。测试含 batch=2 及屏蔽的 memory 位置。

证据范围仅为 bridge 模块自身；输入 hidden states 是否由真实冻结 Qwen 正确因果计算、KV cache、量化误差和生成 hook 生命周期仍须真实模型验证。此前 182 项完整回归与此次 4 项单独测试分开记录。

Attention/自回归背景：[Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)。局部实现性质来自生产模块的实际测试，不由论文替代证明。
