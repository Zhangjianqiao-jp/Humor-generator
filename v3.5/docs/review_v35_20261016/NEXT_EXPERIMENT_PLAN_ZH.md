# v3.5 latent 后续执行计划（2026-10-17）
## 2026-10-17 优先级修订

依据用户要求，停止扩展精度支线。当前以 [LATENT_EFFECTIVENESS_PRIORITY_ZH.md](LATENT_EFFECTIVENESS_PRIORITY_ZH.md) 为优先方案：先计划质量与免训练接口探索，不将原数值 gate 当作所有效果探索的硬阻塞；保留原失败证据，暂不扩展九次 bridge 训练。以下为此前方案，冲突处以此修订为准。


状态：准备中的研究方案，不是已完成实验，不是外部预注册。新正式训练/付费评测尚未启动。始终使用 v3.5；配置名的 v2 是修订号。

## 1. 当前证据边界

- FP32 GPU 工程复验 7097141 已通过；两样本、一次 bridge 更新，不证明质量。
- 实际生成/KV cache job 7097320 已提交，写本文时 QUE，结果待核验。
- 初始 bridge 相对 no-memory 的两例 NLL 改善一正一负。
- 已有 8 个训练 cluster 定性审计揭示计划与 caption 可能仅主题相关；不是总体失配率。
- 正式候选的生成、checkpoint 恢复、语义兼容标签、强基线和盲评仍未齐备。

## 2. 研究问题与数学量

令 x 为固定 description 与两条件共享的上下文，p 为选定 conflict/两条 path，z=E_frozen(p)，y 为 caption。主路线只训练 bridge φ，冻结 Qwen 参数 θ。

文本对照为 qθ(y|x,p)，latent 为 qθ,φ(y|x,z)。这首先是**给定文本计划的表示与条件注入实验**，不是无文本思考，也不自动等于多智能体创新。

固定 caption 的 NLL 目标可定义，但独立选出的 p 不一定支持 y；必须称作弱对齐监督。定义每个 caption 的平均 token NLL，再先在 contest 内平均，最后在 contest 间平均，避免把三条 caption 或多次生成当独立图片。

两项机制量必须分开报告：

- A = NLL(no-memory) − NLL(matched)：正确计划是否比不提供计划更有用。
- B = NLL(shuffled) − NLL(matched)：匹配计划是否比 donor 更有用。

B>0 而 A<=0 可能只是 donor 更有害，不能称为有效利用计划。KL 模仿文本教师也不等于学到幽默；margin 提高 B 不保证提高 A 或质量。图像/description、计划长度、主题近邻 donor 都可能影响 B。

## 3. 执行顺序与进入条件

| 阶段 | 任务与产物 | 进入/停止条件 |
|---|---|---|
| G0 | 核验 7097320：同前缀 cache/replay、hook、18 次输出 | baseline 与 latent 分开判读；任一 cache 检查失败先定位。不得看结果后放宽阈值并称原门槛通过 |
| G1 | checkpoint 保存→重新加载→相同前缀 logits 复验；对训练后权重补生成检查 | 现在随机初始化 smoke 不能替代 checkpoint 验证；恢复不一致停止训练扩展 |
| A0 | 计划-caption 兼容性独立标注与标签协议 | 原 8 个 cluster 用作规则开发；独立审核仍无标签时，不把 AI 单方结论写作人工共识 |
| P0 | 训练集固定 16 cluster 的单 epoch 成本/梯度 pilot；同一份 cluster 列表供三损失使用 | G0 通过；先测运行时间、显存、保存恢复。pilot 不读测试集，不用于宣布质量 |
| P1 | 三损失完整 train/validation 单种子探索 | G1 通过，明确弱对齐任务，保存每 epoch 的 A/B/NLL/生成诊断；不是正式 test 比较 |
| P2 | 三损失各三个预定训练种子，补齐强基线 | 成本预算有记录；不挑最好种子。计划共 9 个 bridge run，不自动提交 |
| E0 | 冻结方法、数据/提示/评审版本、主要比较与统计脚本；锁定 checkpoint | 历史已有评测结果，需核查是否涉及当前测试 ID 及其对决策的影响；暴露审计完成前不能称其为未触碰确认集，已用于决策则需新确认集合 |
| E1 | 原 HOMER 兼容轨 + 明示严格解析扩展轨；图片级区间和人工盲评 | 比较和协议先锁定，再执行。新评估集合未落实时只报告原集合的探索性复现 |

G0 若 baseline 同样失败，优先检查量化/位置编码/缓存数值误差；若仅 latent 失败，检查 hook 时序与 memory 处理。不能凭单一 max-error 值断言架构错误。格式失败须逐例区分截断、marker、拒答；规则若修改，所有条件统一重跑，记录版本。

P0 固定集合建议按 SHA256("v35_cost_pilot_20261017:" + cluster_id) 排序取 16 个训练 cluster。尚未生成 pilot 配置或提交任务，不改已有完整训练配置。

## 4. 对照矩阵：分清每种对照回答什么

| ID | 条件 | 回答的问题/限制 |
|---|---|---|
| T0 | 冻结同骨干 + 同一 selected plan 的原文本 | 当前实现的通信路线对照；与训练 bridge 比较同时改变了可训练能力 |
| N0 | latent 提示 + 关闭 bridge | 机制对照；不冒充调优后的直接生成 baseline |
| L1/L2/L3 | selected-plan bridge，分别 NLL / NLL+KL / NLL+KL+margin | 损失消融；配置已存在，训练未启动 |
| S0 | selected-plan 简单接口，例如 masked mean pooling + 投影为固定 soft prefix | 检查复杂 cross-attention 是否必要；实现待完成，token 压缩程度、参数量差异必须披露 |
| T1 | 同文本计划 + 可训练文本适配模块（优先 LoRA） | 检查优势是否仅来自额外训练；同训练数据、目标、选 checkpoint 规则；尽量匹配参数/步数，并报告不匹配部分 |
| D0 | description-only 的直接 SFT/LoRA | 实用强基线，不是同信息通信对照；数据与训练预算要公平 |

T1/D0/S0 尚无本计划已验证的训练实现，不能把表格当作已准备好的可执行命令。冻结骨干不代表公平：还需分别报告 trainable 参数、训练 token、步数、总 GPU 时间和调参次数。

## 5. 三损失实验的固定项

现有配置：`configs/pilot/v35_selected_plan_ablations/{nll,nll_kl,nll_kl_margin}.yaml`。

- L1: NLL；L2: NLL+0.5 KL；L3: NLL+0.5 KL+0.5 softplus(0.2−s_matched+s_shuffled)。
- checkpoint 统一按 validation caption_nll 最小选；不得按 test/最佳评审结果回选。
- 同种子三组共享初始 bridge state_dict 哈希、训练顺序、donor 列表、完整数据版本和 epoch 上限；不能只因 seed 相同就假定初始化相同，需实际核验。
- 拟定训练种子：20260906、20260907、20260908；保持与生成 seed 分开。
- 当前 5 epoch 上限、early stop patience=2、min_delta=0.002。训练时长可能因 early stop 不同而不同，需报告实际步数；必要时增加等步数敏感性对照。
- 现有 mechanism gate 是工程/探索阈值，不能充当统计显著性或幽默通过标准。
- 不根据这两个 smoke 样本重调门槛。三组结果均保存；中止运行也保留失败原因。

## 6. 对齐标注协议

对每条 caption 分别记录（不是互斥的一维好坏标签）：

1. 场景一致：yes/no/uncertain。
2. 选定 conflict 是否支持该笑点：yes/no/uncertain。
3. path0、path1 是否提供笑点关键关联：分别 yes/no/uncertain。
4. caption 是否与计划矛盾：yes/no/uncertain。
5. 从 description 本身能否解释笑点：yes/no/uncertain。
6. 必填理由：caption 中的机制/双关及计划中具体支持或缺失的位置。

两名独立审阅者先独立标注再裁决；保留原标签、分歧与裁决，不只保存共识。AI 预标注必须注明来源，不能充当两个人类标注者。报告各维度一致率、分布和机会校正一致性，低频标签不只报一个 kappa。

下一批建议在训练集中按独立 hash 种子 `v35_alignment_holdout_20261017:` 取 32 个 cluster，排除已用于定义规则的 8 个。这是约 96 条 caption 的标注预算，不是先验功效保证。若无人工资源，可做显式 AI 探索审计，但“人工验证的对齐率”仍不可报告。

在见到新批次标签前固定规则。不得根据测试标签筛选训练/测试样本；若后续采用 teacher-generated plan-consistent caption，那是新的蒸馏数据路线，需单独版本、人工质量抽检、数据成本和同数据基线，不能悄悄替换当前人类 caption 监督。

## 7. 评价与结论规则

- 以 HOMER 论文与固定代码版本的协议为参照，先逐项核对模型 ID、提示、参考 caption 组、trial/candidate 数及聚合。已有 362 vs 论文 365 的数据差异继续披露。
- 原协议兼容与 strict parser 扩展分别报告；不以扩展结果冒充严格复现。
- 拟定主要比较为最终选定 latent 与 T1 的质量差；T0 为论文路线对照。若 T1 未完成，不能声称已排除额外训练的解释。
- 主要质量量拟用原协议 Pass@5；主要参考组、judge 精确版本和资源成本尚未锁定，所以 E0 当前不能通过。所有参考组/预定 K 均报告，不事后挑最弱组。
- 对系统差值做配对 contest-level bootstrap（拟定 10,000 次），所有同图候选/trial 一起重采样；禁止把 25 次生成当 25 张独立图片。训练 seed 单独报告，只有三个种子时不夸大算法总体置信度。
- 多组主要检验事先定义 family 并控制多重比较（例如 Holm）；探索项明确标注。区间跨零不是等效；若研究非劣需事先确定有实际意义的容忍差距，此处尚未设置。
- 盲评随机左右顺序、隐藏系统名，分别评幽默、场景贴合、计划执行；不让长 Explanation 替 caption 影响主要幽默判断。
- 成本包含上游 planner、冻结计划编码、bridge、解码及 judge；缓存复用和冷启动分开记录，不预先宣称 latent 更快。

## 8. 如何收缩或停止路线

- A 无稳定正向变化、B 却上升：不宣称有效使用计划，检查 donor 惩罚和主题捷径。
- margin 只改善 total/B、caption NLL/质量更差：保留负结果，以更简单损失为候选，不因复杂而保留。
- 复杂 bridge 与简单接口无可靠差异：不主张架构必要性。
- 潜在方案不优于公平后训练基线：如实报告，重新定位为机制/负结果，不能靠选择参考组得到“成功”。
- 若原测试集合已用于反复决策：不能用常规 CI 抹除适应性选择偏差；新的确认集合未准备好时限制结论范围。

## 9. 文献及用途

- [HOMER, ICLR 2026](https://arxiv.org/abs/2602.06423)：上游规划和复现协议参照。
- [Humor in AI](https://arxiv.org/abs/2406.10522)：caption 偏好数据和质量评估背景，提醒后训练基线与人类评估的重要性。
- [Cawley & Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)：模型选择过拟合及评估选择偏差。

以上文献不保证本项目有效；阈值、种子、标注规模为本计划的设计选择。
