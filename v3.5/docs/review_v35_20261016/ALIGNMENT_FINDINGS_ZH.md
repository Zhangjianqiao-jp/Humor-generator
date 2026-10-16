# 计划与 caption 对齐：固定训练样本定性审计

## 范围和证据

`alignment_sample.json` 保存 8 个训练 cluster、24 条原始 caption 及选定计划。抽样规则先于阅读内容固定：按 SHA256("v35_alignment_audit_20261016:" + cluster_id) 排序取前 8。用同目录 `build_alignment_sample.py` 重建；报告含训练数据/context 哈希。未读取验证/测试 caption 进行本次语义判断，未修改生产数据、标签、配置、训练损失。

以下是单一审阅者（Codex）的定性分析，不是人工共识、金标准标签或总体错误率。缺少显式关联不等于模型无法从常识补全；同样，常识能够补全也不证明路径提供了额外信息。

## 案例逐项核验

| Cluster | 原始信息与人类 caption 的关系 | 选定计划的限制 |
|---|---|---|
| nycc_619 | 办公桌插剑；hostile takeover 与暴力办公场景有直接关系；另两条涉及笔剑谚语、亚瑟王拔剑与销售职业 | 冲突有部分场景依据，但 desk→head、office→body 没有显式给出后两条的笑点关联；三条不能统一视为同一计划的实现 |
| nycc_673 | 长发救援场景；caption 分别用护发产品 extra body、滑下/救人、wash that man out of my hair | 冲突关注攀爬受阻，路径 person1→king→leaf、person2→demon→priest 没有显式连接这些语言机制 |
| nycc_562 | 鱼鸟相遇；caption 涉及共同吃虫、危险感、鸟与雕像 | 选定 conflict 的两项均描述通常的游泳/飞行，未明确表达相反脚本；fish→swimming→wear、bird→feathers→weight 没有给出对应笑点 |
| nycc_880 | 心理咨询室有巨大时钟；caption 包含声音烦扰、注意时间、把事情夸大 | 冲突为专业/家庭场景，路径 therapist→therapist's couch、patient→road 没有包含巨大时钟这一显著线索 |
| nycc_894 | 小丑逛画廊；caption 涉及鼻子跟随视线、假装理解漫画、儿童画令人恐惧 | 画廊礼仪/小丑对比存在主题相关性；clowns→fun→stop、gallery→art→door 未显式给出各条语言或反转机制 |
| nycc_868 | 冰淇淋头患者；caption 分别使用 meltdown、frosty/self serving、I scream/ice cream | 泛化的专业/滑稽冲突以及 Therapist→kid、Patient→lot 没有编码这些关键双关；description 提供的冰淇淋信息可能比计划更有用 |
| nycc_825 | 南瓜头商务会议；caption 涉及空脑袋/橙色、雕刻/face time、罐装/解雇 | professional/Halloween 对比与场景相关，但 conference_room→Chair→walk、professional_attire→Glove→man 不直接支持这些具体机制 |
| nycc_742 | 画廊烧烤；caption 涉及 Met/Mets、charcoal 双义、artist-in-residence | 艺术/烹饪冲突与前两条具有合理联系；chef→paintings→art、gallery→visitors→inmate 对更具体的词义关联支持有限 |

## 数学含义与下一步

当前训练实际优化 p(caption | description, selected_plan)，但人类 caption 在独立选择的计划之前就已存在。它们共享图片，并不保证该 caption 是该 plan 的实现。Caption NLL 仍是一个有定义的条件似然目标；问题在于不能据此宣称学到了计划执行能力。描述已经足以解释 caption 时，忽略无用计划甚至可能是最优行为。

Matched/shuffled 项更强地要求正确图片的计划提高该 caption 的分数。若匹配关系弱，这项可能学会图像主题相关性，或仅压低 donor 条件分数，不能解释为笑点机制恢复。KL 只要求模仿文本教师；文本教师也不一定有效使用计划。

GPU smoke 仍可继续，因为它检查工程路径。正式科学训练前需要明确以下证据条件：

1. 建立独立的计划-caption 兼容性标注定义，区分场景相关、关键关联支持、矛盾和不确定；不能以关键词重叠代替语义标签。
2. 先报告原始数据兼容性，不得悄悄筛选更容易的测试样本。若构建训练专用兼容样本，必须报告筛选比例、数据成本，并给文本/SFT 基线相同训练信息。
3. 对原始数据训练可作为“弱对齐条件学习”探索，但不能声称所有配对都是计划执行监督。以 NLL、NLL+KL、完整损失消融检验 margin 是否真正增加正确计划收益。
4. 所有机制实验保留 no-memory 对照；后续还需同主题且不同笑点的 donor，才能区分主题识别与笑点机制。当前描述近邻 donor 不足以完成这种控制。

本次没有根据这 8 个案例修改损失权重、设定总体阈值或宣称整条路线不可行。抽样只提供了真实反例与需要验证的假设。

## 论文参考

- [HOMER, ICLR 2026](https://arxiv.org/abs/2602.06423)：冲突脚本与联想规划框架的背景。上述案例判断来自本地数据，不是论文结论。
- [Humor in AI, NeurIPS 2024](https://arxiv.org/abs/2406.10522)：漫画 caption 与人类偏好基准；高排名 caption 本身不提供本项目独立选定计划的兼容标签。
