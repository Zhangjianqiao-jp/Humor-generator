# Quality64 失败迁移分析与 Preference Learning 修订计划

## 结论摘要

当前证据支持继续研究 preference learning，但不支持立即遍历17,297 pairs，也不支持把 chosen-anchor 当成已确定的修复方案。

Quality64 Pilot 的方向略正，但没有通过预注册 gate；三评委多数为13胜10负1平，95%区间跨50%。所谓两个 Quality64-only `bad transition` 只来自单个generation seed下的采样结果，且 validation chosen log-probability 仅轻微下降。当前最合理的顺序是：先测量逐样本概率变化和seed稳定性，再决定问题属于 chosen-likelihood、偏好噪声、视觉条件利用不足，还是单纯采样方差。

## 一、目前真正观察到的事实

### 1. Pairwise validation

- 旧DPO validation loss：`0.690405`；
- Quality64：`0.689892`；
- 改善：`0.000513`，低于预注册门槛`0.001`；
- reward accuracy：`58.07% → 61.72%`；
- reward margin：`0.005705 → 0.006711`；
- chosen log-probability：`-41.9734 → -41.9945`，变化`-0.0211`；
- chosen per-token log-probability：`-2.90900 → -2.91007`，变化约`-0.00107 nat/token`。

最后一项变化非常小，不能称为 chosen-likelihood collapse。

### 2. 三评委单seed生成评估

Quality64相对旧DPO：

- overall：13胜、10负、1平；
- 平局调整得分：`56.25%`；
- decisive 95% Wilson CI：`36.81%–74.37%`；
- absolute：Quality64为6 good/15 weak/2 bad/1 unresolved，旧DPO为4 good/20 weak/0 bad。

评委一致性：

- overall Fleiss κ：`0.391`；
- absolute κ：`0.112–0.187`。

因此只有方向性证据，没有稳定收益证据。

### 3. 两个 Quality64-only bad transition

#### `nycc_545`

图片是会议室里一盆植物坐在椅子上。Quality64三条caption没有利用植物这一异常点；旧DPO至少产生了“manage the plant”等相关表达。三名评委一致选择旧DPO；Codex和Judge 2把Quality64判bad，Judge 3判weak。

#### `nycc_611`

图片是楼顶上的鸽子和小型非人动物。Quality64输出包含`good boy`、`mouse`、`corner store`等弱相关内容；旧DPO至少提到了pigeon。三名评委一致选择旧DPO；Codex和Judge 2把Quality64判bad，Judge 3判weak。

这两张图片均不在任何DPO train split中，而在384-pair validation中各出现16次，所以不是训练图像记忆导致的直接退化。

### 4. 不能把 bad transition 直接归因于 anchor 缺失

同一generation seed下，Quality64与旧DPO只有`1/24`张图片的三条候选完全相同；72条候选中仅3条文本重合。自回归采样在模型分布稍有变化后会迅速分叉，因此“某checkpoint这一次抽到三条差caption”不等价于“该checkpoint整体概率质量更差”。

此外，chosen per-token log-probability只下降约`0.00107`。在没有逐图片分布、置信区间和多seed生成结果前，chosen-anchor只是候选假设，不是诊断结论。

## 二、原实验计划中合理的部分

以下设计应保留：

1. SFT checkpoint、LoRA placement、rank、学习率和optimizer steps固定，隔离数据方案变化；
2. validation与test47分离，test47保持封存；
3. 先做1,264-pair Pilot而非直接跑全量；
4. 使用Group-of-3、绝对good/weak/bad和多评委，而不是只看相对胜率；
5. GPU前做静态门禁与真实smoke；
6. module search已降级为低成本分析，不再作为默认创新主线。

## 三、原计划中需要修正的部分

### 1. 17,297不是17,297个独立训练单元

完整Quality64只有271张图片，中位数为每图64 pairs；chosen caption最多重复17次。统计与泛化的主要独立单位是图片，而不是pair。直接完整遍历会让每张图的高度相关比较重复主导梯度，并增加对固定Hint/固定场景的过拟合风险。

后续应使用 image-balanced sampler，每图每epoch最多抽固定数量的clear/medium/hard pairs，并把“图片数、每图pairs数、caption复用”同时报告。

### 2. 当前Quality64实验不是纯数据质量因果实验

旧Pilot覆盖79张训练图，Quality64覆盖271张；它同时改变了pair质量、图片覆盖和pair构成。当前结论只能叫“Quality64数据方案整体效应”，不能写成“高质量pair导致提升”。

### 3. 7B objective并未真正筛选

采用DPO主要继承了3B历史结果，但模型大小、SFT起点、LoRA placement和生成分布都已改变。3B结果不能替代7B objective screening。

### 4. 缺少训练seed与生成seed

当前Quality64只有一个训练seed和一个生成seed。多评委只能减少judge variance，不能估计optimizer variance与sampling variance。

### 5. 自动judge未校准

现有Qwen judge给三组所有样本、所有维度全部5分，已经失效。下一次不得把它放进正式gate，除非先用已完成人类/多模型标签校准并证明不饱和。

### 6. 预注册gate的主指标需要改进

`loss改善≥0.001`可复现，但阈值缺少抽样不确定性依据；384 pairs来自24张图，不能把pair当独立样本。后续需要输出per-pair指标，并以image-cluster bootstrap或按图片聚合后再计算区间。

## 四、修订后的执行顺序

### Phase A：只诊断，不训练

1. 对同一384-pair validation导出SFT、旧DPO和Quality64的逐pair：chosen/rejected logp、per-token logp、reference-relative reward和margin；
2. 以24张图片为cluster，报告均值、median、bootstrap CI，以及chosen下降但margin上升的图片比例；
3. 对7B Generator做“固定Hint、正确图像 vs 错误图像”的条件敏感性诊断。它测试的是视觉条件利用，不重复Hint/no-Hint、wrong-Hint或Hint usefulness实验；
4. 对两个bad案例只作failure taxonomy，不把两个样本用于调超参数。

### Phase B：根据诊断选择objective，而不是默认anchor

#### 若chosen likelihood显著下降且margin上升

才允许比较：

- DPO；
- RPO/DPO+chosen-NLL；
- mDPO式reward anchor。

anchor系数不能直接沿用`0.1`。先在同一批次测DPO项和NLL项的gradient norm，令anchor梯度初始约为preference梯度的10%–30%，避免NLL主导训练。

#### 若chosen likelihood没有系统下降

不做anchor主实验。优先比较：

- DPO replicate：换训练seed，估计optimizer variance；
- IPO：针对确定性/主观噪声pair的过拟合风险；
- SimPO仅在确认长度仍影响reward时进入，因为Quality64已经做过长度匹配。

#### 若正确图像条件没有提高preference margin

优先做mDPO式conditional image preference，而不是chosen-anchor。训练只更新当前7B Generator；不恢复已停止的双模型联合训练。

### Phase C：重构训练采样

不直接使用全部64 pairs/图片。第一轮建议：

- 271张图片全部保留；
- 每图固定抽4–8 pairs；
- clear/medium/hard按预注册比例；
- chosen/rejected caption设置复用上限；
- 有score margin时作为confidence weight或分层分析，不把微小分差与巨大分差等权；
- 总optimizer steps与baseline严格一致。

### Phase D：低成本objective Pilot

第一轮只改变objective，固定MLP LoRA `r=3`、数据、steps和generation config。最小矩阵：

| Pilot | 启动条件 | 目的 |
|---|---|---|
| DPO seed-2 | 必做 | 估计当前结果是否依赖训练seed |
| IPO | 必做 | 检查主观/确定性pair下的过拟合 |
| RPO/anchored | 仅Phase A支持 | 防止chosen likelihood系统下降 |
| conditional mDPO | 仅图像条件诊断失败 | 防止language/Hint shortcut |

不同时改变module placement、rank或数据规模。

### Phase E：预注册生成评估

在训练前固定：

- 24张validation图片；
- 3个generation seeds；
- 每图每seed 3个candidates；
- 以图片为cluster；
- overall、best-pick、good/weak/bad、hallucination和generic rate；
- 至少三个独立盲评票，报告κ，不再使用饱和Qwen judge。

建议validation Go条件同时满足：

1. 三个generation seeds的平局调整得分均不低于50%；
2. image-clustered总体得分达到预注册的最小效应值，例如55%；
3. absolute good不低于baseline，bad不增加；
4. pairwise image-cluster指标与生成方向一致；
5. chosen likelihood、长度、hallucination没有预注册范围外退化。

只有一个配置通过validation后，才允许一次性运行封存test47。test结果不再反向用于调参。

## 五、对“chosen-anchor是否是下一步”的最终判断

目前答案是：**不是默认下一步，只是条件分支。**

原始DPO提供简洁的reference-relative偏好优化；IPO针对DPO在确定性偏好下的过拟合问题；RPO通过组合preference loss和SFT imitation loss缓解overoptimization；mDPO则同时提出image-conditional preference与reward anchor，直接针对多模态模型忽略视觉条件及chosen likelihood下降的问题。本项目必须先判断自己属于哪一种失败模式，再选方法。

相关原始来源：

- DPO: https://arxiv.org/abs/2305.18290
- IPO / ΨPO: https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html
- SimPO: https://arxiv.org/abs/2405.14734
- RPO: https://papers.nips.cc/paper_files/paper/2024/hash/fa69e968b7319fd42524febd41475fb3-Abstract-Conference.html
- mDPO: https://arxiv.org/abs/2406.11839

## 六、立即执行建议

当前不提交GPU训练。下一项应实现逐pair policy-logp导出与image-cluster统计，并在7B上完成固定Hint的正确/错误图像条件诊断。得到结果后再决定运行`DPO seed-2 + IPO`，以及是否增加RPO或conditional mDPO。这样每个新Pilot都有明确失败假设，而不是看到bad caption后机械增加正则项。
