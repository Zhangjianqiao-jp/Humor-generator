# 7B Humor Planner 人工监督 SFT 改造计划

版本：v1.0
适用项目：New Yorker Cartoon Caption 7B planner + 3B captioner
核心目标：让 7B 从图片中生成**事实正确、能解释核心不协调、且能实际帮助 3B 写出更好 caption** 的紧凑计划。

## 1. 结论先行

当前 7B SFT 不应直接增加 epoch 或继续使用相同自动标签重复训练。下一轮应改为“人工校正标签 + 分阶段监督 + 因果对照评估”。

本阶段只训练 7B，冻结现有 3B，不进行 DPO。是否进入 DPO 或 reranker 阶段，取决于人工 oracle plan 是否能显著改善 3B，以及训练后的 7B plan 是否优于错配 plan。

推荐路线：

1. 保留现有 v1 数据、模型和评测，作为不可覆盖的 baseline。
2. 对 79 张训练图进行双人独立标注和仲裁；24 张验证图只用于验证，不参与梯度更新。
3. 将“可直接观察的图像事实”和“参考 gold caption 后推断的幽默结构”分开标注。
4. 用人工仲裁后的 v2 compact plan 重新做 7B QLoRA SFT。
5. 首先验证 7B 是否真的看对了图，再验证 plan 是否真的帮助了 3B。
6. 只有通过因果门禁后，才考虑给 3B 做 preference optimization 或训练 reranker。

## 2. 当前基线与已知问题

| 项目 | 当前状态 | 含义 |
|---|---:|---|
| 7B 训练图 | 79 张 | 样本极少，30 epoch 容易记忆标签表达方式 |
| 7B 验证图 | 24 张 | 可做开发验证，但不足以支撑很强的统计结论 |
| 人工 override | 20/103，约 19.4% | 自动教师标签存在明显语义错误，不只是格式问题 |
| planner JSON 合法率 | 24/24 | 模型已学会格式，但合法 JSON 不等于理解图片 |
| 明显视觉误读 | 至少 8/24 | 当前首要瓶颈是 grounding 与语义监督 |
| joint good-caption | 7/72，9.72% | 当前 7B + 3B 没有超过 direct 3B |
| direct good-caption | 9/72，12.50% | 目前没有证据支持直接进入联合 DPO |
| 每图至少一个好结果 | joint 7/24；direct 9/24 | planner 暂时没有带来可量化收益 |
| 当前图像预算 | 约 128 visual tokens | 对线条细、局部关系复杂的漫画可能不足 |
| 7B 输出与 3B 训练接口 | compact JSON vs 三行提示 | 3B 可能没有学会利用当前 JSON 字段 |

当前结果最可能由四个因素共同造成：

- H1：自动生成的 SFT 标签质量不够，模型学到了错误的视觉解释。
- H2：图像分辨率或视觉 token 太少，7B 无法稳定识别漫画细节。
- H3：即使图看对了，7B 对幽默的“不协调—消解”理解仍不足。
- H4：7B plan 是正确的，但 3B 没有学会消费该格式，因此忽略或误用 plan。

必须通过消融实验拆开验证这四个假设，不能用一次 joint 分数笼统归因于“7B 能力不够”。

## 3. v2 标签的设计原则

### 3.1 标签应回答什么

7B 不是直接写 caption，而是完成三个任务：

1. 看清：列出与笑点有关、可直接从图片确认的事实。
2. 想清：指出正常世界预期和图片中的违反之处。
3. 交接：给 3B 提供可写成 caption 的视角，但不替 3B 写完整笑话。

### 3.2 人工主记录格式

人工标注主记录包含元数据、正例 plan、负例和审核记录。负例只供 verifier/reranker 使用，不进入生成式 SFT 的 assistant target。

```json
{
  "image_id": "nycc_000548",
  "image": "relative/path/to/image.png",
  "annotation_version": "planner-human-v2",
  "image_sha256": "...",
  "annotator_id": "A01",
  "caption_accessed": true,
  "plan": {
    "literal_facts": [
      "A police officer is standing beside a stopped vehicle.",
      "The vehicle has the body and exhaust shape of a rocket."
    ],
    "normal_expectation": "A police officer normally stops an ordinary road vehicle for a routine traffic violation.",
    "incongruity": "An extreme rocket-like vehicle is being handled as if it were an ordinary car.",
    "resolution": "The humor comes from applying mundane traffic-stop procedure to a machine whose speed and form make that procedure absurd.",
    "mechanism": "context_collision",
    "speaker_options": ["police officer", "driver"],
    "caption_angles": [
      "The officer treats impossible speed as a routine infraction.",
      "Rocket technology is reduced to ordinary commuting trouble."
    ],
    "external_knowledge": []
  },
  "negatives": [
    {
      "type": "visual_near_miss",
      "text": "The vehicle is a shark-fin-themed car.",
      "error": "The prominent shape is interpreted as a shark fin instead of a rocket body."
    }
  ],
  "review": {
    "status": "adjudicated",
    "reviewer_id": "R01",
    "revision": 2,
    "notes": "..."
  }
}
```

### 3.3 生成式 SFT 的实际 target

SFT target 只保留下列紧凑正例字段：

```json
{
  "literal_facts": ["...", "..."],
  "normal_expectation": "...",
  "incongruity": "...",
  "resolution": "...",
  "mechanism": "context_collision",
  "speaker_options": ["..."],
  "caption_angles": ["...", "..."],
  "external_knowledge": []
}
```

约束：

- `literal_facts`：2–4 条，每条只描述一个可见事实，不写动机或笑点解释。
- `normal_expectation`：一句，写普通场景下会发生什么。
- `incongruity`：一句，明确“正常预期”和“反常事实”的冲突。
- `resolution`：一句，解释观众如何把冲突理解为一个笑点。
- `mechanism`：只能从固定枚举中选择。
- `speaker_options`：0–2 个，必须是图中合理的说话者。
- `caption_angles`：2–3 条短角度，不得写成完整 punchline。
- `external_knowledge`：仅记录确实需要的文化或事实知识；不需要时必须为空数组。
- 不输出 gold caption，不复制其中的特色短语，不虚构看不见的物体或身份。

`mechanism` 初始枚举：

```text
role_reversal
scale_violation
object_substitution
context_collision
literalized_idiom
anachronism
status_reversal
expectation_break
knowledge_reference
unclear
```

校准阶段允许修改一次枚举；正式标注开始后冻结，避免标签版本漂移。

## 4. 人工监督具体流程

### 4.1 人员角色

- 标注员 A：独立完成第一份标签。
- 标注员 B：不知道 A 的答案，独立完成第二份标签。
- 仲裁员 R：比较 A/B，查看争议项并产生唯一 gold plan。
- 数据管理员：运行 schema、泄漏、split、hash 和版本检查，不负责替代人工语义判断。

79 张训练图规模很小，建议全部双标，而不是只抽查一部分。验证集也双标，但禁止并入训练。

### 4.2 两阶段标注，避免 caption 泄漏

#### 阶段 V：Image-only visual grounding

标注员只看图片，不看任何 caption，填写：

- `literal_facts`
- 可能的 `speaker_options`
- 看不清或存在歧义的局部
- 置信度：high / medium / low

提交后锁定该阶段，不允许因为后来看到 caption 而偷偷改写可见事实。若 caption 暴露了确凿误读，只能通过有记录的 revision 修改，并由仲裁员批准。

阶段 V 检查清单：

- 人物数量、身份线索、位置关系是否写对？
- 关键物体是“看见的”还是“猜测的”？
- 是否遗漏了决定笑点的异常物体、尺寸、动作或环境？
- 是否把视觉相似物认成了错误类别？
- 单独读这些事实，另一名标注员能否在图中逐项指出证据？

#### 阶段 H：Humor reasoning with caption evidence

阶段 V 提交后，标注员查看 5–10 条经过清洗和去重的高分 caption。caption 随机排序，不显示具体名次，以降低对某一句文案的模仿。

标注员填写：

- `normal_expectation`
- `incongruity`
- `resolution`
- `mechanism`
- `caption_angles`
- `external_knowledge`
- 至少一个可能混淆模型的近邻错误解释

caption 的用途是帮助识别人类共同理解的笑点，不是给 planner 提供要复制的答案。提交前必须回答：

1. 这个解释是否由图片事实支持？
2. 它是否能概括多条高分 caption，而非复述其中一条？
3. 删除所有 gold caption 后，这个 plan 是否仍然自洽？
4. `caption_angles` 是否只是视角，而不是已经完成的笑话？

### 4.3 校准批次

正式标注前先选 10 张图片：

- 3 张视觉关系简单、机制明确的图片。
- 4 张含局部细节或身份歧义的图片。
- 3 张需要文化知识或存在多个合理解释的图片。

A/B 独立标注，随后逐字段比较。只有满足以下条件才进入正式批次：

- `literal_facts` 的事实精确率达到 95%。
- 核心 `incongruity` 双方一致或可通过一次讨论解决的比例达到 85%。
- `mechanism` Cohen's kappa 达到 0.60；若达不到，应修改定义和示例，而不是强行继续。
- 10 张全部通过 caption 泄漏检查。

校准产生的规则修改写入 `annotation_guidelines.md`。正式标注开始后，任何规则修改必须增加 annotation version，并决定是否回标此前样本。

### 4.4 仲裁规则

仲裁员逐字段处理 A/B 差异：

1. 先检查 image-only facts；事实不正确时，后续幽默推理不能被接受。
2. 若存在两个都合理的解释，保留主解释，并在 `alternative_resolution` 的人工记录中留档；SFT target 只选一个稳定主解释。
3. 若图片本身无法判断，使用 `unclear`，不得为了填满字段编造答案。
4. 若 gold caption 与图片事实冲突，以图片为准，并记录 caption noise。
5. 若标签含完整 punchline 或 gold caption 独特措辞，必须重写后才能通过。
6. 仲裁结果不得静默覆盖 A/B 原始标注；原始文件永久只读。

### 4.5 时间预算

建议按每图每人 8–12 分钟估算：

- 79 张训练图双标：约 21–32 人时。
- 24 张验证图双标：约 6–10 人时。
- 仲裁与最终审计：约 6–10 人时。
- 总人工预算：约 33–52 人时。

若人工资源不足，最低可行版本是：79 张全部完成 image-only facts；其中高歧义和高价值的至少 40 张双标；其余单标后由仲裁员全检。但该方案的标签可靠性低于全量双标。

## 5. 数据目录与版本管理

不得覆盖当前 `newyorker_compact_viewpoint_sft` 或已有 adapter。v2 使用独立目录：

```text
data/annotations/nycc_planner_v2/
  annotation_guidelines.md
  packets/
  raw/
    train_annotator_a.jsonl
    train_annotator_b.jsonl
    validation_annotator_a.jsonl
    validation_annotator_b.jsonl
  adjudicated/
    train.jsonl
    validation.jsonl
  negatives/
    train.jsonl
  audits/
    agreement_report.json
    leakage_report.json
    semantic_audit.json

data/processed/newyorker_planner_v2_sft/
  train.jsonl
  validation.jsonl
  manifest.json
```

每条记录至少保存：

- `image_id`
- 图片相对路径和 SHA-256
- annotation version
- 匿名 annotator/reviewer ID
- 是否查看过 caption
- caption 集合 hash
- revision 编号
- review status

train、validation、test 按 `image_id` 和图片 hash 双重检查。test 标签在模型预测冻结之前不得向训练流程开放。

## 6. 需要新增或修改的脚本

建议按下列职责拆分，不把所有逻辑塞进一个脚本：

| 脚本 | 作用 |
|---|---|
| `scripts/export_planner_v2_annotation_packets.py` | 输出 image-only 与 humor 阶段的标注包 |
| `scripts/validate_planner_v2_annotations.py` | 检查 schema、枚举、长度、空值、图片 hash |
| `scripts/merge_planner_v2_adjudication.py` | 根据明确的仲裁选择生成 gold，不自动替人裁决语义 |
| `scripts/audit_planner_v2_leakage.py` | 检查 target 与 gold caption 的 n-gram 重合及疑似改写 |
| `scripts/build_planner_v2_sft.py` | 只从 adjudicated/ 构建 SFT messages |
| `scripts/render_planner_v2_for_captioner.py` | 将 v2 plan 确定性渲染为 3B 熟悉的三行格式 |
| `scripts/build_planner_v2_counterfactuals.py` | 构造 matched/shuffled/near-miss verifier 数据 |
| `scripts/evaluate_planner_v2.py` | 计算 plan-level 与 downstream 指标 |

数据构建脚本必须 fail closed：只要存在非 `adjudicated` 样本、图片缺失、字段越界或 split 重叠，就以非零状态退出，不生成半成品数据。

## 7. SFT 数据构成

### 7.1 多任务监督

建议把同一张图构造成两类训练样本，以优先修复“看错图”：

- 40% visual-facts task：只生成 `literal_facts` 和 `speaker_options`。
- 60% full-plan task：生成完整 v2 compact plan。

两类任务使用明确不同的 system instruction 和 task tag。这样模型不能仅靠固定 JSON 模板完成所有样本，同时 visual facts 获得更密集的监督。

若 79 张图全部双任务展开，则每 epoch 约 158 个训练实例。增强只改变任务表述，不对图片做会破坏笑点的随机裁剪或翻转。

### 7.2 不应放入生成式 SFT 的数据

- 错配图片的 plan。
- 人工构造的错误视觉事实。
- 被拒绝的 teacher candidate。
- 完整 gold caption。
- test 图片的任何派生标签。

把负例当作 assistant target 会教模型生成错误内容。负例应保存在单独数据集，供后续 verifier、reranker 或 preference 阶段使用。

### 7.3 3B 输入兼容层

本轮不训练 3B。对 7B v2 输出使用确定性 renderer，转成 3B 当前熟悉的格式：

```text
ANCHOR: <由 literal_facts 压缩得到>
CONTRAST: <normal_expectation> BUT <incongruity>
ANGLE: <选择一条 caption_angles>
```

评测时同时比较：

- 原始 compact JSON 直接送入 3B。
- renderer 后的三行格式送入 3B。

如果三行格式显著更好，问题主要是接口失配，不应错误归因于 planner 没有价值。

## 8. 训练配置与资源策略

### 8.1 首轮推荐配置

保持 QLoRA，不做全量微调：

| 参数 | 建议值 |
|---|---|
| Base | 当前 Qwen2.5-VL-7B-Instruct |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| 学习率 | 1e-5 起步；最多与 2e-5 做小规模对照 |
| epoch | 5–10，禁止默认回到 30 epoch |
| effective batch | 8 左右 |
| weight decay | 0.01 |
| warmup ratio | 0.05–0.10 |
| checkpoint | 每 20–25 optimizer steps |
| selection | 人工语义指标优先，validation loss 仅作辅助 |

训练前保存 longest-sample 的 token 数、图像尺寸、峰值显存和 target 长度。先做 CPU 数据 dry-run，再做 2-step GPU smoke，最后才提交正式任务。

### 8.2 图像 token 消融

当前约 128 visual tokens 可能不足，但不能仅凭猜测申请更大 GPU。先做只推理的分辨率对照：

- 同一 base/SFT checkpoint，在 128 和 256 visual tokens 下生成 24 张验证图的 plan。
- 对 `literal_facts` 做盲评，统计视觉错误数。
- 若 256 相对 128 将视觉错误降低至少 20%，再安排 256-token SFT。
- 若没有明显改善，优先修标签和推理结构，不增加显存申请。

梯度累积只能降低 batch 相关显存，不能解决单样本图片和模型本身的峰值显存。若 12GB MIG 无法稳定运行 256 tokens，应申请能容纳该配置的最小分区，而不是默认申请完整 A100。

### 8.3 实验矩阵

至少完成以下实验，除指定变量外保持相同：

| ID | 标签 | 输出结构 | visual tokens | 用途 |
|---|---|---|---:|---|
| B0 | 现有自动标签 | v1 JSON | 128 | 已有 baseline |
| S1 | 人工纠错 | v1 JSON | 128 | 单独测标签质量作用 |
| S2 | 人工 v2 分阶段标签 | v2 JSON | 128 | 测结构化监督作用 |
| S3 | 人工 v2 分阶段标签 | v2 JSON | 256 | 仅在分辨率预实验通过后运行 |

如果预算有限，优先顺序为 S1 → S2 → S3。S1 很重要，因为它把“标签质量”和“schema 改动”分开了。

## 9. 评测设计：量化模型发生了什么变化

### 9.1 Planner 本身的指标

| 指标 | 计算方式 | 本轮门槛 |
|---|---|---:|
| schema valid | 可解析且字段/类型正确 | ≥99% |
| literal fact precision | 人工逐条判断事实是否可见且正确 | ≥95% |
| key-object recall | gold 关键物体/关系被覆盖 | ≥90% |
| incongruity correctness | 是否准确指出主冲突 | ≥80% |
| resolution coherence | 是否由事实推出且能解释笑点 | ≥75% |
| speaker validity | 建议说话者是否在图中合理 | ≥95% |
| caption leakage | 与 gold caption 的特色 5/6-gram 重合 | 0 |
| mechanism macro-F1 | 与人工仲裁类别比较 | ≥0.65 |

每个指标都同时报告分子/分母和 bootstrap 95% CI，不能只报告平均分。`validation loss` 下降但语义指标不升，视为未通过。

### 9.2 关键因果对照

对相同图片和相同 3B checkpoint，比较：

1. Direct：3B 只看图片。
2. Base-plan：未 SFT 的 7B plan + 3B。
3. SFT-plan：新 7B plan + 3B。
4. Human-oracle：人工 gold plan + 3B。
5. Shuffled-plan：另一张图片、但机制相近的 plan + 3B。
6. Near-miss-plan：同一图片的一个关键事实或 resolution 被替换 + 3B。

解释规则：

- Human-oracle 不优于 Direct：当前 3B 不会利用 plan，或 plan 接口本身无价值；先修 3B 输入训练，不继续强化 7B。
- Human-oracle 优于 Direct，但 SFT-plan 不优于 Direct：7B 标签、视觉分辨率或容量仍是主要瓶颈。
- SFT-plan 优于 Shuffled，但不优于 Direct：7B 有传递信息，但带来的收益还不足以抵消提示噪声。
- SFT-plan 与 Shuffled 几乎相同：3B 忽略 plan，或者评测候选数太少。
- SFT-plan 接近 Human-oracle 且优于 Direct：联合路线值得进入后续 preference/reranker 阶段。

反事实一致性不是抽象概念：它测量“正确匹配的 plan”相对于“看似合理但属于别图的 plan”能带来多少分数。若两者没有差异，就不能声称 7B 的推理真正帮助了 3B。

### 9.3 下游 caption 指标

对每个条件每图生成相同数量候选，固定 seed 列表、temperature、top-p、max tokens 和解码后处理。

主要指标：

- candidate good rate：所有候选中被多数评委判为好笑且贴图的比例。
- image hit rate：每张图是否至少产生一个好结果。
- pairwise preference：同图匿名比较 joint 与 direct，允许 tie。
- relevance failure：caption 是否与图片或 plan 明显不一致。
- diversity：同图候选去重率和语义重复率。

最少 3 名独立评委盲评，不显示系统名称。报告多数票、评委间一致性和按图片 bootstrap 的 95% CI。

当前 24 张图只适合 pilot。进入 DPO 前，建议锁定 50–100 张从未训练、未调参的图片作为最终测试，以避免 1–2 张图片左右结论。

### 9.4 进入下一阶段的门禁

必须同时满足：

- Planner 语义门禁通过：事实精确率 ≥95%，incongruity correctness ≥80%。
- SFT-plan 相对 Shuffled-plan 的 image hit rate 至少高 10 个百分点。
- Human-oracle 明显优于 Direct，证明 plan 通道有潜在上限。
- SFT-plan 相对 Direct 的 candidate good rate 至少提高 5 个百分点，且 image hit rate至少提高 10 个百分点；正式结论应要求 CI 支持正向差异。
- 未发现 test 泄漏或 gold caption 文案复制。

任何一项未通过，都先定位瓶颈，不进入联合 DPO。

## 10. SFT 后的负例、reranker 与 DPO

### 10.1 推荐先做 verifier/reranker

从人工记录构造三类负例：

- Cross-image hard negative：换成另一张图但幽默机制相同的 plan。
- Visual near-miss：只替换一个关键物体、人物身份或空间关系。
- Reasoning near-miss：事实正确，但 incongruity 或 resolution 与事实不匹配。

建议正负比例 1:2。verifier 输入图片 + plan，输出匹配分数和错误类型。它可以用于：

- 从 7B 采样的多个 plan 中选最可靠的一个。
- 过滤明显视觉错读后再交给 3B。
- 量化 planner 是否真正把图片信息传给 captioner。

在目前数据规模下，reranker 往往比立即对两个生成模型做联合 DPO 更可控，也更容易解释改进来自哪里。

### 10.2 何时做 DPO

只有在以下前提成立后再做：

- 人工 oracle plan 能稳定帮助 3B。
- 新 7B 已能产生较高事实精确率的 plan。
- 已收集同图、同解码条件下的明确 preference pair。
- chosen/rejected 的差异来自幽默质量或 plan 利用，而不是长度、格式或图片相关性错误。

优先次序：

1. 冻结 7B，给 3B 做 caption preference optimization。
2. 训练 image-plan verifier/reranker。
3. 若证据表明 7B 的多个候选 plan 质量差异大，再对 7B 做 plan preference optimization。
4. 暂不做端到端联合 DPO；信用分配不清楚且 103 张图片远远不够。

## 11. 执行时间表

| 阶段 | 工作 | 产物 | 预计时间 |
|---|---|---|---|
| D0 | 冻结 v1 数据、模型、配置和评测 | baseline manifest | 0.5 天 |
| D1 | 10 张双人校准与规范修订 | guideline v2 | 1 天 |
| D2–D4 | 79 train + 24 validation 双标 | raw annotations | 2–4 天 |
| D5 | 仲裁、agreement、泄漏与 split 审计 | adjudicated labels + reports | 1 天 |
| D6 | 构建 S1/S2 数据；CPU dry-run | processed SFT datasets | 0.5–1 天 |
| D7 | 最长样本 2-step GPU smoke | smoke report | 0.5 天 |
| D8 | 顺序运行 S1/S2，单任务不抢占多 GPU | adapters + logs | 1 天左右 |
| D9 | planner 盲评与六条件 downstream pilot | evaluation report | 1–2 天 |
| D10 | 根据门禁决定 S3、reranker 或停止 | decision record | 0.5 天 |

## 12. 每次正式训练前的提交清单

- [ ] 数据 manifest 包含条数、split、图片 hash、annotation version。
- [ ] 所有训练标签状态均为 `adjudicated`。
- [ ] schema 校验 100% 通过。
- [ ] train/validation/test 无 image ID 和图片 hash 重叠。
- [ ] gold-caption 5/6-gram 泄漏为 0。
- [ ] 对 10 个随机样本人工逐项复核。
- [ ] 最长样本完成 CPU collator dry-run。
- [ ] 最长样本完成 2 optimizer step GPU smoke。
- [ ] smoke 使用与正式任务相同的模型、量化、LoRA target、image token 和 collator。
- [ ] 峰值显存低于分区上限，且保留日志证据。
- [ ] 输出目录是新目录，不覆盖 v1 adapter。
- [ ] 训练命令、依赖版本、git commit、seed 写入 run manifest。
- [ ] checkpoint 保存和恢复流程实际测试过一次。

## 13. 停止条件

出现任一情况应停止扩大训练：

- Human-oracle plan 仍不优于 direct 3B。
- 256 visual tokens 对视觉事实精确率没有可复现改善。
- 人工 v2 标签训练后，validation loss 下降但事实错误不降。
- Correct plan 与 shuffled plan 对 3B 输出没有显著差别。
- 仅通过增加 epoch 获得训练集改善，验证语义指标不升。
- joint 提升来自更长输出、拒答减少或单个图片，而不是多数图片的一致改善。

对应行动：

- Oracle 无收益：先改造或微调 3B 的 plan 接口。
- 7B 看不清：提高视觉 token、改视觉 encoder 输入或更换更强 VLM。
- 7B 看对但推理差：增加人类幽默结构标签，或使用分阶段推理和 verifier。
- 7B plan 正确但 caption 不好笑：瓶颈在 3B 生成和 preference 数据，不再继续堆 planner SFT。

## 14. 与现有工程的衔接

应复用并扩展当前资产，而不是删除重来：

- 当前 prompt：`prompts/7b_image_to_compact_viewpoint.txt`
- 当前 v1 SFT：`data/processed/newyorker_compact_viewpoint_sft/`
- 当前人工 overrides：`data/overrides/compact_viewpoint_*_overrides.json`
- 当前 finalizer：`scripts/finalize_compact_viewpoint_labels.py`
- 当前审计：`scripts/audit_compact_viewpoint_labels.py`
- 当前 SFT 审计：`scripts/audit_newyorker_compact_sft.py`
- 当前联合评测：`docs/JOINT_VS_DIRECT_COMPACT_VIEWPOINT_EVAL.md`
- 当前最佳 adapter：`outputs/newyorker_compact_viewpoint_7b_qlora/best_val_loss`

v1 override 中已经发现的错误应迁移为 v2 校准案例，但不能直接视为最终人工 gold。所有迁移样本仍需按 image-only → humor reasoning → 仲裁流程重新检查。

## 15. 论文依据

1. Hessel et al., “Do Androids Laugh at Electric Sheep? Humor ‘Understanding’ Benchmarks from The New Yorker Caption Contest,” ACL 2023. 该工作表明现有多模态模型在 New Yorker humor understanding 上仍很困难，即使提供人工视觉描述，机器解释也明显落后于人类解释。<https://aclanthology.org/2023.acl-long.41/>
2. Tanaka et al., “Incongruity-Resolution Chain-of-Thought for Generative Commonsense Reasoning,” Findings of NAACL 2024. IRCoT 将视觉细节、不协调和消解分阶段，并使用负采样抑制泛化、空洞的 resolution，是本计划分阶段标签和 hard negative 的直接依据。<https://aclanthology.org/2024.findings-naacl.152/>
3. Zhang et al., “The New Yorker Caption Contest Dataset,” NeurIPS 2024 Datasets and Benchmarks. 该数据集包含大规模人类投票，并展示了使用高置信 preference pair 对 7B 模型做 DPO 的方法，同时也讨论了创意任务中 preference optimization 的局限。<https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html>
4. Zhou et al., “Improving Multimodal Humor Understanding and Generation through Visual Annotation, Humor Reasoning and Preference Alignment,” Findings of EMNLP 2025. 该工作支持“更好的视觉标注 + humor reasoning + 小规模针对性 preference alignment”的组合，而不是仅扩大普通 SFT。<https://aclanthology.org/2025.findings-emnlp.884/>
5. Shang et al., “HOMER: A Multi-Agent Framework for Open-Ended Multimodal Humor Generation,” ICLR 2026. 该工作把冲突脚本提取、层次化想象和 caption generation 分为不同角色，支持 planner/captioner 的模块化路线。<https://openreview.net/pdf?id=SzaRhPom4o>
6. Zhang et al., “HUMORCHAIN: Theory-Guided Multi-Stage Reasoning for Interpretable Multimodal Humor Generation,” CVPR 2026. 该工作采用视觉描述、策略判断、理论引导生成和 discriminator 的多阶段结构，支持在生成器之外增加可验证的中间表示和判别器。<https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_HUMORCHAIN_Theory-Guided_Multi-Stage_Reasoning_for_Interpretable_Multimodal_Humor_Generation_CVPR_2026_paper.pdf>
7. Vural et al., “Incongruity-Resolution Scaffolding for Multimodal Humor Generation,” arXiv 2026. 该预印本提出 incongruity modeling → resolution modeling → preference alignment 的顺序，可作为后续实验参考，但证据等级低于已正式发表论文。<https://arxiv.org/abs/2604.15210>

## 16. 最终决策标准

本轮 SFT 的成功不是“loss 更低”或“JSON 更整齐”，而是同时发生三种可观察变化：

1. 7B 对关键人物、物体和关系的事实错误显著减少。
2. 正确匹配的 plan 明显优于 shuffled/near-miss plan，证明中间推理与图片存在因果对应。
3. 在固定 3B 和固定解码条件下，联合系统稳定超过 direct 3B，并接近 human-oracle plan 的收益。

只有达到这三点，才值得把数据和算力投入后续 DPO；否则应根据 oracle、resolution 和 interface 三组消融结果，明确选择修 7B、修 3B 接口或停止联合路线。
