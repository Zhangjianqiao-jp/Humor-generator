# Codex 重新打开窗口用上下文说明

生成日期：2026-06-24  
项目目录：`/home/zhang.jianqiao/projects/v2.5`

这份文件用于在新的 Codex 窗口中恢复项目上下文。请严格按本文限定继续工作，不要凭记忆补全不存在的实验结果。

---

## 0. 上下文与实验记录保存规范

后续上下文按四层保存，避免只依赖聊天记忆：

1. `instruction.md`
   - 只保存长期稳定约束、模型/数据路径、硬性实验规则、新窗口恢复入口。
   - 不记录未经验证的新实验结论。

2. `docs/obsidian/`
   - 保存人类可读的阶段复盘、失败点、决策理由、论文脉络和下一步计划。
   - 命名格式：`YYYY-MM-DD 主题.md`。

3. `docs/experiments/`
   - 每个关键实验一张 experiment card。
   - 必须记录 commit、dataset split、sample/seed、模型、adapter、prompt renderer、生成参数、输出路径、指标、风险和决策。

4. `outputs/manifest/runs.jsonl`
   - 机器可读 run manifest，每行一个 JSON object。
   - 只记录小索引，不提交大型生成结果、checkpoint 或 JSONL 产物本体。

新 Codex 窗口继续本项目时，优先读取：

```text
instruction.md
docs/obsidian/2026-07-27 HIC Humor Captioning 项目复盘与下一步.md
docs/obsidian/2026-07-29 HIC Humor Captioning 本轮上下文与失败复盘.md
docs/experiments/README.md
outputs/manifest/runs.jsonl
README.md
```

---

## 1. 项目研究主题

本项目研究目标是提升预训练视觉语言模型 `Qwen2.5-VL-3B-Instruct` 在图像幽默 caption 生成任务上的表现。

核心问题：

> 给定一张图片，是否可以通过额外输入“图片描述、幽默点、结构化视觉事实、注意力增强图、文化/常识知识”等指导信息，让 Base Qwen2.5-VL-3B 生成更自然、更贴图、更好笑的短 caption？

当前研究重点不是单纯追求训练 loss，而是要通过严格对照实验判断指导信息是否真的提升幽默 caption 质量。

---

## 2. 绝对硬性限制

除非用户明确改变要求，否则必须遵守：

1. 工作目录：
   - `/home/zhang.jianqiao/projects/v2.5`

2. Python 环境必须使用：
   - `/home/zhang.jianqiao/miniconda3/envs/humor/bin/python`

3. Generator / Base 模型必须是：
   - `/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct`

4. Teacher / analyst / judge 可使用：
   - `/home/zhang.jianqiao/models/Qwen2.5-VL-7B-Instruct`

5. 不使用用户旧的 LoRA/SFT adapter。

6. 不使用用户之前的 CLIP reranker，因为用户明确说它效果不好。

7. 不要把旧 adapter、pilot adapter、Base 模型混为一谈。

8. 原始 caption 生成 prompt 在相关实验中必须保持完全一致，并放在最终生成指令位置：

   ```text
   Generate one short, natural, image-specific humorous caption for this image. Do not explain.
   ```

9. 标准采样参数：

   ```text
   temperature = 0.8
   top_p = 0.9
   max_new_tokens = 48
   repetition_penalty = 1.05
   ```

10. 对照实验中，Plain 与 Guided 必须使用相同采样参数、相同随机种子、相同图片集合。

11. 如果实验要求是 “每种方法每张图生成 8 个候选”，必须严格生成 8 个候选，不能用 4 个候选替代。

12. 不要在没有用户确认的情况下启动新的长时间训练。小规模只读检查、脚本审查、结果统计可以直接做；完整训练或大规模生成前要说明会做什么。

---

## 3. 原始严格 A/B 实验设定

用户最初要求完成一个 1000 张不重复图片的严格 A/B 对照实验，用来判断：

> “图片描述 + 幽默点”是否能提升预训练 Base Qwen2.5-VL-3B 的幽默 caption 生成能力。

测试集：

```text
data/processed/sft_test.jsonl
```

用户说已经建立了 1000 张不重复图片子集，相关路径前缀为：

```text
outputs/evaluations/base3b_guidance_comparison_
```

原始两种方法：

### A. Plain

输入：

```text
image + 原始 prompt
```

模型：

```text
Base Qwen2.5-VL-3B-Instruct
```

输出：

```text
8 candidates
```

### B. Guided

第一步，用 7B teacher 提取：

- 一句保守、准确的图片描述
- 最多 1 个高置信幽默点

第二步，输入：

```text
image + description + humor cue + 相同原始 prompt
```

模型：

```text
Base Qwen2.5-VL-3B-Instruct
```

输出：

```text
8 candidates
```

原始幽默点限制：

- 只能基于明确可见的异常、尺寸差异、动作反差、构图关系或角色错位。
- 不得猜测身份、职业、关系、情绪、意图和隐藏故事。
- 不得引入 visual facts 中没有的对象。
- 没有可靠幽默点时必须输出空。
- 不要强行制造幽默点。
- 不能直接写最终 caption。

重要说明：

> 到目前为止，不要假设这个“1000 张、每图 8 候选、严格 Plain vs Guided”的最终实验已经完整完成。后续如果要报告结论，必须先检查对应输出文件与统计脚本。

---

## 4. 已经实现过的脚本和文件

项目里已经有以下重要文件。继续工作前优先阅读它们，而不是重新凭空写一套。

### 数据集构造

```text
src/training/guided_humor_sft_dataset.py
```

该文件用于构造 guided SFT 数据。当前 prompt 形式大致为：

```text
Image description: ...
Humor cue: ...

Generate one short, natural, image-specific humorous caption for this image. Do not explain.
```

### LoRA 训练脚本

```text
scripts/train_guided_humor_lora.py
```

关键事实：

- 这是新 LoRA 训练脚本。
- 它从 `model.model_name` 加载 base 模型。
- 它显式设计为不 resume、不加载旧 adapter。
- 文件里有注释说明：总是从 base 创建新的 PEFT adapter。

### Pilot / full 配置

```text
configs/guided_humor_lora_pilot.yaml
configs/guided_humor_lora_full.yaml
```

关键事实：

- `model_name` 指向 `/home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct`
- 后来已把 `device_map` 改为 `cuda:0`
- 之前 `device_map: auto` 会导致 meta-device backward 相关错误
- LoRA 大致设置：rank 16, alpha 32, dropout 0.05, target q/k/v/o

### Pipeline / gate

```text
scripts/run_guided_sft_pipeline.py
```

关键事实：

- `load_vlm(model_path, adapter=None)` 默认只加载 base。
- 只有当 `adapter is not None` 时才会调用 `PeftModel.from_pretrained(base, adapter)`。
- base generation 使用 `adapter=None`。
- pilot LoRA generation 使用的是本项目新训练出的：

  ```text
  outputs/guided_sft_pipeline/pilot_lora/final_lora
  ```

### Teacher pipeline

```text
scripts/run_guided_sft_teacher_pipeline.py
scripts/run_guided_sft_teacher_pipeline_v2.py
scripts/run_guided_sft_teacher_pipeline_v3.py
scripts/run_guided_sft_teacher_pipeline_v4.py
scripts/run_guided_sft_teacher_pipeline_v5.py
scripts/run_guided_sft_teacher_pipeline_v6.py
```

其中 v6 是更严格的 cue filtering / production 版本。

### LoRA loader

```text
src/models/qwen_vl_lora_loader.py
```

关键事实：

- 使用 `Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)` 加载 base。
- 使用 `LoraConfig` + `get_peft_model(model, lora_config)` 创建新 adapter。
- 没有加载旧 adapter。

---

## 5. 已经做过的 pilot LoRA 实验

用户后来要求先写一个“用高质量数据，把 description 和幽默点和图片一起作为输入交给 generator，对模型微调训练”的方案；如果测试效果好再自动训练。

已经做过一个小规模 pilot：

### Pilot 数据

```text
outputs/guided_sft_pipeline/data/pilot_train.jsonl
outputs/guided_sft_pipeline/data/pilot_val.jsonl
outputs/guided_sft_pipeline/data/pilot_gate.jsonl
outputs/guided_sft_pipeline/data/pilot_train_guidance.jsonl
outputs/guided_sft_pipeline/data/pilot_val_guidance.jsonl
outputs/guided_sft_pipeline/data/pilot_gate_guidance.jsonl
```

规模：

- train: 512
- val: 64
- gate: 128

### Pilot adapter

```text
outputs/guided_sft_pipeline/pilot_lora/final_lora
```

重要限定：

> 这个 adapter 是本项目新训练出的 pilot LoRA，不是用户之前的旧 LoRA/SFT adapter。

### Pilot 训练情况

- 训练 1 epoch
- 64 optimizer steps
- step 32 eval_loss 约 1.8335，ppl 约 6.2558
- step 64 eval_loss 约 1.8084，ppl 约 6.1009

### Pilot gate 结果

评估文件：

```text
outputs/guided_sft_pipeline/pilot_evaluation/base_candidates.jsonl
outputs/guided_sft_pipeline/pilot_evaluation/pilot_lora_candidates.jsonl
outputs/guided_sft_pipeline/pilot_evaluation/blind_judgments_7b.jsonl
outputs/guided_sft_pipeline/pilot_evaluation/gate_summary.json
```

结果：

```json
{
  "base_wins": 36,
  "lora_wins": 37,
  "ties": 55,
  "failures": 0,
  "decisive": 73,
  "lora_decisive_rate": 0.5068493150684932,
  "lora_wilson_95pct": [
    0.39472240623307775,
    0.6182914018771178
  ],
  "exact_binomial_p": 1.0,
  "mean_lora_minus_base_usable": -0.0078125,
  "gate_rules": {
    "minimum_decisive": 80,
    "wilson_lower_above": 0.5,
    "p_below": 0.05,
    "usable_difference_at_least": 0,
    "max_failure_rate": 0.05
  },
  "passed": false
}
```

结论：

> Pilot LoRA 没有通过 gate。不能说 LoRA 显著提升。full data generation / full training 已停止。

必须向用户准确解释：

- 这次 pilot 的确训练了一个新的小 LoRA。
- 它不是旧 LoRA。
- 它没有明显变好。
- 因为 gate failed，所以没有继续 full training。

---

## 6. 已确认：测试过程没有使用用户旧 LoRA

用户明确问过：“你测试的过程里，是否使用了我之前 lora 的模型？”

已检查并确认：

> 没有使用用户之前的旧 LoRA/SFT adapter。

证据点：

1. `scripts/train_guided_humor_lora.py` 是从 base model 创建新 adapter，不加载旧 adapter。

2. `configs/guided_humor_lora_pilot.yaml` 的 `model_name` 是：

   ```text
   /home/zhang.jianqiao/models/Qwen2.5-VL-3B-Instruct
   ```

3. `scripts/run_guided_sft_pipeline.py` 中 base 生成使用 `adapter=None`。

4. LoRA 对照只使用本次新生成的：

   ```text
   outputs/guided_sft_pipeline/pilot_lora/final_lora
   ```

5. `outputs/guided_sft_pipeline/teacher_pipeline_manifest.json` 中有：

   ```json
   "old_adapter_loaded_for_training": false
   ```

6. `outputs/guided_sft_pipeline/pilot_lora/final_lora/adapter_config.json` 的 `base_model_name_or_path` 指向 base 3B 模型。

---

## 7. 已生成的可视化页面

为了让用户查看 Base vs Pilot LoRA captions，已经创建：

```text
scripts/visualize_pilot_base_vs_lora.py
outputs/guided_sft_pipeline/pilot_evaluation/base_vs_lora_gallery.html
```

HTML 页面内容：

- 128 张 gate 图片
- 图片缩略图
- description
- humor cue
- full prompt
- Base candidates
- Pilot LoRA candidates
- blind judge reason
- usable counts
- confidence
- Base win / LoRA win / tie / cue / no cue 过滤

如果用户想继续查看，可让他打开：

```text
/home/zhang.jianqiao/projects/v2.5/outputs/guided_sft_pipeline/pilot_evaluation/base_vs_lora_gallery.html
```

---

## 8. 当前 guidance 的问题

对 pilot guidance 做过统计：

- `pilot_train_guidance`: 512 条，其中非空 humor cue 51 条，约 9.96%
- `pilot_val_guidance`: 64 条，其中非空 humor cue 6 条，约 9.38%
- `pilot_gate_guidance`: 128 条，其中非空 humor cue 11 条，约 8.59%

主要问题：

1. 约 90% 样本没有有效 humor cue。
2. 很多 cue 被严格规则过滤掉。
3. 当前 cue 多是中性的视觉关系，不是强幽默机制。
4. 实际训练更像是 “image + plain description → caption”，幽默指导信号太弱。

示例：

```text
Image description: A green alien sits next to a human person in a vehicle.
Humor cue: The green alien has a large, exaggerated head compared to its body.
```

空 cue 示例：

```text
Image description: Two people are standing outdoors near a body of water, with one person raising their arm.
Humor cue:

Generate one short, natural, image-specific humorous caption for this image. Do not explain.
```

当前判断：

> Pilot 失败不能证明“指导信息无效”，更可能说明当前 guidance 太保守、太稀疏、表达方式不够适合 3B generator。

---

## 9. 用户提出的新 idea

用户提出几个方向：

### Idea 1：结构化幽默引导

例如：

```text
{{"man's face","angry"}{"coffee cup","big"}}
```

或误会类：

```text
{"xxx","mistake","bbb"}
```

目标：让 generator 更容易理解幽默点。

建议：

- 不要用太随意的 `{{...}}` 格式作为最终方案。
- 更推荐内部保存浅层 JSON / scene graph。
- 喂给 3B 时，可以把 JSON 渲染成固定标签的自然语言。
- 对 3B 来说，固定标签自然语言通常比深层 JSON 更稳。

### Idea 2：修改后的图片 / 注意力增强图

用户希望有算法能模仿人类注意到图片重点的方式，对图片做处理。

建议：

- 不要替换原图。
- 使用：

  ```text
  original image + attention-enhanced image + optional crop
  ```

- 注意力图可以用 soft spotlight、背景降饱和、轻微模糊、边缘增强等方式突出重点。
- 尽量避免箭头、文字标签，因为这会引入新的视觉对象，可能干扰 caption。
- 可结合：
  - generic human saliency
  - 7B teacher 预测的 humor anchor region
  - object detector / SAM / grounding model
  - crop-based local view

### Idea 3：文化知识库 / RAG

原先严格规则禁止猜测身份、职业、隐藏故事和流行文化，因此 RAG 会冲突。

但用户后来指出：

> “我觉得可以不用限定这个。”

所以当前研究方向可以放宽：

- 可以使用合理身份、角色、情绪、意图、文化背景、时代冲突、流行文化、常识知识。
- 但是必须明确区分：
  - visible facts
  - inferred context
  - retrieved knowledge
  - humor mechanism

绝不能把推断内容伪装成图片中明确可见事实。

### Idea 4：自由组合 ablation

组合项包括：

- 原图
- 注意力增强图
- 普通 description
- 结构化 visual facts
- 结构化 humor cue
- RAG / cultural knowledge

建议做 factorial ablation，而不是直接全部混在一起。

---

## 10. 文献调研后的方法建议

已经做过一轮文献/方向调研，结论如下：

1. 没有直接证据证明某一种结构化格式一定最适合 `Qwen2.5-VL-3B` 做视觉幽默 caption。

2. 相关研究支持：
   - scene graph / object-attribute-relation 对视觉推理有帮助。
   - incongruity-resolution 结构对幽默理解与生成有帮助。
   - JSON、edge list、固定标签自然语言各有优势。
   - 小模型通常更吃 prompt 表达形式，过深 JSON 不一定好。

3. 推荐内部结构用浅 JSON，外部喂给 generator 用固定标签自然语言。

推荐内部 schema：

```json
{
  "visible_facts": {
    "entities": [
      {
        "id": "e1",
        "label": "person",
        "attributes": ["holding a cup"]
      },
      {
        "id": "e2",
        "label": "coffee cup",
        "attributes": ["large relative to the person's hand"]
      }
    ],
    "relations": [
      {
        "subject": "e1",
        "predicate": "holding",
        "object": "e2"
      }
    ]
  },
  "inferred_context": {
    "items": [
      {
        "claim": "The cup may be comically oversized for a normal drink.",
        "confidence": "medium",
        "basis": "scale contrast visible in the image"
      }
    ]
  },
  "humor_mechanism": {
    "type": "scale_contrast",
    "anchors": ["e1", "e2"],
    "expected_frame": "A handheld cup normally fits comfortably in one hand.",
    "observed_violation": "The cup looks disproportionately large.",
    "resolution": "Treat an ordinary coffee serving as absurdly oversized.",
    "caption_strategy": "understatement_or_absurdity"
  },
  "retrieved_knowledge": []
}
```

推荐喂给 3B 的自然语言渲染：

```text
Relevant visible facts:
- A person is holding a coffee cup.
- The coffee cup is much larger than the person's hand.

Possible interpretation:
- This may be an ordinary coffee moment exaggerated by the cup's size.

Humor structure:
- Expected: A handheld cup normally fits comfortably in one hand.
- Violation: This cup is disproportionately large.
- Reinterpretation: Treat an ordinary coffee serving as absurdly oversized.
- Suggested strategy: Absurdity or understatement.

Use this information to understand the visual joke.
Do not repeat or explain the guidance.

Generate one short, natural, image-specific humorous caption for this image. Do not explain.
```

重要建议：

> metadata 里可以保存 confidence，但不要轻易把 confidence 喂给 generator，因为它可能让模型输出犹豫、解释、或机械复述。

---

## 11. 当前推荐研究路线

不要马上重新训练。优先做 prompt-only / data-only 的小规模消融，验证哪种指导形式有效。

推荐路线：

1. 构建新的 richer guidance extractor：
   - 使用 7B teacher。
   - 输出 visible facts、inferred context、humor mechanism、optional retrieved knowledge。
   - 不直接写最终 caption。

2. 构建多个 prompt renderer：
   - plain description only
   - old description + simple humor cue
   - raw compact JSON
   - edge list / triples
   - fixed-label natural language
   - fixed-label natural language + RAG
   - original image + attention image + structured text

3. 先做小规模 prompt-only ablation：
   - 例如 100 到 200 张图片。
   - 每图每方法 8 candidates。
   - 相同 seed、相同采样参数。
   - 用 7B blind judge 或人工抽查。

4. 如果 prompt-only 方法显著优于 Plain，再考虑生成高质量 SFT 数据。

5. 如果 SFT 数据质量足够，再训练新的 fresh LoRA。

6. LoRA 训练后必须 gate：
   - 与 Base 盲评对比。
   - Wilson lower bound / p-value / decisive count / usable difference 都要检查。
   - gate 不过，不要继续 full training。

---

## 12. 评估规则建议

评估时至少关注：

- caption 是否贴图
- 是否自然短句
- 是否真的有幽默感
- 是否没有解释
- 是否没有捏造严重视觉事实
- 是否没有照抄 guidance
- 是否没有输出多个 caption
- 是否没有模板化

对照评估应尽量 blind：

- 打乱 A/B 顺序
- judge 不知道哪个是 Base 或 Guided/LoRA
- 同一张图两个方法使用相同候选数
- 记录 ties
- 记录 failures
- 不只看 win rate，也看 decisive count 与置信区间

当前 gate 规则曾用：

```json
{
  "minimum_decisive": 80,
  "wilson_lower_above": 0.5,
  "p_below": 0.05,
  "usable_difference_at_least": 0,
  "max_failure_rate": 0.05
}
```

---

## 13. 新窗口需要特别避免的误解

1. 不要说 pilot LoRA 提升了；它没有通过 gate。

2. 不要说 full training 已经完成；它因为 pilot failed 停止了。

3. 不要说之前用了用户旧 LoRA；已经确认没有。

4. 不要把 `outputs/guided_sft_pipeline/pilot_lora/final_lora` 当成旧 adapter；它是本项目新训练出的 pilot adapter。

5. 不要把严格可见事实规则和新放宽的 RAG/文化推断方向混在一起。正确做法是分层：

   ```text
   visible facts != inferred context != retrieved knowledge
   ```

6. 不要用 CLIP reranker。

7. 不要把 attention image 单独替代原图；应保留原图。

8. 不要默认开始长时间训练。

9. 如果要跑 1000 图严格 A/B，要先检查 1000 子集文件和现有输出是否完整。

10. 如果要继续训练或生成，要先确认当前 GPU/进程状态，避免撞上用户自己在终端跑的任务。

---

## 14. 如果用户问“现在应该怎么做”

推荐回答：

> 先别急着训练。当前 pilot 失败最可能不是因为“模型不能学”，而是 guidance 太稀疏、太保守。下一步应该先做结构化 guidance + prompt-only ablation，找出 3B 真能吃懂的表达方式。等某个 prompt 版本在 100-200 张图上显著赢过 Plain，再把它用于构造训练数据和 fresh LoRA。

建议的第一批实验：

```text
Plain:
  image + original prompt

Desc:
  image + description + original prompt

OldCue:
  image + description + simple humor cue + original prompt

StructNL:
  image + fixed-label visible facts + humor structure + original prompt

StructJSON:
  image + compact JSON + original prompt

StructNL+RAG:
  image + visible facts + inferred context + retrieved knowledge + humor structure + original prompt

Attention+StructNL:
  original image + attention-enhanced image + fixed-label structured guidance + original prompt
```

优先预测：

```text
StructNL > OldCue > Desc
```

Raw JSON 不一定最强，尤其对 3B 小模型。

---

## 15. 如果用户要求继续实现

优先实现顺序：

1. 检查当前项目文件与输出：

   ```bash
   cd /home/zhang.jianqiao/projects/v2.5
   rg --files | rg 'guided|humor|lora|evaluation|caption'
   ```

2. 读取现有脚本，避免重复造轮子：

   ```bash
   sed -n '1,240p' scripts/run_guided_sft_pipeline.py
   sed -n '1,240p' scripts/run_guided_sft_teacher_pipeline_v6.py
   sed -n '1,220p' src/training/guided_humor_sft_dataset.py
   ```

3. 新增 structured guidance extractor。

4. 新增 prompt renderers。

5. 新增 ablation runner。

6. 新增 HTML 可视化页面，方便用户人工看 captions。

7. 小规模跑通后再扩大。

---

## 16. 交互风格

用户偏好直接、清楚、有判断力的协作。

回答时：

- 用中文。
- 先给结论，再给依据。
- 不要把不确定的事情说死。
- 如果涉及已有实验结果，必须引用具体文件或数值。
- 如果准备运行长任务，先说明阶段和风险。
- 用户很在意自己能否在终端看到进度；如果训练需要长时间跑，优先给用户命令，让用户自己在终端启动。

---

## 17. 一句话总纲

本项目现在的关键不是继续堆训练，而是先证明“哪种幽默指导表示能被 Base Qwen2.5-VL-3B 真正利用”；当前最有希望的是：

```text
原图
+ 可选注意力增强图
+ fixed-label visible facts
+ separated inferred context / retrieved knowledge
+ structured incongruity-resolution humor mechanism
+ 原始 caption prompt
→ Base 3B
→ 8 candidates
→ blind evaluation
```

只有当 prompt-only ablation 显著有效后，再构造高质量 SFT 数据并训练新的 fresh LoRA。
