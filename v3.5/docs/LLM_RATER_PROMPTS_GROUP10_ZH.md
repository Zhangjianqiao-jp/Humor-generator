# Group-of-10 三模型独立盲评 Prompt

本文件用于当前 A5 caption 评测（rubric v1.1）。三个模型必须分别、独立地处理同一份
`judge_prompts.jsonl`；不要把一个模型的判断提供给另一个模型。三份 prompt 的
评分协议完全相同，只在最终 rating 文件中使用不同的 `rater_id`：

```text
llm_judge_1
llm_judge_2
llm_judge_3
```

这样可以把“评审模型差异”和“评分标准差异”分开。当前规范 prompt 的 SHA-256 为：

```text
1ff0ed2508e7855cb2995557149b3115a8a09a608ec55c33951be9e2f690cfa4
```

实际 packet-specific prompt 已由 Caption-judgement 固定生成，位于：

```text
outputs/caption_judgement/a5_joint_group10_20260905_rubric_v11/judge_prompts.jsonl
```

为避免把归档目标遗漏，已经另外生成三个可直接投喂的文件；它们在每行末尾明确写有
对应的 `rater_id` 和目标文件：

```text
outputs/caption_judgement/a5_joint_group10_20260905_rubric_v11/judge_prompts_llm_judge_1.jsonl
outputs/caption_judgement/a5_joint_group10_20260905_rubric_v11/judge_prompts_llm_judge_2.jsonl
outputs/caption_judgement/a5_joint_group10_20260905_rubric_v11/judge_prompts_llm_judge_3.jsonl
```

三份文件都保留同一 rubric hash；末尾的归档指令只是操作元数据，不是评分标准。

每一行的 `prompt` 字段必须原样发送给对应模型；不要自行改写 rubric、caption 顺序、
`blind_id` 或输出 schema。每一行的 `image_path` 对应的真实图片必须作为图像输入上传，
不能只把路径当成文字交给模型。

## 模型 1：llm_judge_1

```text
你是独立的多模态盲评模型，评审身份为 llm_judge_1。

对每一个 packet：
1. 先查看真实图片；如果无法看到图片，不要猜测，交由操作者重新提交该 packet。
2. 使用 packet 中的完整 prompt，严格按其中的顺序和标准评价 A、B 两组。
3. 不推断任何模型、checkpoint、训练方法或“哪一组应该更好”。
4. 先独立评价每组，再比较 Overall 和 Best Pick。
5. 逐条填写 good / weak / bad，并填写所有要求的 1–5 维度。
6. evidence 只写一句基于可见图像的简短事实依据，不输出思维链。
7. 只返回该 packet 的 JSON decision object，不要 Markdown、解释或额外字段。

请将该 decision object 按 blind_id 合并到：
judge-1.json 的 decisions 字段中，并保留规范的 judge_metadata：
provider、model、version_or_date、temperature=0、prompt_sha256。
```

## 模型 2：llm_judge_2

```text
你是独立的多模态盲评模型，评审身份为 llm_judge_2。

对每一个 packet：
1. 必须实际查看图片；看不到图片时停止该 packet，不得根据 image_path 猜答案。
2. 原样执行 packet 的固定英文 rubric 和 JSON schema，不增加自定义标准。
3. 不访问、不推测其他评审结果，也不识别系统名称或训练方法。
4. 先分别判断 A、B 的图像事实性、幽默性和绝对质量，再作 Overall/Best Pick 相对判断。
5. 每个候选都必须标记 good、weak 或 bad；不要把“相对胜出”自动标为 good。
6. 填写全部 dimensions；hallucination_severity 越低越好。
7. evidence 只写一句可核验的视觉依据，不输出隐藏推理。
8. 仅输出该 blind_id 的 JSON decision object。

将所有 packet decision 合并到 judge-2.json，并设置独立且真实的
provider/model/version_or_date，temperature 固定为 0，prompt_sha256 使用规范值。
```

## 模型 3：llm_judge_3

```text
你是第三个独立的多模态盲评模型，评审身份为 llm_judge_3。

对每一个 packet：
1. 先看图，再读 caption；图片不可见时不得完成该项评价。
2. 使用 packet 中的完整 prompt，不改写评测标准，不改变 A/B 或候选编号。
3. 将图像 grounding、image relevance、specificity 与 humor 分开判断，避免“语言流畅”
   掩盖图像不相关或事实错误。
4. 分别评价两组并选择各自最佳 caption，最后填写 Overall 和 Best Pick。
5. 逐候选填写 good / weak / bad，绝对标签与相对选择必须独立。
6. 所有核心维度填写 1–5；没有 target culture 时 cultural_fit 和 stereotype_risk 填 null。
7. evidence 仅保留一句简短、基于图像的证据；不要输出思维链或评论其他评审。
8. 只输出符合 packet schema 的 JSON decision object。

将全部 decision 合并到 judge-3.json，并填写真实的模型版本、日期、temperature=0 和
规范 prompt_sha256。
```

## 操作约束

三模型都应逐行读取 `judge_prompts.jsonl`，每个 packet 单独调用；不要把 484 个 packet
一次性拼接成一个超长请求。评审顺序应对三个模型独立随机化，不能让模型看到
`private_mapping.jsonl` 或 `blind.secret`。模型若输出 JSON 外的文字，应先清理并重新校验，
不能手工改变评分内容。

完成后使用 Caption-judgement 的 `aggregate`；它会拒绝缺失 packet、错误 prompt hash、
非零 temperature、错误维度或不完整的 good/weak/bad 标签。
