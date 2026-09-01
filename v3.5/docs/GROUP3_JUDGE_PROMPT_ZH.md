# v3.5 匿名 Group 盲评说明（主 Group-of-10 / 历史 Group-of-3）

## 评审输入

每个 `blind_id` 包含同一张漫画，以及匿名的 A/B 两组 caption。主评测每组 10 条；每组 3 条仅用于历史连续性。两种 group size 必须生成和汇总为不同报告。正式 Group-of-10 对每个比较发出 A/B 镜像项，用于检测位置偏差。每个 packet 还必须携带相同的五个非测试图片 A/B 人类偏好校准例；其 schema 在 `configs/evaluation/five_shot_calibration.schema.json`，其 hash 写入私有 mapping。A/B 映射保存在独立私有文件中；评审不得接触模型名、训练方法或映射。正式主评测不展示 Planner/数据集的 standard description，以免把图片 grounding 评测退化成文本匹配。仅当评审系统确实不支持图像时，才可生成显式标记的 text-only fallback；其结果必须单独报告，不能冒充主评测。

## 判断标准

按以下优先级综合判断整组：

1. 图片/标准描述的事实一致性与具体性；
2. 是否抓住漫画中的冲突或违和点；
3. 幽默效果、意外性与原创性；
4. 简洁、自然，适合作为 New Yorker 风格 caption；
5. 避免通用 meme 套话、幻觉对象和只靠语言形式像笑话的捷径。

`overall` 只能是 `A`、`B` 或 `Tie`。相对胜出不代表真正好笑，因此还必须对 A/B 整组及每条候选分别标注 `good`、`weak`、`bad`。

- `good`：与图片紧密相关，有清晰且有效的幽默转折，可作为真实可用结果；
- `weak`：基本相关但笑点普通、解释性强、陈词滥调或力度不足；
- `bad`：不相关、幻觉、难以理解、明显不成笑话，或质量不可接受。

## 强制 JSON 格式

```json
{
  "rater_id": "independent_judge_name",
  "judge_metadata": {
    "provider": "provider_name",
    "model": "exact_model_name",
    "version_or_date": "exact_version_or_access_date",
    "temperature": 0,
    "prompt_sha256": "sha256_of_this_prompt"
  },
  "decisions": {
    "<blind_id>": {
      "overall": "A|B|Tie",
      "best_pick": "A|B|Tie",
      "best_A_index": 1,
      "best_B_index": 1,
      "absolute_A": "good|weak|bad",
      "absolute_B": "good|weak|bad",
      "candidate_labels_A": ["one good|weak|bad label per A caption"],
      "candidate_labels_B": ["one good|weak|bad label per B caption"]
    }
  }
}
```

`candidate_labels_A/B` 的数组长度必须严格等于当前 group size；`best_A/B_index` 为 1-based，范围也是 1..group size。不得省略 judge provenance、best index 或逐候选标签。A/B 两组内部的候选顺序也已独立随机化；私有 mapping 保存随机顺序后的 seed 映射。主统计单位始终是图片 cluster，而不是把每条 caption 当作独立样本。汇总同时报告 `overall` 与 `best_pick`，使用 image→rater 两层 bootstrap、Krippendorff nominal alpha，并对预注册比较做 Holm 多重比较校正。

## 方法依据

- Humor in AI（NeurIPS 2024）正式使用每组 10 条 caption 的 Group Overall / Group Best Pick，并以 5-shot examples 校准评审及镜像 A/B 顺序。v3.5 对齐这些核心设计，但增加了 `Tie`、绝对标签、多评审与聚类统计，因此准确称呼是 paper-aligned extension，而非官方 evaluator 的逐行复现。
- Electronic Sheep（ACL 2023）强调对漫画幽默的相关性、质量与多样性进行独立分析。
- v3.5 在此基础上增加逐候选绝对标签、层次 bootstrap、评审一致性和多重比较控制，避免只报告相对胜率。
