# v3.0 匿名 Group-of-3 盲评说明

## 评审输入

每个 `blind_id` 包含同一张漫画，以及匿名的 A/B 两组三条 caption。A/B 映射保存在独立私有文件中；评审不得接触模型名、训练方法或映射。正式主评测不展示 Planner/数据集的 standard description，以免把图片 grounding 评测退化成文本匹配。仅当评审系统确实不支持图像时，才可生成显式标记的 text-only fallback；其结果必须单独报告，不能冒充主评测。

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
  "decisions": {
    "<blind_id>": {
      "overall": "A|B|Tie",
      "best_pick": "A|B|Tie",
      "best_A_index": 1,
      "best_B_index": 1,
      "absolute_A": "good|weak|bad",
      "absolute_B": "good|weak|bad",
      "candidate_labels_A": ["good|weak|bad", "good|weak|bad", "good|weak|bad"],
      "candidate_labels_B": ["good|weak|bad", "good|weak|bad", "good|weak|bad"]
    }
  }
}
```

不得省略逐候选标签。只有逐候选标签齐全时，报告才会计算 generation-seed 方差。主统计单位始终是图片 cluster，而不是把三条 caption 当作三个独立样本。

## 方法依据

- Humor in AI（NeurIPS 2024）采用成组比较来提高漫画 caption 评价的稳定性。
- Electronic Sheep（ACL 2023）强调对漫画幽默的相关性、质量与多样性进行独立分析。
- v3 在此基础上增加逐候选绝对标签与 image-cluster bootstrap，避免只报告相对胜率。
