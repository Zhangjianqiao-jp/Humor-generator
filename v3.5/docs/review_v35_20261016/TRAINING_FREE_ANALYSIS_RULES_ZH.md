# 无训练对比结果分析规则

分析器 `scripts/analyze_training_free_comparison.py` 已准备；尚无真实对比结果。可以对已完成的单批或多个不重叠批次运行。输出 contest 数量只指提供的批次，不暗示 16 个样本全部完成。

示例（在 v3.5 下运行）：

```bash
python scripts/analyze_training_free_comparison.py \
  --run-dir results/engineering_smoke/v35_training_free_serial_20261017_batch0 \
  --context-index data/cache/homer_pretrained_7b_homer_context/index.jsonl \
  --output results/preflight/v35_effectiveness_20261017/batch0_analysis.json
```

多个批次用多个 `--run-dir` 参数，输出文件必须不存在。文件必须包含 manifest 规定的全部 cluster/arm/task/seed；缺失、重复、跨批次 cluster 重叠、context 哈希或实验契约不一致均报错。不静默删除失败样本或只分析完成得好的条件。

## 指标

- 格式：保留所有 parse_status，以及到达生成长度上限的数量。
- 恢复：实体集合严格一致（不要求顺序，拒绝重复实体）；路径包含链条顺序的严格一致；冲突只忽略大小写/空白；完整 JSON 精确一致。
- 无计划：{} 是正确遵循“无计划返回空对象”的行为，与完整计划恢复分开报告。
- 条件间：相同 cluster/seed 的有效 caption 归一化后完全一致率，单列双方有效的分母，不称作幽默相似度。
- 格式率差：相对 text_plan，先在 contest 内平均两次 caption 的成功指示量，再对 contest 配对差值做 10,000 次 bootstrap。少于两个 contest 不给区间；这些是探索性边际区间，未作多重比较校正，不支持确认性质量结论。

不做自动幽默裁判。不把 exact recovery 当作语义恢复的充分测量；同义改写/格式/截断须另行审阅。no-memory 与文本条件长度不同，也没有等长无意义前缀控制，不能用一次比较宣称已隔离所有提示长度因素。

## 已完成验证

`python3 tests/test_training_free_analysis.py`：8 项通过，覆盖空对象、无效 JSON、实体/路径顺序、异常字段、单图区间、缺失调度、重复记录及失败样本分母。验证只针对分析器，不替代 GPU 模型结果，也未重跑历史完整测试集。

## 参考

[Cawley & Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)：模型选择与评估偏差背景。图片级重采样是本实验依据相关结构作出的设计，不是该论文提供的本任务统计保证。
