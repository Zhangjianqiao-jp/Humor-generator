---
date: 2026-07-29
project: HIC humorous image captioning
tags:
  - experiment-card
  - reproducibility
---

# Experiment Cards

这个目录保存人类可读的实验卡。它和 `outputs/manifest/runs.jsonl` 配套使用：

- `docs/experiments/*.md`：解释实验为什么做、怎么做、结果如何、下一步怎么决策。
- `outputs/manifest/runs.jsonl`：给脚本和新 Codex 窗口读取的机器可读 run 索引。

每张实验卡至少包含：

```text
commit
status
research question
dataset / split
sample selection
model / adapter
teacher / judge
prompt renderer
generation parameters
output paths
metrics
known caveats
decision
next action
```

不要把大型输出、checkpoint、完整候选 JSONL 复制进这个目录。实验卡只写索引、指标和判断。