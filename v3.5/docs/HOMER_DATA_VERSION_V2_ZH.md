# HOMER v2 数据版本与输出血缘

更新时间：2026-09-06

## 目的

`homer_pretrained_7b_v2` 是在不修改历史文件的前提下建立的新数据版本。它复制
`data/processed/latent_bridge_v35` 的 JSONL 输出，并只修复图像字节 hash 不一致的字段；
caption、standard description、split、排序和 cluster 语义均不改写。

## 版本文件

- lineage manifest：`manifests/homer_population_v2.json`
- 生成脚本：`scripts/create_homer_population_v2.py`
- 完整性检查：`scripts/verify_homer_population_v2.py`
- copied artifacts：`data/processed/homer_pretrained_7b_v2/`（数据目录被 git 忽略，
  由 manifest 中的每个文件 hash 固定）
- pretrained-only route 配置：`configs/homer_public_code_pretrained_7b.yaml`

## 唯一修复

HIA contest 878 的历史 hash：

```text
c87ee5f587d420f095fc35dae6c46780c39d767ffd3662d7ce5ea1e4369e4e01
```

当前磁盘 `878.jpg` 的 canonical hash：

```text
b690e9d81a36d6a2cf5fd7a8bbf411ecd4c0f959afb6d97aba1f8f0f4af7ae0a
```

共更新 4 个依赖字段：三个 `humor_in_ai:878` caption rows 和一个 trace-input row。
旧 hash、新 hash、路径、row IDs 和 source files 都写入 v2 manifest；没有原地覆盖 v1。

## 当前检查结果

```text
v2 verification: pass
copied rows: 3633
image repair groups: 1
formal population gate: blocked
```

`blocked` 不是脚本失败，而是 fail-closed 结果：HIA 论文报告 365 contests，而本地发布
物的 standard descriptions/evaluator population 仍有 362/385 的发布物差异。没有可核验的
365-contest source-aware allow-list 时，不能把本地目录全集、numeric range 或历史
representative rows 伪装成论文人口。因此 v2 已恢复字节 provenance，但尚未解锁正式
Planner trace、Qwen baseline caption 或 bridge training。

下一步只有在 allow-list、图片/排名/description join、trace index 都通过后，才按固定
顺序运行：

```text
pretrained Qwen2.5-VL-7B no-adapter Text-HOMER baseline
→ HOMER Pass@1/3/5, five trials, GPT-5 protocol
→ project Ω extension (separate)
→ frozen-policy bridge-only experiments
```

Group-of-10/multi-judge/absolute labels 仍是 Caption-judgement 的辅助扩展，不与 HOMER
主轨道的 Pass@K 分母合并。

## 依据

- HOMER：<https://arxiv.org/html/2602.06423>
- HOMER official implementation：<https://github.com/Shang-hub/HOMER-Official-Implementation>
- Humor in AI benchmark：<https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf>
