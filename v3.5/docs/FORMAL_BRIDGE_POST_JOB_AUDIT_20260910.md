# v3.5 formal bridge post-job audit

审计时间：2026-09-10 22:40 JST
作业：`6754067` (`v35pubform90`)
路线：adapter-free Qwen2.5-VL-7B public-release-362；Planner/Generator 冻结；仅训练 bridge。

## 调度与 provenance

| 字段 | 证据/结果 |
|---|---|
| source commit（提交记录） | `2477f4e`，唯一解析为 `2477f4e461dc4477928a9b8cf3e9cbb4a54120ea` |
| runtime commit | `run_manifest.json` = `2477f4e461dc4477928a9b8cf3e9cbb4a54120ea` |
| source/runtime 一致性 | `pass` |
| formal script SHA-256 | `7d5c3d4cabbf8b2379ea8365b7da5b575e888cb1b98d6bb2c207995c98bebdf8` |
| config SHA-256 | `e1352ff0932212b9ae17acea68b1c095cdfaa8b91e55aa28cdd0a486cf465b7b` |
| 节点/设备 | `b-batch`, `node=1`, `genkai0002`, NVIDIA H100 |
| start → end | `22:20:49 → 22:34:02` |
| 实际耗时 / exit | `00:13:13` / `0` |
| 峰值主机内存 | `6131.8 MiB` |
| CUDA allocator | native；CUDA 12.6；PyTorch `2.12.0+cu126` |

原始 `submission.json` 保持不变；其中的短 SHA 只作为历史调度证据，不能用于新的作业。
所有后续提交必须写入完整 40 位 commit。

## 科学 gate

- `trace_gate.json`: 362/362，missing/extra/failure 均为 0。
- `bridge_input_gate.json`: 362 context/trace，train/validation/test bridge rows 为
  813/132/141。
- `formal_pre_training_gate.json`: pass；`policy_trainable_parameters=0`，bridge-only。
- `run_manifest.json`: bridge trainable parameters `6,949,572`，current public HOMER route 为 true。
- `complete.json`: `status=complete`，5 epochs，340 global steps，best validation total
  `3.8378183390154983`。

## 训练趋势（validation）

| epoch | total | caption NLL | matched−shuffled log-prob |
|---:|---:|---:|---:|
| 1 | 3.97637 | 3.18892 | 0.002057 |
| 2 | 3.89828 | 3.07889 | 0.001407 |
| 3 | 3.86281 | 3.08168 | 0.001047 |
| 4 | 3.84423 | 3.05664 | 0.000831 |
| 5 | 3.83782 | 3.04309 | 0.002100 |

relative update norm 约 `0.0212–0.0344`，低于配置上限 `0.25`；全量 metrics 为有限值，
没有 OOM、NVML 或 NaN。near-zero channel gap 不是失败判定，也不是 latent 因果使用证据；
必须在 test caption 上继续做 matched/text/latent 对照和独立评测。

## 下一道门禁

1. 将已完成的 generation-only provenance/完整键集合修复 commit 并推送。
2. 只提交一个 `b-batch + node=1` generation 作业；不请求 MIG，不复制作业。
3. 通过 `generation_gate.json` 严格验证 47 images × 5 trials × 5 candidates × 2 conditions
   = 2,350 条记录（每 condition 1,175）。
4. 再由 Caption-judgement 建包；HOMER 主评测和辅助盲评分开报告。没有外部评审输出前，不得
   生成 win rate、Pass@K 或 good/weak/bad 数字。

参考依据：HOMER 的分阶段图像幽默生成协议（[OpenReview](https://openreview.net/pdf?id=SzaRhPom4o)）；
Qwen2.5-VL 模型与视觉语言接口（[论文](https://arxiv.org/abs/2502.13923)）；下游生成对齐应
先完成表示对齐再验证任务输出（[BLIP-2](https://proceedings.mlr.press/v202/li23q)）。
