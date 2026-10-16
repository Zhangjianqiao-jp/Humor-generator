# 执行恢复与 FP32 GPU 复验交接

## 2026-10-16 本轮实际进展

- 工具通道、shell、文件访问、Git 远程访问恢复；之前宿主关闭 stdout 的具体根因仍未确认。
- 原始 latent 工作区及未提交修改保留。审阅分支恢复至持久目录，不再依赖节点本地 /tmp。
- PR 代码参考提交：`c9fff47f3614a2d31cbe8e152df1e01de6962ba6`。提交作业时逐文件核对 PR 的运行代码、配置、测试和 job 脚本，与原始工作区一致。
- CPU 定向回归：`tests/test_bridge_losses.py`，9 passed in 73.19s，包含 BF16 logits 的 FP32 NLL 回归。
- 已提交 GPU job **7097141**，两样本、一次 bridge 更新。记录时处于 QUE，不能计为通过。
- 调度器当时预计 2026-10-17 00:10 JST 启动；这是估计，不是完成时间。

## 可恢复入口

- 原始工作区：`/home/pj26000152/ku60000936/projects/Humor-generator/v3.5`
- 审阅工作树：`/home/pj26000152/ku60000936/projects/Humor-generator-v35-review`
- genkai0001 上的 tmux：`v35-recovery`，日志窗口 `fp32-7097141`。
- 结果目录：`results/engineering_smoke/v35_selected_plan_fp32_20261016_recovery/`
- 作业日志：`v35_selected_plan_smoke.pjm.7097141.out` 与 `.stats`。
- PR：https://github.com/Zhangjianqiao-jp/Humor-generator/pull/7

## 完成判据与后续操作

1. 不重复提交。先检查 job 7097141 状态、日志和 `complete.json`；只有作业消失不代表成功。
2. 需要 complete/preflight/smoke 均通过，且配置哈希一致；核对 runtime_source_manifest 中 losses.py 哈希与 validation_evidence.json 的提交记录一致。
3. 复核两独立 cluster、冻结模型无训练参数、非零有限 bridge 梯度与更新、no-memory 验证和显存峰值。
4. 通过后在 PR 更新 GPU 实测证据；失败则保留原输出，在新目录重跑修复版本。
5. 此验证不覆盖真实生成 KV cache，不证明幽默收益；不以工程 gate 通过启动无条件的大规模训练。

PJM 作业独立于当前聊天执行；tmux 日志窗口只负责显示，不会自动恢复 Codex 推理或自动更新 PR。

## 参考

[BLIP-2, ICML 2023](https://proceedings.mlr.press/v202/li23q.html) 提供冻结骨干加可训练接口的先例；不是本次工程测试结果或本方法幽默效果的证明。
