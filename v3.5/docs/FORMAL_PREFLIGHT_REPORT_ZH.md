# v3.5 正式训练 Preflight 报告

更新时间：2026-09-01 19:26 JST。

## 门禁顺序

正式实验严格执行：

1. Python compile 与 67 项单元/契约测试；
2. 训练、stress smoke、生成、monitor 四个 CLI 的 import/argument smoke；
3. 独立环境与 v3.5 隔离检查；
4. 两个冻结 7B adapter 的逐文件 SHA-256 检查；
5. 全量数据逐行、逐图片、逐原始输入文件检查；
6. 666 条 Planner trace 逐文件、逐 channel 检查；
7. 单完整 GPU/allocator 资源检查；
8. 首训练样本与最大原始图片样本的真实 forward/backward stress smoke；
9. 只有以上全部通过才进入正式 bridge training。

任何一步非零退出都会阻止后续步骤。

## 已完成的 CPU 与数据检查

- Python/CLI/测试：全部通过，67 tests passed；
- dataset rows：`2846/2846`；
- unique image files：`949/949`，全部存在、可完整解码且实际 SHA-256 与记录一致；
- upstream source files：`367/367`，全部 SHA-256 与 dataset manifest 一致；
- image clusters：810，所有 split 两两无交叉；
- train + validation Planner traces：`666/666`；
- trace failures/missing/extra/duplicate/invalid：全部为 0；
- 每条 trace 的 conflict/local/global tensors 均完成 hash、shape、finite、token ID、
  causal replay/provenance 检查；
- Planner 与 Generator adapter 的 manifest 内所有文件逐字节通过。

机器可读证据位于：

- `results/preflight/dataset_audit/dataset_audit_summary.json`
- `results/preflight/dataset_audit/row_checks.jsonl`
- `results/preflight/dataset_audit/image_checks.jsonl`
- `results/preflight/dataset_audit/source_input_checks.jsonl`
- `results/preflight/trace_audit/trace_audit_summary.json`
- `results/preflight/trace_audit/trace_checks.jsonl`
- `results/preflight/formal_preflight.json`

这些文件只记录 ID、hash、shape、尺寸和 pass/fail，不输出 held-out caption/description
文本，因此不会把 test 内容用于选择模型或阈值。

## 尚未完成的门禁

最大图像的真实 7B forward/backward 需要 GPU。作业 `6669043` 在同一正式 allocation
内先执行 stress smoke；只有 `resource_smoke.json` 证明两个样本都通过、receiver
trainable parameters 为 0、bridge update 非零、visual tokens 不超过 1280，才运行训练。

## 参考依据

- Qwen2.5-VL Technical Report: https://arxiv.org/abs/2502.13923
- QLoRA, NeurIPS 2023:
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html
- Hidden Technical Debt in Machine Learning Systems, NeurIPS 2015:
  https://proceedings.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
