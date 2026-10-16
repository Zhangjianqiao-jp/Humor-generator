# HOMER 三模型盲评：可恢复工程闭环

本文件是本项目评测的独立执行说明。评测中断后不需要恢复聊天上下文，也不需要重新检查模型、prompt 或数据；只需回到 `v3.5` 目录再次运行入口脚本。控制器从磁盘上的 manifest 和 batch checkpoint 恢复，已通过校验的判断不会被覆盖。

这是一个“磁盘状态机”，不是依赖当前对话的临时脚本：公开 packet、每个 100-item batch、评审模型、协议 hash 和最终完成标记都保存在 evaluation root。新的 Codex 会话只需读取这些小型状态文件，不需要把本次聊天重新喂给模型。

## 目标与边界

当前评测是 HOMER 官方 pairwise evaluator 的 account-agent 辅助轨道，不冒充官方 GPT-5 API 主结果：

- 条件：`latent_bridge` 与 `text_homer_context_replay`；
- 公开测试集：47 张图片；每图 5 trials、每 trial 5 candidates；
- 官方 reference groups：`1-9`、`200-209`、`1000-1009`，每组 5 条 reference；
- 总匿名 packet：`35,250`（2 × 47 × 5 × 5 × 3 × 5）；
- 评审智能体：`gpt-5.5`、`gpt-5.6-sol`、`gpt-5.6-terra`；当前没有可调用的 `gpt-5.4`，绝不将 Terra 改写成 5.4；
- 不使用 API key。worker 由 `codex exec --ephemeral -m MODEL` 使用已登录 account；controller 和 runner 都会从子进程环境移除所有名称以 `_API_KEY` 结尾的 provider key；
- temperature 协议要求为 0，但 Codex CLI 不暴露独立 temperature 开关，结果中固定写为 `requested-0-not-exposed`，不得声称已强制设置；
- 评审阶段只读取 `homer_primary/*_prompts.jsonl` 和公开 shard；`private_mapping.jsonl`、`blind.secret` 仅由本地 finalizer 读取，不会送入智能体。

HOMER 的论文与官方代码采用多角色生成和独立 pairwise humor evaluator；本项目保持其文本 `Cartoon description / Caption A / Caption B` 输入、匿名 A/B 判断、5 candidates、Pass@1/3/5 和重复 trial 结构。参考：[HOMER 论文](https://arxiv.org/abs/2602.06423)、[HOMER 官方实现](https://github.com/Shang-hub/HOMER-Official-Implementation)。数据集来源和 New Yorker Caption Contest 评测背景见：[Hessel et al., ACL 2023](https://aclanthology.org/2023.acl-long.41/)。

## 一条命令启动或恢复

在服务器上执行（推荐的 context-free 入口）：

```bash
cd /home/pj26000152/ku60000936/projects/Humor-generator/v3.5
scripts/resume_homer_evaluation.sh start --interval-seconds 120
```

这条命令可以在 SSH 断开、终端关闭、配额耗尽、单个智能体进程退出或节点重启后原样重新执行。不要删除 `*_account_batches/`、`terra_reaudit_batches/` 或 `multimodel_agent_shards/` 中的 checkpoint 和 `.invalid*` 证据。

若希望 controller 本身也在 SSH 断开后保持运行，推荐放进 tmux（同一 root 不会重复启动）：

```bash
tmux new-session -d -s homer-eval \
  'cd /home/pj26000152/ku60000936/projects/Humor-generator/v3.5 && \
   .venv/bin/python scripts/homer_evaluation_controller.py \
   --root outputs/caption_judgement/pretrained_bridge_comparison_20260911 \
   --mode start --interval-seconds 120'
```

查看 controller 输出：`tmux attach -t homer-eval`；退出查看但不停止：按 `Ctrl-b` 后按 `d`。重新连接服务器后只需执行 `tmux ls` 和上述恢复命令即可。

入口脚本默认使用当前 evaluation root，也可以显式指定另一份已经封存的 root；它不会猜测数据集或模型：

```bash
HOMER_EVAL_ROOT=/abs/path/to/another/root \
  scripts/resume_homer_evaluation.sh status --json
```

若只想检查而不启动进程：

```bash
.venv/bin/python scripts/homer_evaluation_controller.py \
  --root outputs/caption_judgement/pretrained_bridge_comparison_20260911 \
  --mode status
```

机器可读状态：

```bash
.venv/bin/python scripts/homer_evaluation_controller.py \
  --root outputs/caption_judgement/pretrained_bridge_comparison_20260911 \
  --mode status --json
```

## 文件状态机

每个智能体使用连续 packet shard；每个 shard 又按 100 个 packet 切成独立 batch：

```text
public_packets.jsonl
  -> state/<shard>_<start>_<end>.json       (Codex 返回的原始 batch)
  -> state/<shard>_<start>_<end>.log        (stdout/stderr/attempt)
  -> judge_<model>.jsonl                    (shard 完成后的规范判断)
  -> multimodel_agent_merged/judge_*.jsonl (严格 canonical order)
  -> multimodel_agent_final/multimodel_summary.{json,md}
```

控制器在每个周期执行以下门禁：

1. canonical public packet 与 shard 必须是相同顺序、无重复、无缺失的精确分割；
2. 读取已有 batch 时逐项检查 packet ID、A/B、model、temperature、prompt hash 和 source provenance hash；
3. 不完整或 JSON 损坏的 batch 被保留为 `.invalidN`，下一次只重做该 batch；
4. 只有所有 shard 的规范判断都完整，才允许 merge；
5. Terra 的五个历史启发式 shard（`001201_002400`、`006001_010800`、`010801_015600`、`025201_030000`、`030001_034800`）必须先由同一 account-agent 重新评审。旧文件改名为 `judge_gpt56terra.heuristic_invalid.jsonl` 备份，未审计文件不能通过 Terra complete gate；
6. merge 之后，finalizer 才在本地读取 private mapping，计算每个评审器的 Pass@1/3/5、image-cluster bootstrap CI、评审一致性和三模型 majority vote。

启动时还会把本 evaluation root 的 worker PID、命令、启动时间和是否仍存活写入
`evaluation_controller_state.json`。进程检测同时要求 evaluation root、runner 脚本和
model 参数一致，不会因为服务器上另一个项目恰好使用同一模型而错误地跳过本任务。

## 配额中断的恢复规则

- worker 返回非零状态不会删除任何 batch；控制器下一周期检测不到该模型进程后重新启动同一 runner；
- worker 遇到 account usage limit、token 过期或模型不支持时，会写入对应 state 目录的 `worker_blocked.json`；controller 会持久化阻塞原因并停止无效热循环。usage-limit 若解析到 reset 时间会在到点自动唤醒；登录刷新或启用模型后，删除相应 marker 即可从同一缺口恢复；
- worker 遇到网络 5xx、rate-limit 或单次调用超时会写入带 `retry_after_epoch` 的退避 marker，默认等待 5 分钟后自动继续；未知错误不填充判断，需先看日志再恢复；
- controller 以“无新 checkpoint 的重复启动次数”作为安全预算（`--max-restarts`，默认 100），而不是旧版本累计的历史启动总数。只要有新的有效 batch，预算就归零；持续崩溃会进入 `restart_limit_exceeded`，避免无限消耗配额；
- runner 看到完整、顺序正确的 batch checkpoint 会跳过 Codex 调用；它同时兼容当前的 `<start>_<end>.json` 和早期的 `<start>.json` checkpoint 命名，因此旧 runner 已完成的批次也不会重复消耗配额；
- runner 不把“最终文件存在”误当作完成：会再次验证行数、packet 顺序、model、temperature 和两项 provenance hash；历史版本留下的残缺/过期最终文件会改名为 `.invalidN` 并从 batch checkpoint 重建，避免恢复死锁；
- 截断/错误响应会被改名为 `.invalidN`，保留错误证据并重试，最多每个 batch 3 次；三次仍无效会写入 `invalid_response_exhausted` marker，暂停而不是无限消耗配额；跨配额中断可在新进程中继续；
- 如果同一 batch 三次都只是缺一行、乱序或 JSON 截断，runner 不再无限重试大请求：自动递归二分为较小的 packet 区间，优先复用已有子 checkpoint，只有子区间全部通过严格校验后才写入父 checkpoint。父/子 checkpoint 均保留为审计证据，controller 会按不重叠区间统计进度；singleton 仍无效才写入 `invalid_response_exhausted` 并暂停；
- `.evaluation_controller.lock` 防止同一 evaluation root 启动两个 controller；
- Terra promotion 只有在五个重审 shard 全部严格通过后才原子进行；
- finalizer 只在三套 35,250 行 judgement 都通过一对一 packet gate 后运行。任何缺失都会停止并保留 `evaluation_progress.json`，不会产生部分科学结论；
- 控制器不会自行判断 A/B，也不会把错误响应填成默认 B。

查看阻塞原因：

```bash
for d in gpt55_account_batches sol_account_batches terra_reaudit_batches; do
  test -f "outputs/caption_judgement/pretrained_bridge_comparison_20260911/$d/worker_blocked.json" && \
    jq . "outputs/caption_judgement/pretrained_bridge_comparison_20260911/$d/worker_blocked.json"
done
```

完成 `codex login` 或确认模型重新可用后，只删除对应的 `worker_blocked.json`；不要删除 batch JSON、`.invalidN` 或最终 shard 文件。若出现 `restart_limit_exceeded`，先修复日志中的根因，再以更大的 `--max-restarts` 运行 `scripts/resume_homer_evaluation.sh start ...`，不要清空已有 checkpoint。

## 不需要重新提供给 Codex 的信息

所有恢复所需信息都写在：

```text
evaluation_manifest.json          # 不可变 packet/protocol/model/hash 记录
evaluation_progress.json          # 每模型完整行数、Terra 审计状态
evaluation_controller_state.json  # PID/命令、重启次数、最近周期、finalizer 状态
evaluation_controller.log         # 子进程与 merge/finalize 日志
```

manifest 中保存 canonical packet SHA、官方 prompt SHA、source provenance SHA、模型列表、不可用模型、shard 行数、packet 生成的 `source_commit`，以及控制器恢复所用的 `controller_commit` 和 `public-only during judging` 策略。两种 commit 分开记录，避免恢复工具版本更新后错误改写原始 packet 的 provenance。恢复时无需依赖本轮对话或人工复制 prompt。

`evaluation_progress.json` 同时报告已经封存的完整 shard 行数，以及 state 目录中已校验的 batch 行数/数量和最近 batch；因此配额窗口内可以区分“尚未完成 shard”和“正在处理的 batch”。

## 完成判据与报告措辞

只有出现以下文件且 `status=complete_auxiliary_multimodel` 才算完成：

```text
outputs/.../multimodel_agent_final/multimodel_summary.json
outputs/.../multimodel_agent_final/multimodel_summary.md
```

报告必须明确写：

```text
auxiliary account-agent HOMER evaluation
evaluators = gpt-5.5, gpt-5.6-sol, gpt-5.6-terra
no API key
temperature requested 0, not independently exposed by Codex CLI
not canonical gpt-5-chat-latest API result
```

不能把这条辅助轨道与官方 `gpt-5-chat-latest` 结果合并，也不能把 “A 胜 B” 直接改写成绝对的“真正好笑”；Pass@K 只表示在该 evaluator 和该 reference protocol 下的相对判断。

## 恢复决策表

| 状态/证据 | 允许的动作 | 不允许的动作 |
|---|---|---|
| `running` 且 checkpoint 行数增加 | 等待或重新执行 `start` | 删除 state/batch 文件 |
| `quota_exhausted` 且有未来 reset 时间 | 等待 controller 自动唤醒 | 立即重复提交同一 batch |
| `auth_expired` / `model_unsupported` | 修复登录或模型可用性，再删除对应 marker | 用另一个模型替代并合并结果 |
| `call_timeout` / `transient_error` | 等待退避后自动重试，必要时查看 log | 用默认 A/B 填补 |
| `restart_limit_exceeded` | 修复根因并人工增加预算 | 无限循环重启 |
| 所有模型 complete 且 final summary sentinel 通过 | 读取最终报告 | 在 finalizer 前解释科学结论 |

这套流程只保证工程上的可恢复性和证据完整性；它不把较低 loss、进程退出或评审器的相对胜负自动解释成“caption 真正好笑”。

## 本地验证

修改 controller 或 runner 后先做：

```bash
.venv/bin/python -m py_compile scripts/homer_evaluation_controller.py scripts/run_homer_account_agent_batches.py
.venv/bin/python scripts/homer_evaluation_controller.py \
  --root outputs/caption_judgement/pretrained_bridge_comparison_20260911 \
  --mode status --json
```

然后再用 `--mode start`。不要直接手工运行 finalizer，也不要把 private mapping 复制到智能体工作目录。
