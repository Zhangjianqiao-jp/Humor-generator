# 本项目评测机制：可中断、可恢复、可审计

本文档是项目评测的总入口。目标是把一次大规模幽默 caption 评测变成一个与聊天上下文无关的磁盘状态机：配额耗尽、SSH 断开、Codex 进程退出或节点重启后，重新执行同一个入口即可从最后一个通过校验的 batch 继续，不重复消费已经完成的评审，也不需要重新向 Codex 解释项目背景。

## 评测对象与科学边界

当前 evaluation root 为：

```text
outputs/caption_judgement/pretrained_bridge_comparison_20260911/
```

其中每个匿名 packet 比较 `latent_bridge` 和 `text_homer_context_replay` 两个条件；评审者只看到 HOMER 的 `Cartoon description / Caption A / Caption B` 文本，不看到条件名、模型名、私有 A/B 映射或盲评 secret。当前完整网格为：

```text
47 images × 5 trials × 5 candidates × 3 reference groups × 5 references × 2 conditions
= 35,250 pairwise packets
```

HOMER 的 primary 结构（5 candidates、Pass@1/3/5、重复 trial、pairwise evaluator）来自 [HOMER 论文](https://arxiv.org/abs/2602.06423) 和 [官方实现](https://github.com/Shang-hub/HOMER-Official-Implementation)。New Yorker Caption Contest 数据背景参考 [Hessel et al., ACL 2023](https://aclanthology.org/2023.acl-long.41/)。

本轮使用的是无 API key 的 account-backed Codex auxiliary track，评审模型固定为 `gpt-5.5`、`gpt-5.6-sol`、`gpt-5.6-terra`。可调用性与实际 model ID 会写入状态和最终报告；`gpt-5.4` 不可用时不会静默替换。temperature 只记录为 `requested-0-not-exposed-by-codex-cli`，不能声称 CLI 强制了独立 temperature=0，也不能把该轨道冒充 canonical `gpt-5-chat-latest` API 结果。

## 组件关系

```text
immutable public packets + provenance
                 │
                 ├── 12 ordered shards
                 │       └── each shard → 100-packet Codex batch
                 │                         ├── atomic checkpoint JSON
                 │                         └── stdout/stderr log
                 │
                 ├── strict shard validator
                 │       └── judge_<model>.jsonl
                 │
                 ├── exact canonical-order merger
                 │       └── one complete JSONL per evaluator
                 │
                 └── local private finalizer
                         ├── Pass@1/3/5
                         ├── image-cluster bootstrap CI
                         ├── pairwise evaluator agreement
                         └── majority-vote auxiliary summary
```

公开 packet 和每个 batch 都带有 packet ID、官方 prompt hash、source provenance hash、model ID、decision 和温度协议字段。私有 mapping 与 `blind.secret` 只在最后的本地 finalizer 阶段读取，因此恢复过程不会把测试身份泄漏给评审 agent。

## 状态机

```text
initialized
   │
   ├─ worker produces valid batch ──> running (checkpoint advances)
   ├─ quota with reset time ────────> blocked_waiting → auto wake-up → running
   ├─ auth/model unavailable ───────> blocked_waiting → user repairs account
   ├─ timeout/5xx/rate-limit ───────> blocked_waiting → bounded backoff → running
   ├─ repeated crash without progress > budget → restart_limit_exceeded
   └─ all 3 evaluator sets complete + Terra audit + finalizer sentinel → complete
```

控制器的 `.evaluation_controller.lock` 防止同一个 root 重复启动。worker 启动时记录 PID、命令和时间；进程探测必须同时匹配本 evaluation root、runner 脚本和 model 参数，不会被服务器上另一个项目的同模型调用误判为“已在运行”。

## 一条命令启动/恢复

```bash
cd /home/pj26000152/ku60000936/projects/Humor-generator/v3.5
scripts/resume_homer_evaluation.sh start --interval-seconds 120
```

SSH 断开后重新执行同一命令即可。只查看状态、不启动 worker：

```bash
scripts/resume_homer_evaluation.sh status --json
```

如果 evaluation root 不同，显式传入路径；不要让恢复 agent 猜路径：

```bash
HOMER_EVAL_ROOT=/abs/path/to/root scripts/resume_homer_evaluation.sh status --json
```

要让 controller 在 SSH 退出后继续运行，可使用 tmux：

```bash
tmux new-session -d -s homer-eval \
  'cd /home/pj26000152/ku60000936/projects/Humor-generator/v3.5 && \
   scripts/resume_homer_evaluation.sh start --interval-seconds 120'
```

## 配额/中断恢复规则

1. batch 只有在完整 JSON、正确 packet 顺序和 A/B 决策通过验证后才会原子落盘。
2. 已通过的 batch 在恢复时直接跳过；残缺或旧格式文件改名为 `.invalidN` 并保留证据。
3. usage-limit 若含可解析 reset 时间，写入 `worker_blocked.json`，到点自动解除；无 reset 时间、登录过期或模型不可用则暂停，不盲目重试。
4. 网络 5xx、rate-limit 和单次调用超时采用 5 分钟退避；未知错误不填默认判断，先看 `evaluation_controller.log`。
5. 每模型的安全重启预算按“无新 checkpoint 的连续启动次数”计算，默认 100；只要 checkpoint 前进就归零。旧版本累计启动次数只作为历史 telemetry，不会突然阻塞升级后的健康任务。
6. 若一个大 batch 连续三次返回缺行、乱序或损坏 JSON，runner 将其视为输出截断而非评审结果：保留每个 `.invalidN`，把区间递归二分，两个子区间均通过校验后再原子写回父 checkpoint。父区间与子区间同时存在时，controller 按最大的不重叠区间计数，避免进度重复计算。
7. 只有三个模型都完成 35,250 行、Terra 五个历史 shard 完成同模型重审并 promotion、merge 严格通过后，才读取私有 mapping 并生成最终报告。

## 恢复时只读取的文件

```text
evaluation_manifest.json          # packet/protocol/model/hash 合同
evaluation_progress.json          # 每模型 durable rows 和 Terra gate
evaluation_controller_state.json  # PID、命令、退避、重启预算、finalizer 状态
evaluation_controller.log         # 可审计事件与错误原文
<model>_account_batches/*.json    # 已验证 batch checkpoints
<model>_account_batches/*.log     # 每次 agent 调用证据
```

恢复 agent 不需要阅读聊天记录、重新搜索论文、重新构造 prompt，也不允许读取 `private_mapping.jsonl` 或 `blind.secret`。

## 完成判据

只有同时满足以下条件才报告“评测完成”：

- `multimodel_summary.json` 的 `status` 为 `complete_auxiliary_multimodel`；
- packet_count 恰为 `35,250`；
- evaluator 集合恰为 `gpt-5.5`、`gpt-5.6-sol`、`gpt-5.6-terra`；
- 每套判断一对一覆盖 canonical packet ID，无缺失、重复或 provenance 漂移；
- Markdown 报告非空，并明确 account-agent、无 API key、temperature caveat、非 canonical GPT-5 API。

在完成 sentinel 之前，任何局部 A/B 比例都只能作为进度，不得解释成科学结论。最终报告也只支持该 evaluator/reference protocol 下的相对 Pass@K；不能把“相对胜出”改写成绝对的“真正好笑”。

## 工程验证

```bash
.venv/bin/python -m py_compile \
  scripts/homer_evaluation_controller.py \
  scripts/run_homer_account_agent_batches.py \
  scripts/finalize_homer_auxiliary_models.py
.venv/bin/pytest -q \
  tests/test_homer_evaluation_controller.py \
  tests/test_homer_auxiliary_runner.py \
  tests/test_homer_auxiliary_finalizer.py
scripts/resume_homer_evaluation.sh status --json
```

工程可恢复性不等于科学有效性。科学结论仍需在完整评审集上查看 Pass@K、image-cluster CI、评审一致性和条件间差异，并与 HOMER 的 canonical 轨道分开报告。
