# v3.5 HOMER 复现与 Ω/Bridge 修正版执行计划

更新时间：2026-09-06
状态：只完成工程与协议修正；正式 7B trace、bridge 训练和 caption 结论尚未解锁。

## 目标与声明边界

本计划分成三个互不混淆的层次：

1. **HOMER-comparable 主轨道**：固定官方 public-code 阶段、prompt、数据分组和
   Pass@1/3/5 评测协议；本地模型只能称为 pinned Qwen2.5-VL substitution，不能称
   论文原始权重的 100% reproduction。
2. **Ω 控制扩展**：使用本项目显式注册的 `project_omega_grid_v1`，研究叙事策略和
   语言风格控制；它不是官方发布的 Ω 离散词表，不能进入 exact baseline。
3. **latent bridge 扩展**：在主轨道通过数据和 zero-adapter baseline 后，冻结两个
   7B policy，只训练 bridge；其结果是本项目新增方法，不属于 HOMER 原方法。

所有结果必须带有 `track`、`variant`、model revision、prompt hash、数据 manifest
hash、代码 commit、seed 和 evaluator provenance。任何缺失一项的结果只能是 smoke，
不能进入论文主表。

## Phase 0：不占 GPU 的数据与来源门禁

### 0.1 population 版本与 365 论文口径

HIA 论文报告 365 contests；本地 ranking 目录有 385 个 contest 文件，官方 HOMER
评测 JSON 也有 385 条，而公开 standard-description release 只有 362 条。这些数字来自
不同发布物，不能用 numeric range 或目录全集互相替换。经过本轮审计，已经建立并封存
一个有完整机器可核验来源的当前版本：

```text
data_version = homer_pretrained_7b_public_release_362
source = yguooo/newyorker_caption_ranking@1cd70477b6a99a473690a25a2fed359f75184c64
contest_ids = exact union(train=271, validation=44, test=47)
population_key = (dataset, contest_number, image_sha256)
manifest_sha256 (self-hash) = 589448e581839cab49db074e989091829437116d1f13a68194e36828f3b32af7
allowlist_sha256 = 367e283ef1b454875a1bfe9c7a8eafb26f442b2ca8ec0a0c12279c4f31cba54d
```

因此 **adapted public-release route 的 population gate 已解决**。362-ID allow-list
及其三份 description 文件、图片和 ranking join 均由 verifier 逐项检查。HIA 论文的
“365”是仍未有作者发布机器可读清单支撑的 paper-level claim；不能把 362 个公开
description 通过猜测、补零或 numeric range 强行扩成 canonical 365。该 canonical claim
单独记录为 `unresolved`，不再阻塞当前 adapted route。

当前封存的 source-aware allow-list 为：

```text
(dataset, contest_number, image_sha256)
```

同时输出 allow-list 文件 hash、纳入/排除原因和每 contest caption 行数；详情见
[`docs/HOMER_PUBLIC_RELEASE_362_ZH.md`](HOMER_PUBLIC_RELEASE_362_ZH.md)。历史
2,846-row representative pilot 仍不能代替当前 population。

### 0.2 contest 878 字节 lineage

旧 v1 的三条 878 记录确实曾与磁盘字节不一致；本项目没有原地改写旧 manifest，而是
建立 `homer_pretrained_7b_v2` 父版本并记录 old/new SHA-256 lineage。当前
`homer_pretrained_7b_public_release_362` 从当前本地 source bytes 重新生成，
`image_repairs_relative_to_parent=[]`，并由 verifier 检查 362 个图片 hash 和 ranking
hash。因而 878 已不再是当前 population 的 blocker；历史 v1/v2 结果仍不可冒充当前
版本。

### 0.3 通过条件

adapted public-release route 允许生成当前预训练 trace 的条件为：

```text
official assets hash/schema = pass
public_release_362 verifier = pass
all image/ranking/description joins = pass
current image-byte audit = pass
source-aware trace-input manifest = present
```

若论文需要声称 **canonical 365-contest**，还必须额外满足：

```text
authoritative 365-ID allow-list = verified
all 365 image bytes = present and hashed
```

这是一条独立的 claim gate；当前没有把它伪装成已通过。

## Phase 1：HOMER-comparable pretrained-7B baseline

### 1.1 模型与管线

Planner 和 Generator 都使用本地固定的
`Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`，不加载
SFT adapter；两个 policy 冻结，bridge 在本阶段为零或关闭。执行完整阶段：

```text
description → conflict → global imagination → local imagination
→ independent summary → retrieval/pruning
→ select conflicts → select entities → DFS/sample path
→ ##Caption + ##Explanation
```

`standard_description_replay`（使用官方描述、7 requests）和 `online_description`
（在线 description、8 requests）分开统计。论文附录的 2+3+2=7 与 raw
`generator.py` 的 8-request 证据同时保存，不能静默选一个数字。

待测 base model 可以变化，但必须作为独立的 `model_variant` 重跑同一协议；不能把
GPT-4o、Claude-4、Qwen-VL、LLaVA-1.5 等不同模型的候选混在同一 Pass@K 分母。HOMER
论文的 Table 3 正是以相同的五候选/Pass@K 框架比较这些 base models；本项目当前只先
注册固定 revision 的 Qwen2.5-VL-7B，其他模型属于后续独立对照。

### 1.2 successor 与随机性

主 baseline 固定 public prompt 要求的 `root + 3 successors`。`n∈{2,3,4,5}` 仅在
validation 上做预注册小 pilot，不能在看到结果后改变主 baseline。conflict/entity
selection 保留官方独立请求；DFS path 用记录 seed 的 `random.Random(seed)`，并标注为
reproducibility adaptation。

## Phase 2：HOMER 主评测

默认 canonical 评审器为官方脚本的 `gpt-5-chat-latest`（论文标签 GPT-5），temperature
为 0；caption generation temperature 为 1.0。每图生成 5 个候选，5 次重复，报告：

```text
Pass@1, Pass@3, Pass@5
HIA #top10, #200–209, #1000–1009
Electronic Sheep High-Humor, Low-Humor
```

同时保留官方要求的 visual understanding、humor understanding、stylistic
expression 诊断和 diversity 指标。每次评测保存候选顺序、seed、评审 prompt、实际
model ID/snapshot、请求 hash 和失败重试记录。

若实际部署不能使用 GPT-5，可以**预先声明 evaluator substitution**：保持候选数、
Pass@K、重复次数、参考组、温度和 prompt 完全不变；结果配置必须写实际 model ID、
snapshot/date、`evaluator_substitution=true`，并只能标为
`HOMER-protocol adapted evaluation`。替代评审结果不能和 canonical GPT-5 数字合并。

HIA NeurIPS 2024 的 Group Overall（GPT-4-Turbo + Hessel description）和 Group Best
Pick（GPT-4o-vision + raw image）是独立 compatibility 表，不替换 HOMER 的主表。

## Phase 3：Ω 扩展（只在 baseline 完成后）

论文写出 Ω∈NS×LA，但没有可核验的有限词表。项目采用固定的：

```text
NS = 6: setup_punchline, deadpan_observation, question_answer,
     dialogue_voice, role_reversal, exaggerated_consequence
LA = 6: plain_wit, pun_wordplay, idiom_twist, sarcastic_understatement,
     personification, metaphorical_comparison
```

36 个组合以稳定顺序预注册。命令行只有显式提供
`--omega 'NS|LA'` 才开启；不提供时严格是 public-code baseline。Ω 结果单独按
`project_omega_grid_v1` 报告，不声称复现官方 Ω。

## Phase 4：latent bridge 扩展

只有 Phase 0–2 的文本 baseline 完成并通过 provenance gate 后，才开始 bridge：

```text
Text-HOMER
StateBridge-inspired (training-free)
Learned bridge
Typed learned bridge
```

两个 7B policy 继续冻结；第一轮只训练 bridge。先用小规模 semantic/caption gate
排除工程错误，再在同一 HOMER 主评测上比较 text/latent/hybrid。latent 的 matched/
shuffled、channel-isolated 和 donor reconstruction 只作为机制诊断，不能替代最终
caption 的盲评。

现有 Group-of-10、镜像 A/B、三个独立 judge、`good/weak/bad`、seed 方差和
image-clustered bootstrap 继续保留，但固定为 `project_extension` 辅助轨道；不得与
HOMER Pass@K 主表混算。它适合分析绝对质量、位置偏差和 cluster 相关性，不是 HOMER
论文的 primary evaluation。

## 当前 Go/No-Go

```text
工程 smoke：通过
模型/协议静态检查：通过
评测资产 hash/schema：通过
adapted public-release 362 population：已验证 → Go
canonical 365 paper claim：未验证 → 仅对 canonical claim No-Go
contest 878 current-byte lineage：已验证 → Go
正式 7B trace：未生成
正式 bridge/caption 训练：未提交
```

因此数据来源 blocker 已解决（针对 adapted route）。下一步是运行当前预训练模型的
zero-adapter smoke，并按 `trace_inputs.jsonl` 生成新的 362 条 Planner trace；在 trace
gate 通过前仍不提交 bridge GPU 训练。若未来要发表 canonical 365 数字，必须先获得
作者/官方机器可读 allow-list，而不是从现有 362 条推断。

## 权威依据

- Shang et al., *On the Wings of Imagination: Conflicting Script-based Multi-role
  Framework for Humor Caption Generation*, ICLR 2026：
  https://arxiv.org/html/2602.06423
- HOMER 官方固定实现：
  https://github.com/Shang-hub/HOMER-Official-Implementation
- Zhang et al., *Humor in AI*, NeurIPS 2024：
  https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf
- Hessel et al., *Do Androids Laugh at Electric Sheep?*, ACL 2023：
  https://aclanthology.org/2023.acl-long.41/
- Bai et al., *Qwen2.5-VL Technical Report*：
  https://arxiv.org/abs/2502.13923
