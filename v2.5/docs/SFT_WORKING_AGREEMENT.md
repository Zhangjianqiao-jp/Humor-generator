# SFT 工作准则与本次试验复盘

> **范围更新（2026-08-28）：3B Captioner 与 7B×3B 联合方案继续废弃；活动 Generator checkpoint 仍为 `outputs/7b-generator/best_val_loss`。现新增授权仅限“冻结 7B Planner SFT + 冻结 7B Generator SFT + 可训练小型 latent bridge”的 7B→7B 通信实验，以及通过门禁后的 Text/Latent/Hybrid 三路受控 Generator DPO。不得由此恢复 3B、联合更新两个 7B、joint RL 或未经过 Bridge Go/No-Go 的大规模 DPO。详细协议见 `docs/LATENT_COMMUNICATION_EXPERIMENT_PLAN_ZH.md`。资源门禁、冒烟、checkpoint 验证和可复现要求继续适用。**

适用范围：New Yorker compact 数据上的 7B humor-plan planner 与 3B captioner 的 SFT。本阶段只允许 SFT；未经明确批准，不提交 DPO、reference-logprob 或其他偏好优化任务。

## 资源门禁

任何训练变更均按以下顺序推进，前一关失败时不得提交下一关：

1. **静态检查（不占 GPU）**：YAML 能解析；`python -m py_compile`；`git diff --check`；全量 `pytest -q`。
2. **数据/批处理预检（不占 GPU）**：对两个配置运行 `--debug-data`；验证图像可读、视觉 token 数不超过 `max_seq_len`、监督 token 数大于下限、train/validation 图像隔离。
3. **真实 CPU LoRA step + QLoRA loader 单测（不占 GPU）**：运行 `scripts/preflight_cpu_lora_sft_step.py`，以缓存的真实 Qwen 权重完成图像处理、collator、LoRA 注入、loss、反向和 optimizer step；另以 mock 模型走过 NF4 `BitsAndBytesConfig`、`prepare_model_for_kbit_training` 分支。这个测试不能只做 import 或语法检查。
4. **单步 GPU 冒烟**：每个模型最多一个 MIG 分区、一个样本、一个 optimizer step。成功日志必须包含 `completed one optimizer step`，且无 OOM、traceback、视觉 token mismatch；不保存 adapter。
5. **正式 SFT**：仅在两份 GPU 冒烟日志都通过后提交。继续使用一张 MIG 分区/模型；开始后核验 LoRA 可训练参数比例、loss、checkpoint 与验证生成。

提交前须记录：git commit/dirty diff、配置哈希、数据行数、GPU 类型和分区、命令、job ID、日志位置。若一个任务失败，先停止同批次后续提交，再按根因修复并从第 1 关重新验证。

### GPU 组选择与减少排队规则（2026-08-27）

目标是在不重复占卡、不虚报“马上开始”的前提下，**以最短预计完成时间为第一优先级**。资源成本与最小显存占用为次要指标；只要完整 GPU 当前可启动且能显著缩短任务，就不得仅因 MIG 已能容纳模型而默认选择 MIG。

#### 提交前必须查看实时资源

```bash
show_rsc
pjshowrsc --rscunit rscunit_pg01 --rscgrp
pjstat -E
```

判断 GPU 空闲必须优先使用 `show_rsc` 中各组的 `gpu free/total`，不能只看 `pjshowrsc` 的 node `FREE`，也不能把节点内尚未分配的 GPU resource units 误当作实际可立即调度的 GPU。提交后使用 `pjstat -E JOB_ID` 查看真实预测启动时间。

当前 Genkai 组别含义：

| 资源组 | 资源 | 用途与选择规则 |
|---|---|---|
| `a-batch` / `a-inter` | CPU-only | 不用于 CUDA 训练或推理 |
| `b-batch-mig` | H100 的 1g.12GB MIG | 完整 GPU 无法及时启动，或完整 GPU 不产生明显加速时使用 |
| `b-batch` | 完整 H100 约 96GB | 有空闲时的高速候选；通过更大 batch 和完整算力缩短训练/推理 |
| `c-batch` | 完整 H100 80GB | 有空闲时的高速候选；即使 MIG 能容纳模型，只要预计完成更快也优先使用 |
| `*-inter` | 与对应 batch 共用物理池 | 只用于需要交互终端的短调试，不把正式后台任务伪装成交互任务 |
| `*-reserve` | 预约资源 | 仅在已有合法 reservation 时使用 |
| `d-*` | 当前项目账号不可见 | 不猜测、不提交；以 `pjshowrsc` 可见列表为准 |

#### 自动选择顺序

1. 先用历史 `.stats`、真实冒烟或 `nvidia-smi` 估计各资源组的排队时间加运行时间，不按模型参数量猜速度。
2. 在实际可见的 `b-batch`、`c-batch`、`b-batch-mig` 中选择预计完成时间最短者；完整 GPU 有空闲且可通过更大 batch/完整算力明显加速时，优先完整 GPU。
3. 只有完整 GPU 无空闲、预测排队抵消加速收益，或任务在完整 GPU 上没有明显速度优势时，才选择 `b-batch-mig`。
4. 同等预计完成时间下，再选择资源开销更小、兼容性已经验证的组。
5. 任何切换必须先取消旧的 `QUE` 作业，再提交新作业；禁止用多个资源组同时 race，避免两个副本同时启动。
6. 始终只申请任务需要的 GPU 数；单卡任务固定 `gpu=1`。walltime 根据历史实测收紧，不用过大的“安全余量”降低 backfill 机会。
7. `pjsub` 成功不等于任务开始。提交后必须确认 `ST=RUN`，随后检查真实单图/单步冒烟输出；冒烟通过后才允许同一脚本继续正式工作。
8. 将 job ID、资源组、预测/实际开始时间、退出码和切换原因记录到本文或对应实验报告，并用 tmux 监控当前唯一活动作业。

#### 本次实测案例

- 7B SFT-vs-DPO 47-image × 3-seed 生成任务最初提交到 `b-batch-mig`：作业 `6603466` 仅申请 1 个 MIG，但 `show_rsc` 显示 MIG `0/56` 空闲，`pjstat -E` 预测到 2026-08-28 00:00 才启动。
- 同时检查发现 `b-batch` 为 `0/44` 空闲，而 `c-batch` 为 `3/8` 空闲。经项目负责人明确要求切换后，先取消尚未运行的 `6603466`，再把同一份已通过预检的脚本改为 `c-batch`。
- 新作业 `6603601` 只申请 1 GPU、walltime 1 小时，于 2026-08-27 15:12:34 JST 立即进入 `RUN`。此次切换是时效优先选择，不改变“NF4 7B 实际可在 MIG 上运行”的资源结论，也不把完整 H100 设为默认。

## 本次失败记录

| 时间/任务 | 现象 | 根因 | 已采取修复 | 防复发规则 |
| --- | --- | --- | --- | --- |
| 全量 LoRA 7B/3B 初次 SFT | 第一个 optimizer step 前报 image token count mismatch | 图片视觉 token 在 `max_seq_len` 截断后与 image feature 数不一致 | 在 Qwen 图像消息与 processor 均明确设置 `image_max_pixels=200704`，并添加数据集测试 | 任何视觉配置改动必须先做 CPU collator/token 预检 |
| H100 全量 LoRA 正式提交 | 调度器预测最早 8 月 18 日启动 | 申请的是普通 H100 队列资源，等待来自调度/资源碎片而非训练需要 | 改为冻结 NF4 base 的 QLoRA + 单个 MIG 分区 | 不把 scheduler forecast 表述为“马上开始”；先确认实际资源组与 job 状态 |
| QLoRA MIG 单步任务 `6454795`、`6454796` | 两个任务已获得 MIG 后立即报 `NameError: torch is not defined` | 新增 `BitsAndBytesConfig` 分支引用 `torch.bfloat16`，但 loader 未导入 `torch`；此前只做了语法检查，未走到该分支 | 补充 `import torch`；取消重提交的 `6454803`、`6454804`，先补 CPU 预检 | 新增实际调用 QLoRA loader 分支的单元测试；未通过前禁止 GPU 冒烟 |
| QLoRA MIG 单步任务 `6454813`、`6454814` | 模型权重加载完成后，第一步优化前报 `image_processor.max_pixels` 不存在 | 所用 Qwen processor 版本支持图像消息级 `max_pixels`，但未暴露可写的 processor 属性 | loader 改为：属性存在时同步设置；不存在时继续使用每条图像消息中的预算 | 单测同时覆盖“属性存在”和“属性不存在”的 processor；不把非必要的版本差异当作训练失败 |
| 7B planner 固定生成 step 25/50 | 保存文件只有 `ANCHOR` 一行 | captioner 共用的输出清洗器默认只保留第一行，截掉 planner 的 `CONTRAST/ANGLE`；训练标签、loss 和梯度不受影响 | 为清洗器添加 `preserve_newlines`，planner 配置显式启用，并在记录中保留 `image_id` | 多行结构输出必须有输出后处理回归测试；训练结束后用最终 adapter 重新生成三行验证 |
| 3B captioner 固定生成 step 100 | 8 条样例全部来自同一张图 | caption JSONL 按图像聚集，固定生成直接取前 8 行；validation loss 使用 24 图像的 image-balanced 数据，不受影响 | 固定生成改为按 `image_id` 确定性去重取样 | 人工质检集必须检查图像覆盖数，不能只检查输出条数 |
| v1 compact 标签与 7B 最终验收 | 79 个训练图像中 43 个至少含一条疑似残句；最终 7B 也复现了 `placed`、`an` 等截断结尾 | `compact_text` 对描述硬取前 12 个词，没有保持句法边界 | v2 改为使用 GPT-4o 描述的完整首句；停止旧 3B 作业 `6454887`，旧 adapter 仅留作失败记录 | 数据门禁必须检查字段完整性与生成语义，不得只检查 schema、泄漏和文件可读性 |
| CPU LoRA 门禁脚本 | step 能完成，但报告的 captioner 序列长度与旧版相同 | 门禁构造数据集时未透传 `normalize_prompt: false`，实际使用默认短 prompt | 透传 `normalize_prompt` 与 `sft_prompt`，增加 prompt 等值回归测试并重跑门禁 | 预检必须验证实际 batch prompt 与配置/源 JSONL 一致，不能只验证模型能反传 |
| clean-v2 监控启动 | 直接执行 `scripts/monitor_sft_jobs.sh` 返回 `Permission denied`，且旧脚本写死了 v1 日志文件名 | 脚本没有 executable bit，并将日志前缀绑定到旧作业名 | 统一通过 `sh scripts/monitor_sft_jobs.sh ...` 启动；按 job ID 自动发现日志文件 | 监控脚本必须先 `sh -n`，日志定位不得依赖某一版配置名 |
| 级联验收 CLI 首次本地执行 | 单元测试通过，但直接执行 `verify_sft_generations.py` 报 `ModuleNotFoundError: src` | pytest 从仓库根目录导入时隐式提供了路径，脚本入口没有显式加入项目根目录 | 与其他 CLI 一致，在导入项目模块前加入 `ROOT`；补充真实 subprocess CLI 测试 | 可执行脚本的门禁必须包含真实命令启动，不能用模块级单测代替 |
| 24 图联合推理盲评 | 使用三行 `ANCHOR/CONTRAST/ANGLE` prompt 得到结果后，才发现目标应为旧 v2.5 的 gold-conditioned `hic-compact-json` 路线 | 没有在运行前把 prompt 版本、输入变量和 renderer 与历史 experiment card 逐项核对 | 将原报告标为作废；恢复 `gold-caption-minimal-viewpoint-v2` 原文、逐行校验，并恢复 viewpoint JSON → compact JSON renderer | 联合实验提交前必须记录 prompt 文件 SHA-256、是否读取 gold、输出 schema 和 renderer；prompt 不一致时不得横向比较 |
| compact-viewpoint 7B 作业 `6464195` | 作业正常启动后，用户指出训练输入语义错误 | 把离线造标签所需的 gold caption 误放进了7B SFT输入；这会把planner变成gold-conditioned分析器 | 运行4分钟即取消并释放MIG；改为“图片+全部高分gold captions只用于teacher造共识标签，7B SFT输入只含图片+固定planner prompt” | 数据构造必须分别审计 teacher input、student input 和 target；gold允许进入teacher input，不得进入student input或test inference |
| compact-viewpoint v1 teacher 标签 `6464270` | schema 修复后抽查仍把束缚衣看成手铐、浴缸看成洗浴用品、落马看成手机致摔 | 只依靠生成模型直接看图和 captions，缺少独立视觉描述作为事实约束；结构校验不能发现语义幻觉 | v1 全部移入 `v1_unreliable` 且禁止训练；重新加入数据集 GPT-4o literal/unusual/entity notes 生成 v2 标签 | teacher 标签必须同时经过 schema、caption 泄漏、辅助视觉描述对照和人工语义抽查；通过 schema 不等于可训练 |
| compact-viewpoint v2 teacher 标签 `6464372` | 79+24 个标签已生成，但末尾严格校验因 `nycc_619` 第二个 anchor 缺少 `role` 而退出 | 自回归 JSON 偶发字段遗漏；另有少量 JSON 合法但包含幻觉或 caption 因果的标签 | 保留原始输出，使用显式 override 文件修订 17 train + 3 validation 标签并记录原因；最终 103/103 schema 合法、无非英文 target、6-gram caption 泄漏为 0 | 原始 candidate 永不覆盖；任何语义修改必须记录 image_id、原因、修订前候选，最终文件由可复现脚本生成 |
| compact-viewpoint 7B SFT `6464590`、`6464805` | 两次都在完全相同的 step 31 backward 报 PyTorch `CUDACachingAllocator` NVML internal assert；第二次已禁用训练内生成，排除了生成是根因 | 256 个视觉 token 时已报告峰值 10.35 GiB，固定的更长 batch 很可能触发 `cudaMalloc` 失败；PyTorch 在 MIG 上查询 NVML 失败，使真实 OOM 表现为 internal assert（与官方 PyTorch MIG issue 模式一致） | 视觉预算降到 128 tokens；GPU 冒烟不再取第一行短样本，改为显式使用全数据最长的 `nycc_583`（原 seq 719）做压力测试；训练内生成仍隔离到训练后 | 冒烟必须覆盖最长真实样本并记录峰值；正式作业必须跨过历史故障 step 才能宣称稳定；MIG 上 NVML assert 首先按潜在 OOM 调查 |

## 当前执行状态

- 新 compact-viewpoint 7B 数据：teacher 离线输入为图片、数据集 GPT-4o 视觉说明和该图全部 top-3% gold captions；student SFT 输入严格只有图片和固定 planner prompt，输出为 compact-viewpoint JSON。训练 79 图、验证 24 图、测试 24 图，三者交集均为 0；测试的 4,415 条 caption 仅作为多参考评价池。
- 标签验收：最终 train/validation 共 103 个标签全部通过严格 schema；20 条明确语义错误使用 `data/overrides/` 中的可追溯人工 override 修订；候选文本无非 ASCII 输出、6-gram caption phrase 泄漏为 0。student JSONL SHA-256 为 train `146db3275c9b43c6c9c010eb3ce4c5263f89c2475c4dea7231eff5bd3b64dc48`、validation `c963909bccafaf48ee3e605e836606e8b054713cc8cdc932f7a925ed66ccf3a9`，prompt SHA-256 为 `2b5bf47892a606f395a7e21152101d185b427c1cebda9608d1819a475154204a`。
- 新 7B 提交前门禁：全量测试 `41 passed`；CPU 使用真实 Qwen2.5-VL-7B 权重完成一次 LoRA forward/backward/optimizer step，`loss=0.809460`、`seq=666`、`visual_tokens=240`、可训练 LoRA 参数 `5,046,272`，耗时 317.1 秒。正式配置为 NF4 QLoRA rank 8，不是全量微调。
- 新 7B SFT 尝试：`6464590` 与禁用训练内生成后的 `6464805` 都在 step 31 遇到相同 NVML allocator assert 并自行退出，均不能视为完成；这反证长生成不是根因。当前修复把 `image_max_pixels` 从 200,704 降至 100,352（视觉 token 上限约从 256 降至 128），并把 GPU 冒烟样本从第一行改为原数据中最长的 `nycc_583`。等待最长样本压力冒烟与跨 step-31 验证后再更新完成状态。
- 降显存修复门禁：processor 全量扫描后，最长样本 `nycc_583` 从 `seq=719 / visual=238 / supervised=207` 降为 `seq=601 / visual=120 / supervised=207`，79 图最大视觉 token 为 126，证明未裁剪监督标签；完整测试更新为 `42 passed`。待提交作业会用 `nycc_583` 做 one-step 压力冒烟，随后必须越过历史故障点 step 31。

- 数据：clean-v2 使用完整首句构造 `ANCHOR/CONTRAST`；planner 为 79/24/24 张图，caption 为 13,190/3,990/4,415 行；残句审计、prompt 对齐、切分隔离和全量 processor 扫描均通过。
- 独立数据审计：`scripts/audit_newyorker_compact_sft.py` 不调用数据构建器，直接从 pinned revision `1cd70477b6a99a473690a25a2fed359f75184c64` 的原始 ranking CSV 重算 721,955 条有效记录，证明 21,595 个目标恰好是每幅漫画各自的 top 3%；127 张图片均可解码、split 图像无交叉、gold-caption prompt 泄漏为 0。带各 JSONL SHA-256 的报告保存在 `outputs/newyorker_compact_v2_data_audit.json`，对应回归测试已纳入全量 `pytest`（当前 29 passed）。
- 训练方式：QLoRA（NF4 冻结基座 + LoRA adapter），不是全参数微调；7B/3B 可训练参数分别为 0.1074%/0.3612%。
- GPU 门禁：clean-v2 单步 MIG 冒烟 `6455335`、`6455336` 均通过；峰值显存分别为 9.492 GB、4.292 GB，无 OOM、NaN、Traceback 或视觉 token 错配。
- 正式 SFT：`6455356`（7B）与 `6455357`（3B）均以 exit code 0 完成，每个只占一个 MIG；20 分钟监控日志写入 `outputs/newyorker_compact_v2_monitor/monitor.log`。
- 完成门禁：`submit_pipeline_eval_after_sft.sh` 检测到两个 final-adapter 标记后，确认四个 best/final adapter 均通过有限值/LoRA 配对验证，再提交唯一一个 MIG 的 best→best 端到端测试 `6455911`；该测试以 exit code 0 完成，提交及预检记录位于 `outputs/newyorker_compact_v2_pipeline_gate/`。
- 本阶段仍不提交 DPO；正式训练完成后先验证 final/best adapter、有限张量、验证指标和跨图生成。

## 实测资源预算

clean-v2 正式任务的 Trainer 总步数为 planner 300、captioner 800。启动后的稳态约为 7B 14.4–15.0 秒/step、3B 15.0–15.6 秒/step，对应终端初始 ETA 约 1 小时 10 分与 3 小时 17 分；加上周期验证和最终生成，PJM 时限分别设为 4 h 和 6 h。两者各只申请一个 12 GB MIG 分区，每 2 小时保存一次可恢复的 LoRA adapter。

## 方法依据

- LoRA 只训练低秩增量参数，显著降低可训练参数和优化器状态开销：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), https://arxiv.org/abs/2106.09685。
- QLoRA 以 4-bit NF4 量化冻结基座、在其上训练 LoRA adapter，是小显存后训练的直接依据：Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023), https://arxiv.org/abs/2305.14314。
- New Yorker caption ranking 数据集及其视觉幽默任务定义：Hessel et al., *The New Yorker Caption Contest Dataset* (NeurIPS 2024), https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html。
