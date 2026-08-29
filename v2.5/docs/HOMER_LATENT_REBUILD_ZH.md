# HOMER 结构化 Planner 与双通道 Latent Communication 重构

更新时间：2026-08-30

## 1. 研究目标与边界

本轮把旧的单次 compact plan 改为 HOMER 的阶段化流程，并只将 `Conflict Scripts` 与 `Associative Imagination` 作为连续表示发送给 Generator。`Grounding / Situation Description` 保持文本形式，便于审计视觉事实。Planner 与 Generator 的 7B SFT checkpoint 均冻结；第一阶段只训练 bridge。

不能把本实现直接写成“完整复现 HOMER”。HOMER 还包含一个由 11 个笑话数据集清洗出的 335,570 条 joke retrieval corpus、WordNet/Wu–Palmer 相似度与 pruning。当前主线先严格复现其 staged prompts 和 local/global association contract；retrieval 与文化知识只作为后续可配置消融，未启用时必须写成 `HOMER-style staged planner`。

## 2. Planner 方案

严格保留以下依赖顺序：

1. `Situation Description / Grounding`：图像输入，描述地点、人物、表情、动作、对象、关系以及可见异常；
2. `Conflict Scripts`：以 Grounding 为输入，按 GTVH Script Opposition 生成至少两组冲突脚本；
3. `Associative Imagination — Local`：以 Grounding 和 Conflict 为输入，对细粒度对象生成三跳关联链；
4. `Associative Imagination — Global`：以原图和 Conflict 为输入，对全局主要对象生成三跳关联链；
5. Caption Generator：基于 description、conflict 和 local/global chains 生成简短 caption。

论文附录 prompt 的代码位置：

- `src/latent_communication/homer.py`
- `prompts/homer_description.txt`
- `prompts/homer_conflict_system.txt`
- `prompts/homer_imagination_local_system.txt`
- `prompts/homer_imagination_global_system.txt`

旧 compact JSON 不再是新主实验的 Planner contract，但保留旧文件以复现实验。

## 3. Latent 方案

新条件形式为：

```text
Generator(image,
          grounding_text,
          typed_conflict_latents,
          typed_association_latents)
```

Conflict 与 Association 使用两个独立 query-resampler，并加入不同的 learned type embedding；输出再拼接到 Generator 输入。这样可以避免旧方案把整段 plan 的末尾 64 token 压到一个无类型的 16-slot prefix。

配置把 sender state 上限提高到 256，并采用从 Hybrid scaffold 到纯 latent 的 curriculum。InterLat 的核心提醒是通信表征需要被接收器显式学会，不能把任意 hidden states 一次强压缩后期待冻结接收器自然理解；StateBridge 则支持从发送方内部状态建立紧凑连续桥接。当前实现因此先训练可解释的字段化 bridge，Planner 与 Generator 不解冻。

## 4. 文化注入

文化注入默认关闭，只作为 ablation：

```yaml
culture_injection:
  enabled: false
  mode: retrieval_context
  ablation_only: true
```

若启用，应检索与图中实体、冲突脚本相关的文化脚本/习语/职业规范，并分别报告文化相关子集和普通子集。不能把文化知识直接混入 Grounding，否则会使可见事实与外部知识无法区分。

## 5. 数据扩充与真实计数

已使用公开官方资源：

- Humor in AI：365 rows；
- Electronic Sheep：679 rows；
- 合计：1,044 dataset rows。

但按 NYCC contest ID 去重后只有 **819 张独立漫画**，跨数据源重复 **225 rows**。训练、验证、测试按 contest ID 聚类，重复 JPEG 编码也不会跨 split。当前 split 为：655 / 81 / 83 independent image clusters；对应 836 / 104 / 104 rows。

因此必须区分：

```text
1,044 benchmark rows != 1,044 independent images
```

不能同时使用这些图训练 bridge，又声称在其中 1000+ 张独立 held-out 图上测试。正式 1000+ independent held-out evaluation 暂不满足数据条件；需要新增至少约 1,000 张同分布、未参与 Planner/Generator/Bridge SFT 的公开漫画。若只按 HOMER 两数据集分别评估，应报告跨数据源重复，并用 contest-cluster bootstrap，不能把重复项当独立样本缩窄置信区间。

## 6. 评测协议

计划比较：

- 冻结 Qwen2.5-VL-7B-Instruct base Generator；
- 之前的 7B SFT Generator；
- 相同在线 HOMER Planner trace；
- 每系统每图 10 candidates；
- 至少 3 个 generation seeds；
- 匿名并随机化 system/order；
- Group-of-3 用于可读盲评，同时保留 10-candidate pool 的 pass@k / best-of-10；
- 以 independent contest/image 为统计单位，报告 win rate、cluster-bootstrap 95% CI、seed variance；
- 绝对标签 `good / weak / bad`，并报告 grounding、humor、originality、specificity 与 hallucination。

Group-of-3 不是 HOMER 原论文全部评测的替代。HOMER 还报告 1–5 人评、pass@k、3-gram coverage 与 NLI diversity；Humor in AI 的生成评测使用排名式 group benchmark。论文级报告应并列这些指标，而不能只报单一 LLM judge 的相对胜率。

## 7. 执行门禁

1. 静态检查与单测；
2. 单 MIG、2 train rows / 1 validation row 的真实端到端 smoke；
3. smoke 成功后，根据实时 `show_rsc` 选择预计完成最快的单 GPU；
4. 正式 bridge training，保存缓存、latest/best checkpoint、数据和配置 hash；
5. 只在 held-out validation 改善后生成盲评候选；
6. 1000+ independent held-out 数据门槛未满足时，禁止把 1,044 rows 写成 1,044 independent images。

当前 smoke job：`6638166`。

`6638166` 首次 smoke 在 staged Planner 图像前向触发 MIG allocator/NVML internal assert。审计发现新入口没有把旧流水线已验证的 `100352 max_pixels` 传到 Grounding 与 Global-Imagination 图像消息；已把该预算变成显式配置，正式训练仍被 smoke gate 阻断，待修复后的重新冒烟通过。

修复后 smoke `6638172` 已完成四阶段 Planner 并进入 Generator backward，说明图像预算修复有效；但 48 latent slots、512 bottleneck 的双 bridge 在 12GB MIG 上仍触发临界 allocator failure。为保持与旧已验证 16-slot 总预算可比，配置收紧为每通道 8 slots（总计 16）与 256 bottleneck；字段仍使用独立 bridge 和 type embedding，不合并语义通道。

收紧后的 `6638176` 仍在冻结 Generator 的 backward-through-input 路径触发相同 allocator failure；此时 bridge 约 5.4M 参数且 latent 总 slots 已与旧方案相同，继续削弱通道会改变研究问题。按执行手册，三次 MIG 证据已证明 12GB 分区不适合该双通道训练路径，后续 smoke 改为单张 `c-batch` H100、45 分钟，不申请多卡。

## 8. 权威参考

1. Zhang et al. *On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation (HOMER).* ICLR 2026. arXiv:2602.06423.
2. Sui et al. *InterLat: Interleaved Latent Communication for Multi-Agent LLM Systems.* ACL 2026, Long Papers, 1248.
3. *StateBridge: Enhancing Multimodal Reasoning through Latent State Communication.* COLM 2026. arXiv:2608.13317.
4. Hao et al. *Training Large Language Models to Reason in a Continuous Latent Space (Coconut).* ICLR 2025.
5. Zhang et al. *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning.* NeurIPS 2024 Datasets and Benchmarks.
6. Hessel et al. *Do Androids Laugh at Electric Sheep? Humor “Understanding” Benchmarks from The New Yorker Caption Contest.* ACL 2023 Best Paper.
7. Attardo and Raskin. *Script Theory Revis(it)ed: Joke Similarity and Joke Representation Model.* HUMOR, 1991.（GTVH / script opposition 理论基础）
