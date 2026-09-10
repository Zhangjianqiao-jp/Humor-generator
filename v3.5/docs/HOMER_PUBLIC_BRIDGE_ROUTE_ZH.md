# 当前 HOMER public-code + latent bridge 路线（v3.5）

更新时间：2026-09-10

本文只描述当前的 **adapter-free Qwen2.5-VL-7B + bridge-only** 路线。历史
`latent_bridge_v35`、A4/A5 trace、SFT adapter 和旧 caption trainer 不得作为当前
路线的输入，也不得用来支撑“当前 HOMER + latent bridge”的结论。

## 1. 路线定义

当前路线的可复核命名是：

```text
HOMER public-code stage order
+ Qwen2.5-VL-7B-Instruct@cc594898...
+ Planner/Generator adapter=null, frozen
+ project-specific latent bridge (only trainable component)
```

这不是 HOMER 论文的 100% 权重复现：官方公开实现使用 `gpt-4o`，而本路线使用
本地固定 revision 的 Qwen；HOMER 本身也没有 latent bridge。latent 结果只能作为
明确命名的扩展实验。

## 2. 已修复的旧偏差

旧 bridge trainer 会把历史 prompt、历史 trace 和 `latent_bridge_v35` 数据带入
caption 阶段，因而缺失官方 caption 协议和 Generator 的后处理上下文。当前代码通过
独立 prompt track 解决：

| 层 | 当前实现 | 硬门禁 |
|---|---|---|
| Planner channel | `cache_pretrained_homer_traces.py` 使用官方 conflict/global/local prompt，并保存 raw response 与 hidden-state trace | 362/362、revision、adapter=null、token/hidden 对齐、raw-response hash |
| Summary | `cache_pretrained_homer_context.py` 调用官方 summary prompt | JSON 可解析且 context provenance 完整 |
| Retrieval | `OfficialCodeRetrieval`，保留官方 TF-IDF/WordNet/entity-tree 可观察行为 | 官方 joke corpus hash 必须匹配 |
| Selection | 独立的 conflict selection、entity selection，再按固定 seed 做 DFS path selection | selected conflict、两个 entity、两条 path 均可回放 |
| Caption teacher | 官方 `caption_messages(description, selected_conflict, free_association)` | 保留官方 system prompt 与两个 user content blocks |
| Latent student | 同一官方 caption request，但只移除 plan text；bridge memory 替代 plan | 不能再使用 generic/legacy caption prompt |
| Receiver | 当前 public config 采用 receiver-driven cross-attention；Planner/Generator 冻结 | 仅 bridge 参数可训练 |

当前路线的正式入口是：

```text
configs/pilot/cross_attention_caption_pretrained_public.yaml
```

旧的 `configs/pilot/cross_attention_caption_pretrained.yaml` 仍保留用于历史审计，
但不得用于当前路线。

## 3. 为什么 context cache 是必要的

bridge 不能只拿 `conflict/local/global` 三个解析后的字符串就声称复现 HOMER。官方
Generator 还依赖 summary、retrieval、选择出的 conflict、两个 entity 以及每个 entity
的一条 DFS 联想路径。因此 context cache 保存：

```text
summary_raw / summary_imagination
retrieved_imagination
selection_raw[conflict, entities]
selected_conflict
selected_entities
selected_paths
free_association_text
call_log
trace_path / trace_sha256
planner_outputs_sha256
prompt/model/population/input-manifest hashes
```

text teacher 使用 selected context 的完整文本；latent student 使用同一 description 和
同一官方 caption prompt，只把 selected plan block 替换为 bridge memory。这样二者的
差别是通信通道，而不是偷偷换了上游规划结果。

## 4. `##Caption` / `##Explanation` 目标策略

官方 caption prompt 和生成入口要求并解析：

```text
##Caption:
...
##Explanation:
...
```

当前公开 ranking/source rows 只有 gold caption，没有人工 gold explanation。因此训练
target 明确记录为：

```text
source_caption_only_no_fabricated_explanation
```

不会伪造 explanation，也不会把空 explanation 当作监督标签。训练阶段两种条件使用
完全相同的 source caption token；在线生成阶段保留官方 marker，并通过
`parse_caption_blob` 分离 caption 与 explanation。这个限制必须写进结果 provenance，不能
把 source-caption-only bridge training 描述成“使用了 gold explanation”。

## 5. 执行顺序（fail-closed）

```text
1. cache_pretrained_homer_traces.py
2. verify_pretrained_homer_traces.py       # 必须 362/362
3. cache_pretrained_homer_context.py       # summary/retrieval/selection/path
4. build_pretrained_bridge_dataset.py      # 仅建立 bridge training view
5. verify_pretrained_bridge_inputs.py      # hash/row/context/frozen gate
6. train_bridge.py                         # 仅训练 bridge
7. 独立运行 public-code Text-HOMER 与 latent caption generation
8. 按 HOMER primary 与 project extension 两条评测轨道分别报告
```

在第 2 步失败时，不得运行第 3 步；在第 5 步失败时，不得加载模型训练。任何
`latent_bridge_v35` 字符串、旧 adapter、旧 trace 或 context/hash 不一致都会被拒绝。

当前正式作业 `6754067` 不再生成 trace，而是消费已经通过 362/362 trace/context gate 的
sealed artifacts，仅负责 bridge-only scientific training。该作业已正常完成并生成完整
checkpoint；因此现在只允许进入严格的 caption generation gate。不得把 near-zero
matched/shuffled gap 当作 latent 因果使用证据，必须等待 test caption 与 HOMER 评测结果。

## 6. 结果命名边界

可以报告：

```text
adapted pretrained-7B public-code baseline
adapted pretrained-7B + latent bridge
```

不能报告：

```text
100% HOMER reproduction
official HOMER latent bridge
canonical 365-contest HOMER result
```

362 公共 release、canonical 365、HIA 原始 Group Overall/Best Pick、HOMER GPT-5
Pass@K 和项目 Group-of-10/multi-judge extension 必须分开统计，不能混成一个数字。

## 7. 权威依据

1. Shang et al., *On the Wings of Imagination: Conflicting Script-based Multi-role
   Framework for Humor Caption Generation*, ICLR 2026：
   [论文](https://arxiv.org/html/2602.06423)；[官方实现](https://github.com/Shang-hub/HOMER-Official-Implementation)。
2. Zhang et al., *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks
   for Cartoon Captioning*, NeurIPS 2024：
   [论文与数据](https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf)。
3. Bai et al., *Qwen2.5-VL Technical Report*：
   [论文](https://arxiv.org/abs/2502.13923)。
4. Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image
   Encoders and Large Language Models*, ICML 2023：
   [论文](https://proceedings.mlr.press/v202/li23q)。
