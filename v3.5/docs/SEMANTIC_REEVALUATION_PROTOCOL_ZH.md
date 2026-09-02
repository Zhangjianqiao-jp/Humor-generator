# v3.5 旧语义 Bridge 修正重评协议

## 状态

本文件记录对旧 `v1`/`v2` semantic bridge 的新 evaluator 审查结果和修正后的正式协议。
它不是 caption 质量报告，也不把尚未完成的 GPU 作业写成实验结论。

- 审查日期：2026-09-03（JST）
- 修正 evaluator：`scripts/re_evaluate_failed_semantic_bridges.py`
- A3 smoke：`6689653`，当时状态为 `QUE`
- 旧 bridge 重评：`6695787`，当时状态为 `QUE`
- 当前修正代码已通过 Python 编译、帮助命令和相关 CPU 单元测试；GPU 结果待作业完成。

## 1. 重评的科学问题和边界

旧实验的 `no_go` 只说明它没有通过当时的操作性 semantic gate，不能直接说明
“latent communication 这一类方法无效”。修正重评只回答：

> 在冻结的旧 checkpoint 下，Receiver 的 semantic reconstruction likelihood 是否会因
> 正确的 Planner memory 被替换而下降，而且这个下降是否不是 memory 长度或全局 softmax
> 分母变化造成的？

本重评明确排除：

- 重新训练 bridge；
- 重新生成 caption；
- 使用 sealed test 或 outer-validation 质量结论；
- 重新验证已解决的 Hint usefulness；
- 根据 24 个 cluster 直接宣布方法级成功或失败。

因此输出的 `strong_go` 只表示“值得进入更大 semantic validation”，不表示
“caption 已经优于 text”。

## 2. 代码与输入审计

| 项目 | 具体实现 | 审查结论 |
|---|---|---|
| v1 architecture | `src/humor_generator_v35/latent/legacy_cross_attention.py::LegacyV1ReceiverDrivenCrossAttentionBridge` | 保留旧的单一 concatenated-memory softmax；strict state load 通过 |
| v2 architecture | `src/humor_generator_v35/latent/cross_attention.py::ReceiverDrivenCrossAttentionBridge(channel_fusion="learned")` | 按旧 checkpoint 的 hierarchical/channel gate 加载；strict state load 通过 |
| Receiver | `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898...cfb5` | 与 `manifests/local_qwen2_5_vl_7b.json` 逐项核对 |
| Generator adapter | `artifacts/checkpoints/generator_sft` | 与 `manifests/frozen_7b_adapters.json` 核对；policy 参数必须为 0 |
| target data | `data/processed/latent_bridge_v35/validation.jsonl` | 固定 hash 抽取 24 个 cluster，每 cluster 选一个代表 row |
| donor data | `data/processed/latent_bridge_v35/train.jsonl` | 与 target cluster 不重叠；排除该 checkpoint 自己拟合过的 train clusters |
| Planner memory | `data/cache/planner_traces_homer_strict_v35/index.jsonl` 及其 hash-checked `.pt` | `conflict/local/global` 三路完整、teacher-forced post-token 对齐 |
| Receiver interface | `ReceiverCrossAttentionTask.prepare` 与 `_logits` | 图像、semantic-recovery prompt、target 固定，只替换 memory |
| statistics | `summarize` | image-cluster 单位；10,000 次 cluster bootstrap；Wilson proportion interval |

正式运行前仍执行：

```text
check_environment.py
check_v35_isolation.py
verify_frozen_artifacts.py
verify_clustered_dataset.py
check_trace_completion.py
```

## 3. 发现的原 evaluator 混淆

旧重评版本对每个 validation cluster 选择另一个 validation cluster 作为 donor，只按
description 相似和 conflict 不同来选，没有控制被替换 channel 的 token 数。

这对 v1 尤其危险。v1 将三个 channel 拼接后做一个 softmax：

\[
\alpha_{t,j}=\operatorname{softmax}_j(q_t^\top k_j/\sqrt d).
\]

替换一个长度为 \(T_c\) 的 channel 为长度不同的 donor，会同时改变 softmax 分母，
即便 donor 的语义完全无关，也会改变所有位置的 attention mass。此前固定 24-cluster
选择中的 channel 长度差最大达到约：

```text
conflict 41 tokens, local 85 tokens, global 163 tokens
```

所以原来的 `gap` 不能被解释为纯粹的 semantic sensitivity。这是一个 evaluation/
confounding 问题，不是旧模型已经被证明无效。

## 4. 修正后的反事实构造

对 target cluster \(i\)，令完整 memory 为：

\[
Z_i=(Z_i^C,Z_i^L,Z_i^G).
\]

对 channel \(c\)，选择 donor \(d_i^c\)，形成：

\[
Z_i^{(c\leftarrow d_i^c)}
 = (Z_i^{-c},Z_{d_i^c}^c).
\]

被评分的 image \(x_i\)、semantic target \(s_i\)、chat prompt 以及其余两个 channel
全部保持不变。只允许 channel \(c\) 变化。

### Donor 选择优先级

函数 `length_matched_channel_donors` 按以下顺序执行：

1. donor 必须来自非 test 的 `train` split；
2. donor 与 target 不得是同一个 cluster；
3. conflict signature 必须不同，避免“换了一个几乎相同的 plan”；
4. **先最小化当前 channel 的绝对 token-length difference**；
5. 在最小长度差集合内优先同数据源；
6. 最后最大化 standard-description TF-IDF 相似度。

此外，checkpoint 的 `run_manifest.json` 中已经用于拟合该 bridge 的 train clusters 不作为
donor，避免 evaluator 依赖模型的 in-sample memory。

固定 pilot 的 donor 选择 smoke 统计为：

| channel | exact-length fraction | mean absolute length delta |
|---|---:|---:|
| conflict | 0.9583 | 0.0417 |
| local | 0.9167 | 0.1250 |
| global | 0.7917 | 0.2083 |

残余差异不会静默 padding、truncation 或伪造 exact match；逐 cluster 的
`target_length_*`、`donor_length_*`、`length_delta_*` 和 gap-length Pearson correlation
均写入 `cluster_counterfactuals.jsonl`/`summary.json`。

### Full-plan swap 的定位

完整 plan swap 仍保留，但只作为 secondary stress test。它不控制每个 channel 长度，
因此不参与主 channel gate，也不能单独支持 semantic claim。

## 5. 评分量和统计

对 semantic target 的 token-average log probability 定义：

\[
\ell_\theta(s_i\mid x_i,Z)
 = \frac{1}{|s_i|}\sum_{t=1}^{|s_i|}
 \log p_\theta(s_{i,t}\mid s_{i,<t},x_i,Z).
\]

主 channel gap 为：

\[
\Delta_{i,c}
 = \ell_\theta(s_i\mid x_i,Z_i)
 - \ell_\theta(s_i\mid x_i,Z_i^{(c\leftarrow d_i^c)}).
\]

\(\Delta_{i,c}>0\) 表示 Receiver 更偏好与当前 image/plan 配对的 channel；它仍然
不是“semantic reconstruction 正确率”，也不是“caption 更好笑”的替代指标。

统计规则：

- 统计单位是一张 image cluster，而非 caption row 或 token；
- 每个 cluster 只保留一个 deterministic representative；
- mean/median gap、`fraction_gap_gt_0`、`fraction_gap_gt_0.02`；
- mean gap 使用 10,000 次 image-cluster bootstrap 95% CI；
- 比例使用 Wilson 95% interval；
- 同时报告 gap 与绝对 length delta 的 Pearson correlation，作为残余混淆诊断；
- selection seed、bootstrap seed 都显式写入 summary。

24 个 cluster 仍是机制 pilot。它不能有力地区分小的正负效应；因此结果状态严格为：

```text
strong_go                         point gate pass + all channel bootstrap lower > 0
go_to_outer_semantic_validation   point gate pass, but CI is not uniformly positive
pilot_inconclusive                engineering valid, but point gate fails
hard_no_go                        only real engineering/invariant failure
```

`pilot_inconclusive` 不得改写为“方法无效”。只有在更大的、预注册的 outer semantic
validation 上仍出现稳定负向 effect，才可作方法级否定。

## 6. Provenance 与可复现性

每个 method 输出：

```text
cluster_counterfactuals.jsonl
summary.json
manifest.json
```

其中保存：

- checkpoint bytes SHA-256；
- checkpoint `run_manifest.json` SHA-256；
- config SHA-256；
- dataset manifest 与 trace-input manifest SHA-256；
- trace index SHA-256；
- frozen adapter manifest SHA-256；
- local Qwen model manifest SHA-256；
- HOMER prompt source 与 Receiver prompt source SHA-256；
- evaluator script SHA-256；
- 当前代码 Git commit；
- target cluster IDs 及其 hash；
- donor policy、donor pool 排除数量、逐 channel donor IDs 和长度信息。

输出目录拒绝覆盖已有目录；作业中途失败后必须使用新 output 目录，不能混合旧的
partial artifact。

## 7. 当前结论的写法

在 GPU 作业结束前，只能写：

> v1/v2 已进入冻结 checkpoint 的修正反事实重评；旧 `no_go` 暂不能升级为方法级
> 无效结论。新的 evaluator 已消除主要的 v1 channel-length/softmax-denominator 混淆，
> 结果需等待 24-cluster summary 和 bootstrap CI。

作业结束后：

1. 若出现 `hard_no_go`，先检查工程 invariant 和 artifact，不作语义结论；
2. 若为 `pilot_inconclusive`，进入预注册 outer semantic validation；
3. 若为 `go_to_outer_semantic_validation`，进入 outer validation，不直接训练 caption bridge；
4. 若为 `strong_go`，仍需在未参与选择的 outer 图片上验证 downstream caption utility；
5. 只有 downstream quality 和 grounding 都不退化，才讨论 latent bridge 的 preference learning。

## 8. 与文献的一致性和限制

- 配对/cluster bootstrap 适合把 image cluster 作为自然统计单位，而不是把同图的多个
  token 或 caption 当成独立样本；参见 Koehn 的 paired bootstrap 方法。
- 24-cluster pilot 具有明显 Type-II 风险，point failure 不能直接作为方法否定；参见
  Graham 等关于 NLP 评测统计功效的分析。
- 生成文本的人类/模型评分方差较大，后续 caption 评测必须保留 image-level 分层和
  evaluator agreement；参见 Howcroft 与 Rieser 对 ordinal ratings 的讨论。
- 错配/信息破坏只能作为 latent 使用的机制证据，不能替代下游任务收益；这一边界与
  InterLat 对 task-mismatched latent 的分析一致。

## 权威参考文献

1. Koehn, *Statistical Significance Tests for Machine Translation Evaluation*, EMNLP 2004：
   <https://aclanthology.org/W04-3250/>
2. Graham, Haddow & Koehn, *Statistical Power and Translationese in Machine Translation
   Evaluation*, EMNLP 2020：<https://aclanthology.org/2020.emnlp-main.6/>
3. Howcroft & Rieser, *What happens if you treat ordinal ratings as interval data?*, EMNLP 2021：
   <https://aclanthology.org/2021.emnlp-main.703/>
4. Dror et al., *The Hitchhiker's Guide to Testing Statistical Significance in NLP*, ACL 2018：
   <https://aclanthology.org/P18-1128/>
5. Du et al., *InterLat: An Interpretable Latent Communication Framework*, ACL 2026：
   <https://aclanthology.org/2026.acl-long.1248/>
6. Shang et al., *HOMER*, ICLR 2026：
   <https://openreview.net/pdf?id=SzaRhPom4o>
