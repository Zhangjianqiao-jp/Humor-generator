# v3.5 旧 Semantic Bridge 修正重评结果

## 结论摘要

修正后的 evaluator 已成功完成 v1 和 v2 冻结 checkpoint 的 24-image-cluster
反事实重评。结果不是工程失败，但两个方法都没有通过“三个 channel 同时被稳定使用”的
pilot gate：

```text
v1 status = pilot_inconclusive
v2 status = pilot_inconclusive
```

这意味着：

1. 原先由于错误 donor 长度造成的 v1 softmax 混淆已被实质降低；
2. v1/v2 都没有显示 conflict、local、global 三路一致且有实际幅度的 semantic use；
3. v1 的 global channel 出现很小但 bootstrap 下界略高于 0 的 gap，不能代表整体成功；
4. v2 的 conflict/global 只有方向性点估计，CI 均跨 0，local 仍为负；
5. 两者都不能被写成“latent communication 已被证伪”，因为 24 clusters 只具备机制
   pilot 的统计功效；
6. 在 outer semantic validation 和 caption-level evaluation 完成前，不允许进入 caption
   bridge 或 preference learning。

## 1. 运行和 provenance

### v1

- checkpoint：`outputs/pilot/cross_attention_semantic_v1/best_bridge.pt`
- checkpoint SHA-256：`38eab8cf3d2d1e6b875289d1251b08234e5667f61e79ed3f36ec8860b45f9fe8`
- output：`outputs/re_evaluation/semantic_v1_corrected/`
- summary：`outputs/re_evaluation/semantic_v1_corrected/summary.json`
- details：`outputs/re_evaluation/semantic_v1_corrected/cluster_counterfactuals.jsonl`

### v2

- checkpoint：`outputs/pilot/hierarchical_cross_attention_semantic_v2/best_bridge.pt`
- checkpoint SHA-256：`dc160213439a1db13856625a068366cb7d9b6a59c80636c5223ac977ae24b5b9`
- output：`outputs/re_evaluation/semantic_v2_corrected/`
- summary：`outputs/re_evaluation/semantic_v2_corrected/summary.json`
- details：`outputs/re_evaluation/semantic_v2_corrected/cluster_counterfactuals.jsonl`

两个 summary 均记录：

- Qwen revision、local model manifest、generator adapter manifest；
- checkpoint/config/dataset/trace/prompt/evaluator SHA-256；
- current code commit：`f6ab9ab7c69eb65454c792a2d4120dec7f489f84`；
- selection/bootstrap seed：`20260830`；
- 24 个 validation cluster ID 及 hash；
- donor pool 与被排除的 checkpoint-fit clusters。

## 2. 修正 evaluator 是否真正运行成功

成功完成的检查：

- `check_environment.py`：pass；
- `check_v35_isolation.py`：pass；
- frozen adapter byte verification：pass；
- clustered dataset：2,846 rows、949 images、810 clusters 审计通过；
- formal traces：666/666、缺失 0、无效 0；
- v1/v2 old bridge strict state-dict load：pass；
- 24 个 representative clusters × 两个 method：全部完成；
- 输出 `summary.json` 与 `manifest.json` 均存在。

因此这次重评不是 CUDA/OOM 或数据缺失失败。重评作业 `6695787` 在 GPU 上完成，
未使用 sealed test、未生成 caption、未更新任何模型参数。

## 3. 反事实定义

令匹配 memory 为：

\[
Z_i=(Z_i^C,Z_i^L,Z_i^G),
\]

目标是同一 image \(x_i\) 下的 structured semantic target \(s_i\)。定义 token-average
log probability：

\[
\ell_i(Z)=\frac{1}{|s_i|}\sum_t
\log p_\theta(s_{i,t}\mid s_{i,<t},x_i,Z).
\]

对某个 channel \(c\)，只替换该 channel：

\[
Z_i^{(c\leftarrow d_i^c)}
 = (Z_i^{-c},Z_{d_i^c}^{c}),
\]

并计算：

\[
\Delta_{i,c}=\ell_i(Z_i)-\ell_i(Z_i^{(c\leftarrow d_i^c)}).
\]

正值表示 Receiver 对匹配 channel 的 target likelihood 更高；负值表示 donor channel
反而更有利，或该 channel 没有稳定的可识别作用。

严格保持不变的量：

```text
image
semantic-recovery prompt
semantic target
teacher-forced target tokens
the other two channels
frozen Qwen receiver and SFT adapter
```

完整 plan swap 也被计算，但只作 secondary stress test，因为它不控制三路长度，不能作为
主 channel gate。

## 4. Donor 长度控制审查

原 evaluator 的 donor 可能改变 v1 单一 softmax 的分母。修正版从 train split 选择 donor，
排除该 checkpoint 拟合过的 64 个 train clusters，并按以下优先级：

1. conflict signature 不同；
2. 当前 channel 的 token-length difference 最小；
3. 在最小长度差集合内优先同数据源；
4. description TF-IDF 相似度最大。

538 个 donor clusters 与 24 个目标 clusters 不重叠。实际重评中的长度统计：

| method | channel | exact-length fraction | mean absolute delta | max absolute delta |
|---|---|---:|---:|---:|
| v1/v2 | conflict | 0.9583 | 0.0417 | 1 |
| v1/v2 | local | 0.9167 | 0.1250 | 2 |
| v1/v2 | global | 0.7917 | 0.2083 | 1 |

残余长度差很小且全部显式保留。gap 与绝对 length delta 的相关性为：

| method | conflict | local | global |
|---|---:|---:|---:|
| v1 | 0.290 | 0.235 | 0.006 |
| v2 | 0.052 | 0.079 | -0.159 |

这些相关性不是正式 causal proof，但说明本次 observed global gap 不能简单归因于 donor
长度；v1 conflict/local 的中等相关性仍要求后续 outer validation 做 exact-length sensitivity
analysis。

## 5. v1 结果

| channel | mean gap | median gap | fraction > 0 | Wilson 95% CI | mean-gap bootstrap 95% CI |
|---|---:|---:|---:|---|---|
| conflict | -0.000098 | -0.000676 | 10/24 = 0.4167 | [0.2447, 0.6117] | [-0.000883, 0.000694] |
| local | -0.000525 | -0.000193 | 12/24 = 0.5000 | [0.3143, 0.6857] | [-0.001507, 0.000438] |
| global | 0.001170 | 0.001434 | 17/24 = 0.7083 | [0.5083, 0.8509] | [0.000056, 0.002230] |

其他指标：

- matched semantic logp mean：`-0.631934`；
- mean gate：`0.107391`；
- mean relative update norm：`0.059995`；
- 三个 channel 的 `fraction gap > 0.02` 均为 `0/24`；
- full-plan secondary gap：`0.005174`，bootstrap CI `[0.002586, 0.007698]`。

### v1 解释

v1 主要呈现“global 有微弱正向迹象，conflict/local 没有稳定正向迹象”。global 的
bootstrap 下界虽略高于 0，但效应约为 `0.00117` nats/token，远低于预注册的 `0.02`
操作性阈值，而且另外两路不通过。因此不能称为三通道 semantic recovery 成功。

full-plan gap 不能推翻这个结论：完整 memory 替换同时改变三路内容和旧 v1 的全局归一化，
它只能说明某种整体 memory 改变会影响输出，不能说明 conflict/local/global 各自被使用。

## 6. v2 结果

| channel | mean gap | median gap | fraction > 0 | Wilson 95% CI | mean-gap bootstrap 95% CI |
|---|---:|---:|---:|---|---|
| conflict | 0.000187 | 0.000289 | 16/24 = 0.6667 | [0.4671, 0.8203] | [-0.000318, 0.000675] |
| local | -0.000552 | -0.000237 | 10/24 = 0.4167 | [0.2447, 0.6117] | [-0.001747, 0.000582] |
| global | 0.000344 | 0.001149 | 16/24 = 0.6667 | [0.4671, 0.8203] | [-0.000779, 0.001440] |

其他指标：

- matched semantic logp mean：`-0.632020`；
- mean gate：`0.108317`；
- mean relative update norm：`0.053946`；
- 三个 channel 的 `fraction gap > 0.02` 均为 `0/24`；
- full-plan secondary gap：`0.002430`，bootstrap CI `[0.000887, 0.004230]`。

### v2 解释

v2 的 conflict/global 方向比 v1 更偏正，但两者 CI 都跨 0；local 的均值和方向仍为负。
因此 hierarchical attention 没有在这次旧 checkpoint 的重评中显示出可靠的三路语义使用。
相较 v1，v2 update norm 略小，但这不是性能优越证据，只能说明旧 v2 在当前设置下的
residual intervention 更保守。

## 7. v1 与 v2 的直接比较边界

不能把两个 gap 表当成正式 architecture ranking：

- v1/v2 的训练 loss、checkpoint 生成 commit 和 bridge 参数不同；
- target 只有 24 clusters；
- 两者不是同一个新训练预算下的 paired intervention；
- 当前目标是判断是否存在稳定 channel use，而不是选出最优 bridge。

可以作出的有限判断是：**两种旧方法都没有提供足够证据支持“三个 channel 已被
Receiver 稳定读取”；v2 没有在本次小规模重评中解决 local channel 的问题。**

## 8. 为什么状态是 pilot_inconclusive，而不是 No-Go

预注册规则要求所有三个 channel 同时满足：

```text
mean gap >= 0
fraction(gap > 0) >= 0.60
```

v1 因 conflict/local 失败，v2 因 local 失败，所以两者 `point_gate_pass=false`。
但 24 个 image clusters 的比例区间很宽，且 mean-gap CI 大多跨 0；这属于低功效的
机制 pilot，不足以排除中等效应或判断方法类别无效。

因此：

- `hard_no_go`：只用于工程不变量或严格预注册的明显负向证据；本次重评没有发生；
- `pilot_inconclusive`：代码和数据有效，但 24-cluster 方向证据不足；本次 v1/v2 均属此类；
- 不允许把它改写为“latent 无效”；
- 也不允许把 v1 global 的窄正 CI 改写为“latent 成功”。

这个处理避免了把 Type-II error 当成方法失败。

## 9. A3 smoke 的独立工程失败

另一个作业 `6689653` 并未完成 A3 smoke。根因不是本次 v1/v2 重评，也不是 CUDA/OOM：

```text
configs/pilot/cross_attention_semantic_phase_a3.yaml
max_target_tokens = 384
actual global target = 492 tokens
trace = electronic_sheep:325:0
```

`real_trace_bridge_smoke.py` 在 forward/backward 前拒绝该样本，随后 validator 找不到
预期 JSON。这个事件已按 `engineering` 失败记录；没有产生 A3 语义结果，也没有提交正式
A3 训练。配置已改为 `max_target_tokens=768`，以保持完整 HOMER chain，不允许截断；需
重新 smoke 通过后才可考虑 formal A3。

## 10. 受控下一步

当前最合理顺序：

1. 用修正后的 `768` 配置重新做真实 trace smoke；
2. 不重用失败 smoke 的 JSON，不覆盖旧输出；
3. 若 A3 工程 smoke 通过，在剩余未用于 early stopping 的 outer semantic clusters 上
   做一次预注册 confirmation；
4. 对 v1/v2 在 outer 集上继续使用同一 length-matched channel protocol；
5. 只有至少一个方法在三路 channel 和 outer semantic gate 上稳定通过，才进入 caption
   bridge；
6. caption 阶段再使用 Text-HOMER、混合 text/latent 和 all-latent 的固定 seed 生成，
   并用 Group-of-10、镜像 A/B、多评审和 image-clustered bootstrap；
7. 在 caption quality、grounding 和 hallucination 均不退化前，不做 DPO 或 preference
   learning。

## 11. 最终研究结论（当前可写入论文的强度）

可以写：

> 在冻结 Qwen2.5-VL-7B receiver 和 Planner traces 的 24-cluster mechanism pilot 中，
> 经 length-controlled, single-channel counterfactual re-evaluation，旧 v1/v2 bridge
> 均未显示 conflict/local/global 三路一致且具有实际幅度的 semantic sensitivity。
> 结果提示全 memory 改变会影响 reconstruction likelihood，但不能证明每个结构化
> channel 被 Receiver 稳定读取；由于 pilot 样本量有限，该结果应标记为 inconclusive，
> 而不是对 latent communication 的方法级否定。

不能写：

```text
latent communication is useless
v2 is definitively worse than v1
full-plan positive gap proves all channels are used
semantic reconstruction proves better humorous captions
```

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
