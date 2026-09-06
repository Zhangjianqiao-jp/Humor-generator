# v3.5：HOMER 方法、数据与评测协议严格审计

更新时间：2026-09-06
审计范围：仅当前 `v3.5` 树；本文件不把其它版本、其它模型路线或历史结果混入当前结论。

> **状态更新（2026-09-06）**：旧审计中的 v2/878 hash 阻塞已通过独立的
> `homer_pretrained_7b_public_release_362` 数据版本解决。当前 adapted public-release
> population gate 为 `ready`；canonical 365-contest allow-list 仍是未核验的论文级 claim，
> 而 current pretrained Planner hidden-state trace 尚未生成，所以 trace/bridge gate 仍按
> trace 状态阻塞。不要把 trace gate 误读为数据集 gate。

## 结论先行

**当前不是 HOMER 的 100% 复现，也不能把现有 caption 数字直接与 HOMER 论文表格比较。**

更准确的历史命名是：

> `HOMER-style staged planner + pinned Qwen2.5-VL substitution + project-specific latent bridge`

本轮之后的当前执行路线已改为：

> `HOMER public-code stage order + pretrained Qwen2.5-VL-7B Planner/Generator + bridge-only training`

当前路线不加载 `planner_sft` 或 `generator_sft`。旧 adapter 轨道只保留为历史 artifact
审计，不能与新的 pretrained-only 结果混合。

### 当前 bridge 路线迁移状态（2026-09-06）

此前的 `formal_bridge.py`/`cross_attention_bridge.py` caption 分支默认使用历史
`homer.prompts`、generic caption instruction 和 `latent_bridge_v35` trace，缺失官方
summary/retrieval/selection/DFS context；那条路径不能命名为 HOMER + latent bridge。
当前修复已在代码层隔离出唯一的 public route：

```text
scripts/cache_pretrained_homer_traces.py
  -> scripts/cache_pretrained_homer_context.py
  -> scripts/build_pretrained_bridge_dataset.py
  -> scripts/verify_pretrained_bridge_inputs.py
  -> scripts/train_bridge.py
```

`configs/pilot/cross_attention_caption_pretrained_public.yaml` 才是当前 caption
bridge 配置。它使用 pinned `official_prompts.py` 的 `##Caption`/`##Explanation`
system prompt 和两个 user blocks；text teacher 保留 selected conflict/entities/path，
latent student 只移除 plan text 并由 bridge memory 替换。context index 还保存 summary、
retrieval、selection 原始响应、DFS path、trace/output hash 以及 model/prompt/data
provenance。当前尚未训练：必须等 adapter-free Planner trace 362/362 通过后才可建立
context 和 bridge dataset；旧 A5 输出不能回填这些文件。

当前实现与 HOMER 的**概念结构**相符（冲突脚本、local/global imagination、检索增强、随机选择一条联想路径），但在模型权重、API 调用序列、prompt 字符串、链长度、检索实现、数据人口、候选数量和评测器上均存在不可忽略的差异。因此：

| 结论层级 | 当前状态 | 能否称为“100%复现” |
|---|---|---|
| GTVH 三角色的概念结构 | 部分一致 | 否，概念一致不等于实现一致 |
| HOMER 官方模型/权重 | 不一致 | 否 |
| 官方调用次数与阶段顺序 | 不一致 | 否 |
| 官方 prompt 与生成参数 | 不一致 | 否 |
| 官方数据人口与 caption 数量 | 不一致 | 否 |
| HOMER 原始自动/人工评测 | 不一致 | 否 |
| v3.5 latent bridge | 新增扩展 | 不属于 HOMER |

注意：HOMER 论文的 B.12 成本表是 `2+3+2=7`，但官方固定 commit 的 raw
`generator.py` 执行两个 selection 请求后还执行 caption 请求，因此 online public-code
路径实际为 8 个请求。这个公开证据冲突已在新协议中显式记录，不用任何一个数字覆盖另一个。

论文明确描述 HOMER 的三个角色和 GTVH 条件，但它并没有提出 latent bridge 或要求每个 channel 通过独立的因果损失被接收器使用。[HOMER 论文](https://arxiv.org/html/2602.06423) 的方法定义见 §2；[官方实现](https://github.com/Shang-hub/HOMER-Official-Implementation) 是本审计的代码依据。

## 1. 可复核的当前工程对象

| 对象 | 当前路径/标识 | 审计意义 |
|---|---|---|
| 运行配置 | `configs/homer_text_reproduction.yaml` | 已改为 `claim_level: adapted_reproduction`；不再宣称 strict exact |
| 当前 pretrained-only 配置 | `configs/homer_public_code_pretrained_7b.yaml` | Planner/Generator 无 adapter；只允许 bridge train |
| 当前 pretrained-only bridge 配置 | `configs/pilot/cross_attention_caption_pretrained_public.yaml`、`configs/bridges/pretrained_7b.yaml` | public-code caption bridge；旧 `cross_attention_caption_pretrained.yaml` 仅历史审计 |
| 官方 prompt 定义 | `src/humor_generator_v35/homer/official_prompts.py` | 固定 public-code prompt 文本与来源 commit |
| 官方阶段管线 | `src/humor_generator_v35/homer/public_code_pipeline.py` | description/conflict/global/local/summary/selection/caption 阶段及调用 ledger |
| Planner/Generator 基座 | `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` | 本地固定替代模型；不是论文原始 Qwen-VL 权重 |
| 基座 manifest | `manifests/local_qwen2_5_vl_7b.json` | 记录 5 个权重 shard 与元数据 hash；architecture 为 28 个语言层、hidden size 3584 |
| 冻结 adapter manifest | `manifests/frozen_7b_adapters.json` | `planner_sft`、`generator_sft` 的路径与逐文件 hash |
| 文本 HOMER 管线 | `src/humor_generator_v35/homer/pipeline.py:49-158` | 当前 `HomerTextPipeline` |
| prompt 定义 | `src/humor_generator_v35/homer/prompts.py:11-82` | 当前实际字符串，不等于官方逐字 prompt |
| 检索实现 | `src/humor_generator_v35/homer/retrieval.py` | TF-IDF/WordNet 适配器 |
| 数据构建 | `src/humor_generator_v35/data/clustered.py:86-324` | 当前 source rows、cluster split 与清洗规则 |
| 当前数据 manifest | `manifests/homer_population_public_release_362.json`、生成后的 `data/processed/homer_pretrained_7b_bridge_362/manifest.json` | 362 public population 与 bridge training view 分开记录 |
| 历史 Planner trace | `data/cache/planner_traces_homer_strict_v35/index.jsonl` | 666 条 trace，使用旧 SFT/旧 prompt；当前路线禁止复用 |
| 历史 A5 bridge | `outputs/pilot/cross_attention_semantic_phase_a5_lengthmatched/best_bridge.pt` | 仅 bridge 可训练，但不属于当前 HOMER public-code route |
| 历史生成入口 | `scripts/generate_formal_baseline.py:65-466` | 使用 `first_rows_by_cluster`；只用于解释旧结果，不可作为当前人口 |
| 当前生成入口 | `scripts/generate_homer_public_code.py` | 预训练 7B、无 adapter、完整 staged pipeline；在新 source-aware manifest/trace 通过前 fail-closed |
| 历史 caption 作业/评测 | `jobs/a5_joint_caption_generation.pjm`、`outputs/caption_judgement/a5_joint_group10_20260905_rubric_v11/` | 3 条条件、10 seeds、3,630 captions；属于旧 A5 extension，不是当前 HOMER 结果 |

## 2. HOMER 论文/官方实现规定的协议

### 2.1 三阶段和信息流

论文形式化为：

\[
\operatorname{Extract}(I)\to(\mathcal C,D),
\qquad
\operatorname{Imagine}(I,\mathcal C,D)\to\mathcal T_{\rm im},
\]

再将 \(D\)、选中的 conflict、单条 imagination path 和 \(\Omega\in NS\times LA\) 交给 caption generator。论文明确说 generator 先随机选择 conflict 与 target，再 DFS 枚举路径并**采样一条路径**，不是把所有 conflict/path 一次性拼接进去。[HOMER §2.3](https://arxiv.org/html/2602.06423#S2.SS2) 的算法行 36–40也这样规定。

官方 `demo.py` / `generator.py` 的一条完整输出包含：

```text
Extractor:  image -> vivid description -> conflict scripts
Imaginator: global image view + local description view
            -> merge/summary -> retrieval/pruning
Generator:  select two conflicts + select two entities
            -> DFS paths -> sample path -> caption + explanation
```

官方代码中，caption 阶段包括两个额外的选择 API 调用（选两条 conflict、选两个实体），随后还有一次 caption API 调用；因此 raw public-code 在线路径是 `1 description + 1 conflict + 2 imagination + 1 summary + 2 selection + 1 caption = 8`。论文附录 B.12 的表格把 Generator 写成 2、总数 7，和 raw code 存在公开不一致。当前协议同时记录 `paper_declared_call_budget=7`、`raw_public_code_call_count_online=8`，不把两者混为一谈。[官方仓库 README](https://github.com/Shang-hub/HOMER-Official-Implementation#code-overview)；[论文附录 B.12](https://arxiv.org/html/2602.06423#S7.SS12)。

### 2.2 论文明确的数据统计

| 数据集 | cartoons | 平均 captions/cartoon | groups | ranking | standard description |
|---|---:|---:|---:|---|---|
| Humor in AI / New Yorker | 365 | 6,044 | 3 | global | GPT-4o |
| Electronic Sheep | 679 | 6 | 2 | pairwise | human |

HOMER 以 HIA 的 `#top10`、`#200–#209`、`#1000–#1009` 三组作比较；Electronic Sheep 将三组 pairwise 结果拆成 High-Humor / Low-Humor。该表和分组见论文 Table 1 与 §3.1。[论文 Table 1](https://arxiv.org/html/2602.06423#S3.T1)

检索语料在论文正文称来自 11 个 one-liner joke datasets，附录列出的来源叙述仍有 11/12 的内部不一致；论文声称清洗后 335,570 条。公开 CSV 的物理行数包含表头，实际 CSV records 是 335,569，不能为了凑数人为复制一行。[论文附录 B.1](https://arxiv.org/html/2602.06423#S7.SS1)。

### 2.3 论文的生成与评测参数

| 项目 | HOMER 论文/官方实现 |
|---|---|
| Imaginator retrieval | `k=5`, humor-relevance rank threshold `delta=5` |
| Caption sampling | base LLM temperature `1.0`，其它参数默认 |
| Humor evaluator temperature | `0` |
| primary automatic evaluator | GPT-5 (paper label); official evaluator script literal model ID `gpt-5-chat-latest` |
| evaluator reliability candidates | LLaMA3-8B、Humor-tuned LLaMA3、Qwen-Turbo、GPT-4.1、GPT-5 |
| HIA benchmark（NeurIPS 2024）原始 group evaluator | Group Overall：GPT-4-Turbo；Group Best Pick：GPT-4o-vision；这是 HIA 基准协议，不是 HOMER 论文的 primary evaluator |
| main automatic metric | unbiased Pass@K，论文实验 `n_i=5`, `K∈{1,3,5}`, 5 repeated trials |
| GPT-5 diagnostic | 每图对 8 个方法按 visual understanding、humor understanding、stylistic expression 排名，5 次重复 |
| diversity | Distinct n-grams + non-entailment percentage from RoBERTa-large NLI |
| human protocol | 每图 7 个候选；先判 relevance，相关后 1–5 funny rating；论文给出 `20 images × 20 raters × 7 methods = 2,800`，但同一段文字后又写 12 raters，原文内部矛盾需在复现中披露 |

这里还必须把 **HIA 原始 benchmark 的评测协议** 单独列出。HIA NeurIPS 2024
论文不是 HOMER 的 primary 评测：它从 358 个可用 contests 中留出 91 个做 evaluator
评测；每个 contest 生成 10 条 caption，并与四组人类 caption（`#1-10`、
`#200-209`、`#1000-1009`、median-ranking group）做 Group Overall/Group Best Pick
比较。Group Overall 使用 GPT-4-Turbo + Hessel description，Group Best Pick 使用
GPT-4o-vision + 原图。HOMER 论文随后采用自己的 `n_i=5`、Pass@1/3/5、五次重复和
GPT-5 primary evaluator；这两个协议的 holdout、候选数、参考组和模型都不同，不能用
“HIA 的十候选”替换“​HOMER 的五候选”。[HIA NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf)

Pass@K 的定义为：

\[
\operatorname{pass@k}
=\frac1N\sum_i\left(1-\frac{\binom{n_i-c_i}{k}}{\binom{n_i}{k}}\right).
\]

它是“候选中至少一个胜过对应 human caption”的无偏估计，不是当前 A5 的 A/B win rate。[论文附录 B.3](https://arxiv.org/html/2602.06423#S7.SS3)。

### 2.4 检索实现的新增逐项核对

本轮把新 public-code 入口的检索器从历史 `HomerRetrievalAugmenter` 中隔离出来，改为
`src/humor_generator_v35/homer/retrieval.py:OfficialCodeRetrieval`。它逐项对应固定官方
`imaginator.py` 的可观察逻辑：实体正则优先、单 query 拟合的
`TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,3))`、只抽取名词/动词、
WordNet WUP 与关系 Jaccard、官方 TF-IDF 频率项、八类 POS 多样性归一化，以及
root/value/retrieved-entity edge tree。生成入口的 provenance 明确记录
`official_code_compatible_tfidf_wordnet_entity_tree`。

这仍不能称为远端运行的字节级复制：官方没有锁定 NLTK/WordNet/sklearn 版本、Python
随机状态或 OpenAI 返回内容；当前代码只把算法和输入/输出边界固定下来，并把这些环境差异
保留为可报告的 adaptation。旧 `HomerRetrievalAugmenter` 不得用于声称 public-code exact。

## 3. 当前 v3.5 与 HOMER 的逐项核对

### 3.1 一致的部分（只能称“概念/接口一致”）

1. **角色分离**：当前 public-code pipeline 有 Planner/Extractor 的 description/conflict、Imaginator 的 local/global/summary/retrieval、Generator 的 selection/caption 阶段，整体对应 Extractor → Imaginator → Generator；历史 `HomerTextPipeline` 不再作为当前主路线。
2. **冲突脚本**：`CONFLICT_SYSTEM` 要求至少两个 script pair，与 GTVH/script opposition 的方向一致。
3. **两种观察视角**：local 读取 standard description，global 读取 image，与论文的 local/global view 定义一致。
4. **逐步联想**：当前 schema 强制每个 chain 具有 3 个 successor，符合“每一步由前一实体产生”的思想。
5. **检索与 pruning**：当前有 TF-IDF 与 WordNet 相关分支，配置锁定 `top_k=5`、`delta=5`，与论文公开超参一致。
6. **单路径采样**：当前 `HomerPublicCodePipeline` 通过显式 seeded Python RNG 选 DFS path，保留了“不要把所有路径同时交给 generator”的原则；这改善复现性但不是官方未设 seed 的随机状态逐字复现。

**联想请求粒度必须单独说明。** 论文用
\(e_{\tau+1}=f_{\rm chain}(e_\tau)\) 描述逐步依赖，但固定公开代码的
`carton_imagination_gpt4_o`/`description_imagination_gpt4_o` 每个视角只发出一次
请求，由该请求返回一个包含三个 successor 的 JSON 列表；它没有为 successor 1、2、3
分别发三次在线请求。当前 `public_code_exact` 轨道复现的是这个**一次请求返回列表**的
公开代码行为，并在 schema 中要求每个列表长度为 3；若要研究真正的逐步多请求递归，必须
另命名为 `paper_recursive_association`，并单独报告 API 成本，不能把它写成官方代码的
逐字复现。[HOMER 论文 §2.2](https://arxiv.org/html/2602.06423#S2.SS2)，[官方实现](https://github.com/Shang-hub/HOMER-Official-Implementation)

### 3.2 不一致部分（使“100%复现”不成立）

| 协议层 | 论文/官方实现 | 当前 v3.5 | 判定 |
|---|---|---|---|
| **基座模型** | 论文表格报告 Qwen-VL 等 base LLM；论文没有提供不可变 Qwen-VL revision。官方代码实际使用 OpenAI `gpt-4o` API。 | 当前主路线 Planner/Generator 都是本地 `Qwen2.5-VL-7B-Instruct@cc5948…`，`adapter:null`、policy frozen；这是明确的 pretrained-7B model substitution。 | **权重不等价，但已移除 SFT 混杂** |
| **Extractor** | 新图片先用 `gpt-4o` image→vivid description，再用 `gpt-4o` description→conflict，均 `temperature=1`, `max_tokens=1000`。 | 新 `HomerPublicCodePipeline` 按同一阶段顺序运行；standard-description replay 单独标注为跳过 description 请求的 replay；主线后端换成冻结本地 Qwen。 | **阶段/参数对齐；模型仍为 substitution** |
| **Conflict prompt** | user 内容是 `Description:\n{description}`，官方 system/user 字符串来自 `extractor.py`。 | 新路线使用 `official_prompts.py` 的固定字符串、官方 system 字符串形状和 `temperature=1`, `max_new_tokens=1000`；旧 `prompts.py` 仍仅供历史路线。 | **文本/序列化形状对齐；模型替代仍存在** |
| **Local/global prompts** | 官方 system prompt 含 JSON 示例 `{entity: ['idea1','idea2','idea3']}`；global user 同时给 image 与 conflict；local user 给 description 与 conflict。 | 新路线固定使用官方 prompt 常量和内容顺序；schema 校验仍作为工程约束单独记录。 | **文本/内容顺序对齐；严格 schema 是额外 gate** |
| **Imagination 深度** | 论文正文把 successor 写成由前节点递归生成、长度由 LLM 自适应（经验平均约 4，含 root）；但公开代码的 imagination prompt 明确要求每个 root 生成 **3 个** successor。 | 当前固定 3 successor，即 root+3；这与公开代码 prompt 的字面长度接近，但不能复现论文所述 adaptive length distribution，也未复现官方图构建/扩展细节。 | **对公开 prompt 部分一致；对论文级机制不等价** |
| **Summary/merge** | 官方对 local/global JSON 另做一次 `summary_imagination_gpt4_o` 合并、去重后再检索。 | 新 `HomerPublicCodePipeline` 强制 summary 请求完成后才进入 retrieval；旧 cached-trace pipeline 不得用于新路线。 | **新路线已补齐；旧 artifact 仍不等价** |
| **Retrieval** | 官方代码先 exact substring；否则 `TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,3))`，对每个 chain node 取 top-5，再做 token/lemma/POS/WordNet 与 H 分数 pruning。论文将 embedding backend 留作可选实现。 | 当前是 `OfficialQueryFittedTfidfIndex` + 项目 `NltkWordNetGraph`，查询/合并/字段保留规则为适配版本，不能证明逐行等价。 | **适配实现** |
| **Conflict/entity selection** | `generator.py` 额外调用 `gpt-4o` 选择两条 conflict 和两个 key entities。 | 新路线执行两个独立 selection 请求，再以 seeded Python RNG 选择 DFS path；旧 caption pipeline 不满足此项。 | **新路线阶段对齐；seed contract 是可复现适配** |
| **Caption prompt** | raw public prompt 要求 caption 与 explanation，格式为 `##Caption:` 与 `##Explanation:`；官方 generator 将 context 和 free-association 放在两个独立 user blocks；论文形式化变量另记为 `Ω` narrative/style，但 raw prompt 未公开完整 Ω 词表。 | 新路线保存 raw caption blob、解析 caption/explanation，并保留两个 user content blocks；Ω 仅在独立、版本化的 `project_omega_grid_v1` variant 显式启用（36 个 NS×LA 组合）。 | **文本、标记和 block 边界已对齐；项目 Ω 已定义，但不是官方词表** |
| **Caption token budget** | 官方 generator 的 `max_tokens=1000`。 | `HomerPublicCodePipeline` 使用 `max_new_tokens=1000`；Qwen backend 的视觉 token 预算另行记录。 | **解码预算对齐；后端视觉预算是适配参数** |
| **Randomness** | 官方 `random` 未在代码中给出跨运行 seed contract。 | 当前为 Python seeded RNG，方便可复现但不等价于官方未固定随机流。 | **可复现性改善但非 exact** |
| **Latent** | HOMER 没有 bridge、hidden-state injection 或 latent semantic loss。 | A5 是 receiver-driven layer-output residual cross-attention bridge，两个 7B 冻结，仅 bridge 训练。 | **项目扩展** |

这里要特别区分：当前的 cross-attention 数学形式可以是一个合法的 gated residual adapter，但“数学上合法”不等于“HOMER 的原实现”。它也不自动证明每个 channel 对最终 caption 有因果贡献；当前 A5 的 semantic gate 是项目自定义机制验证，不是 HOMER 的评测指标。

## 4. 数据集、数据量和数据污染审计

### 4.1 官方发布物与论文数字

`manifests/homer_official_assets.json` 已固定官方仓库 commit `d1334f295cc1a8f8f6dc67ba7e846c5939dddcec`，并保存：

| 资产 | 实际发布物 |
|---|---|
| HIA standard descriptions | train 271 + validation 44 + test 47 = **362 records** |
| Electronic Sheep descriptions | **679 records** |
| joke CSV | 335,570 physical lines（含 header），335,569 CSV records，335,569 unique IDs，335,569 unique exact jokes |
| joke CSV SHA-256 | `681059f010868c1021eeb9150828536b9cfe99ad5d81288acd98efb2c19d7d31` |

论文/NeurIPS HIA 数据卡宣称 HIA 为 365 contests（530–895）、2.2M captions、平均 6,044 captions/contest；当前发布的 GPT-4o description 三文件只有 362 条。官方 evaluator 资产在公开仓库 README 中另列 `sample5gt.json`（HIA）和 `sampled3_evaluate_data.json`（ES）；HIA evaluator 文件覆盖的 contest 人口与 365/362 描述子集并不完全相同，不能静默混用。[HIA NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf)，[官方 HOMER README 数据说明](https://github.com/Shang-hub/HOMER-Official-Implementation#datasets)。

本地 ranking 目录还需要单独披露：当前扫描到 **385 个 contest CSV（510–895）和 2,297,153 条 caption 行**；它既包含论文 HIA 人口之外的 510–529，也不等同于论文宣称的 530–895/365 contests。因此不能把“目录里有 385 个 CSV”当作 HOMER 论文 benchmark 已完整复现。论文级复现必须先建立一份明确的 365-contest allow-list，并记录每个 contest 的图片、ranking、description 和 evaluator asset hash。

因此“365”是论文 benchmark 人口 claim，“362”是当前已封存、可逐项核验的 public
standard-description release。按简单的闭区间 `530..895` 对照，当前描述中缺少
`[605, 644, 657, 890]` 四个编号；但该闭区间本身包含 366 个整数，和论文写的 365 个
contest 不一致，不能武断地把其中一个编号当作“论文人口之外”。当前 adapted route 以
362 版本运行并明确标注 claim level；只有获得官方逐项 365-ID allow-list 后，才能另建
canonical route，不能用空描述或新生成描述补齐。

### 4.2 当前 v3.5 bridge 数据

当前 `data/processed/latent_bridge_v35/manifest.json` 的真实统计为：

| split | cluster 数 | JSONL rows | image hashes | 用途 |
|---|---:|---:|---:|---|
| train | 602 | 2,162 | 721 | bridge fitting |
| validation | 64 | 216 | 72 | checkpoint selection |
| internal_test | 97 | 327 | 109 | sealed internal test |
| official_hia_seen_diagnostic | 23 | 69 | 23 | adapter-seen diagnostic only |
| official_hia_unseen_test | 24 | 72 | 24 | adapter-unseen confirmatory test |
| 合计 split rows | 810 clusters | 2,846 | 949 | 不含 trace index 文件 |

另外，历史 Planner trace 是 **666 个 train+validation clusters**；历史 caption outer 生成使用 **121 个 clusters**（97 internal + 24 official-unseen），每个条件 10 seeds，因而是 3,630 条 caption，而不是 1,000 张独立图片。当前预训练-only 轨道尚未生成新的 trace/outer rows，不能把这些历史数量当作当前评测人口。

当前历史 bridge 数据每个 source 最多保留 3 个 caption：HIA 取排名靠前的 3 条，ES 取最多 3 个 finalist。它适合机制 pilot，但不是 HOMER 论文所用的完整 HIA 6,044-caption/contest 人口，也不是 HOMER 自动评测的 5-candidate reference 文件。新路线必须从完整 ranking/source 目录建立 source-aware population；任何截断只能作为显式 pilot，不得进入论文级 HOMER claim。

### 4.3 必须修正的数据键问题

`src/humor_generator_v35/data/clustered.py:101-103` 将 cluster 写成 `nycc_{contest}`，没有带 dataset namespace；同一 contest number 在 HIA 与 ES 中会发生 cross-source collision。manifest 记录了 172 个 cross-source duplicate cluster candidates；split 检查还显示 `internal_test` 的 97 个 clusters 对应 109 个 image hashes，证明 cluster 不是唯一图片。

随后 `scripts/generate_formal_baseline.py:65-69` 的 `first_rows_by_cluster()` 只保留字典中的第一行。因此：

1. 一个 cluster 可能有多个 source/image，但 caption generation 只取其中一张代表图；
2. 代表图取决于排序后的 `row_id`，不是论文定义的 contest population；
3. 不能把当前 810 clusters 或 121 outer clusters 写成 810/121 张独立 benchmark 图片而不披露这一折叠。

这不是“数据量小所以一定幻觉”的直接证明，而是**统计人口与样本独立性未满足论文级复现**的硬问题。应先改为 source-aware key，例如 `(dataset, contest_number, image_sha256)`，然后重新生成 manifest、trace 索引和所有 outer packet。

### 4.4 历史 v2 与当前版本的实时完整性门禁

历史 v2 parent 在早期审计中实际运行：

```text
.venv/bin/pytest -q                                  -> PASS (current suite)
.venv/bin/python scripts/verify_clustered_dataset.py -> FAIL (historical v2 parent)
.venv/bin/python scripts/verify_homer_public_release_362.py -> PASS (current release)
```

失败不是推测：`humor_in_ai:878:0/1/2` 的 manifest 声明图片 hash 为
`c87ee5f587d420f095fc35dae6c46780c39d767ffd3662d7ce5ea1e4369e4e01`，而当前工作区
`data/external/benchmarks/humor_in_ai/cartoons/source/878.jpg` 的实际 hash 为
`b690e9d81a36d6a2cf5fd7a8bbf411ecd4c0f959afb6d97aba1f8f0f4af7ae0a`。同一张图片对应
3 行，所以审计报告了 3 个 row errors。

（以下是历史 v2 parent 的 provenance 诊断；当前路线已改用新的 public-release 362 manifest。）
这意味着历史 v2 数据/生成物不能称为 immutable provenance：可能是图片在 manifest 构建后被重新下载或转换。**不得直接把旧 manifest 中的字符串改成新 hash**，否则无法知道旧 trace/caption 是否针对同一字节。正确修复方式是建立新数据版本：

1. 保留 v2 作为父 lineage，不修改其历史 hash；
2. 从固定公开 description split 重建 source-aware 362 allow-list，并固定每个文件 hash；
3. 使当前路线只引用新 manifest，旧 trace、outer generation、blind packet 不得混入；
4. 用 `verify_homer_public_release_362.py` 返回 `status=pass` 后，再锁定当前路线数字和结论。

所以历史 A5 caption 数字仍可作为工程记录；当前 362 release 的 data gate 已通过，但新的
pretrained trace 和正式 bridge gate 未通过前，不能把旧数字升级为当前 HOMER 主结果。

## 5. 当前 caption 盲评与 HOMER 评测的差异

### 5.1 当前真实执行的评测

`outputs/caption_judgement/a5_joint_group10_20260905_rubric_v11/provenance.json` 记录：

```text
images/clusters: 121 (97 internal + 24 HIA adapter-unseen)
conditions: text_homer, full_plan_text, a5_typed
seeds: 10
captions: 3,630
blind packets: 484
mirror pairs: 242
protocol: anonymous Group-of-10 A/B, mirrored sides, absolute good/weak/bad
bootstrap unit: image cluster
```

三个 judge 文件都覆盖 484 packets，temperature=0、prompt hash 一致，且 prompt 要求传入真实 `image_path` 对应的图片、每个 packet 独立判断、不读取 `private_mapping.jsonl` / `blind.secret`。这些是良好的盲评工程约束。

但 evaluator 身份并不等于 HOMER 论文的单一 primary evaluator：

| 文件 | 实际 judge metadata | 与 HOMER primary GPT-5 是否相同 |
|---|---|---|
| `judge-1.json` | local `codex-gpt-5-rule-assisted-visual-rubric` | 否 |
| `judge-2.json` | OpenAI `GPT-5` | 名称相同，但 API snapshot/version、系统 prompt 和 5-shot calibration 未按论文锁定，不能直接宣称 exact |
| `judge-3.json` | OpenAI `GPT-5.6-Sol` | 否 |

此外，`provenance.json` 的 `evaluation_status` 仍是 `awaiting_independent_ratings`，与已存在的三个 judge 文件不一致；这属于 provenance 尚未闭环，不应在论文中直接使用该文件作为“已聚合”的证明。

### 5.2 与 HOMER 原始自动评测的逐项不同

| 项目 | HOMER 论文 | 当前 v3.5 |
|---|---|---|
| primary 输出 | GPT-5 对候选相对 human groups 的判断 | 3 个异质 judge 对两个系统做 A/B group comparison |
| candidates/image | `n=5`，用于 Pass@1/3/5 | 每条件 10 seeds/candidates |
| comparison target | HIA #top10/#200–209/#1000–1009；ES High/Low | `text_homer`、`full_plan_text`、`a5_typed` 相互比较 |
| 评测输入 | 按 paper protocol 使用 description 或 raw image；5-shot | 当前 custom rubric 使用真实 image，并含项目自定义字段 |
| 主指标 | unbiased Pass@K，5 repeated trials | image-cluster bootstrap A/B win rate、Best Pick、good/weak/bad |
| 多维评测 | GPT-5 对 8 methods 做 4 维诊断；Distinct + RoBERTa-large NLI | 当前 rubric 的 humor、grounding、originality、specificity 等，未按论文的 8-method/5-trial protocol 完成 |
| human evaluation | 7 candidates/image、relevance→1–5；论文样本量存在 20/12 raters 文字矛盾 | 不是人工 raters，而是 3 个 LLM/本地 judge |

因此当前盲评是一个合理的**项目扩展评测**，可以回答“在同一 121-image、同一 seeds 下 latent 是否优于当前 text controls”，但不能回答“HOMER 论文 Table 3 的 Pass@K 是否被复现”。

## 6. 为什么当前 Text-HOMER 结果偏低不能直接归因于 HOMER

当前报告中的 A5 相对 `text_homer` 较差，具有项目内部比较价值；但不能据此说 HOMER 论文方法本身很差，原因是比较中同时改变了：

1. base model（本地 Qwen2.5-VL，而非论文/官方调用链）；
2. prompt 字符串和输出 contract；
3. Extractor、summary、selection API 调用次数；
4. adaptive chain 与固定 chain 长度；
5. retrieval/merge 实现；
6. caption max token 和 `Ω` style control；
7. 评测人口（121 representative clusters，而非 HIA 365/ES 679）；
8. reference（当前系统互比，而非 human-ranked groups）；
9. evaluator（异质三个 judge，而非论文 primary GPT-5 + 5-shot calibration）。

这解释了为什么“我们的 Text-HOMER 很差，而论文 HOMER 较好”不能用单一 failure case 解释。当前数字只能命名为 `v3.5 adapted-Qwen internal comparison`。

## 7. 对用户提出的复现问题的明确回答

### 7.1 是否一比一复现了 HOMER？

**没有。** 当前新 public-code route 已补齐官方 prompt、summary、两个 selection 请求、DFS 单路径和 `##Caption/##Explanation` block contract；但它仍使用 Qwen2.5-VL-7B 替代官方代码的 `gpt-4o`，默认固定公开代码的 3 successors（不能复现论文所述 adaptive-length 分布），检索实现和 seed contract 也是可复现适配。因此它可称 `adapted pretrained-7B public-code protocol`，不能称论文权重级 100% 复现。当前 362-ID public-release data gate 已通过；canonical 365 仍是未核验的 claim，且 current pretrained trace 尚未生成。历史 cached-trace/A5 路线则另外缺失上述阶段，不得混入。

### 7.2 是否按 StateBridge 原方法保存和使用 hidden state？

**不能这样声称。** 当前 trace 的 `teacher_forced_post_token` 是已读入 token 后的 hidden state，形状 `[1,T,3584]`；它是 v3.5 自定义 Planner trace。它既不是 HOMER 原组件，也不能自动等同于 StateBridge 的 predictor-state/receiver-compatible state protocol。StateBridge 只能作为另一个有明确命名的 baseline，不能反向证明 HOMER 复现完成。

### 7.3 当前 cross-attention 数学是否“错了”？

形式上，当前每路 `QK^T/√d`、softmax、`W_O` 和 gated residual 是合法的注意力适配器；失败风险不在一个明显的线性代数符号错误，而在**协议与可识别性**：latent 的 channel mask、文本 anchor、图片 shortcut、post-token 对齐和 target/objective 都与 HOMER 无关。即便 semantic gap 为正，也只能证明当前 bridge 的机制 gate，不是“复现 HOMER”。

### 7.4 A5 的语义损失是不是最终 caption 目标？

不是。A5 的 bridge loss 是 semantic reconstruction/counterfactual/alignment 等加权目标；HOMER 本身没有这个训练损失。最终 caption 的 humor、grounding、specificity 和绝对 good rate 必须单独在下游 caption evaluation 中验证。损失数值改善不能代替 caption 质量结论。

## 8. 论文级复现前必须执行的修正门禁

### Gate A：数据人口与键

1. 用 `(dataset, contest_number, image_sha256)` 重建 cluster key；禁止 `nycc_{contest}` 跨数据集复用。
2. 不再用 `first_rows_by_cluster()` 静默丢图；若一个 contest 有多张图，必须保留并在 manifest 中声明选择规则。
3. 当前 adapted route 使用 `manifests/homer_population_public_release_362.json` 的 362-ID
   allow-list；canonical 365 仍需官方逐项清单，不能由数值范围补齐。
4. 单独固定官方 evaluator assets（HIA `sample5gt.json`、ES `sampled3_evaluate_data.json`）及 hash；不与 bridge training rows 混为同一人口。
5. 保留官方发布 joke CSV 原 hash，并披露 `335,570 physical lines / 335,569 records`。

### Gate B：HOMER-style reference run

二选一，必须在结果标题中写清楚：

- **Exact public-code track**：运行官方 `gpt-4o` extractor/imaginator/generator 调用序列、官方 prompt、`temperature=1`、summary/selection calls 和官方 evaluator assets；这仍不等于论文权重复现，但能最大程度复现公开代码。
- **Pinned-Qwen adapted track**：保留当前 Qwen2.5-VL revision，但把所有报告标为 adapted；不得与上面的 official-code track 合并成一个“Text-HOMER”。

两个 track 应共享同一 source-aware test manifest，但不能共享“exact”标签。

### Gate C：评测器和统计量

1. HOMER-comparable 主表：实现 GPT-5 primary、temperature 0、论文 5-shot prompt、5 candidate sampling 与 unbiased Pass@1/3/5（5 trials）。
2. 分别报告 HIA 三个 human rank groups 和 ES High/Low；不要只报告 system-vs-system win rate。
3. 继续保留当前 Group-of-10/mirror/absolute labels 作为 extension table；明确它不是 HOMER primary metric。
4. 若使用多个 GPT judge，固定每个 API model ID/snapshot/date、prompt hash、seed/temperature，并把 judge identity 写入最终 provenance；先修正 stale `evaluation_status`。
5. 以 image/contest cluster 为 bootstrap 单位，不能把同一图片的 10 seeds 当成 10 个独立样本。该统计原则与 NLP paired evaluation 的建议一致。[Dror et al.](https://aclanthology.org/P18-1128/)，[Peyrard et al.](https://aclanthology.org/2021.acl-long.179/)。

### Gate D：latent extension

只有 Gate A–C 的 reference baseline 通过后，才继续报告：

```text
Text-HOMER (official-code or adapted-Qwen, explicitly named)
StateBridge-style baseline
Learned bridge
Typed bridge
```

latent 结果需要另报：matched caption quality、shuffled latent quality、image-cluster bootstrap CI、seed variance 和 absolute good/weak/bad。它们回答的是“latent extension 是否有下游收益”，不能被倒写成“HOMER 复现指标”。

## 9. 最终判定表（当前日期）

| 问题 | 判定 |
|---|---|
| 当前三角色是否体现 HOMER 思路？ | 是，概念上 |
| 当前是否使用了 HOMER 官方公开 joke artifact？ | 是，文件和 hash 已固定 |
| 当前是否使用了论文宣称的完整 365/679 benchmark 人口？ | 否 |
| 当前是否保留了 HIA 全量 captions/官方 human-group evaluation？ | 否，bridge 数据只保留每 source 最多 3 条；outer 是 121 representative clusters |
| 当前历史盲评是否使用与 HOMER 相同的 evaluator model/protocol？ | 否；是异质多 judge custom rubric。新的 exact-comparable route 已锁定官方脚本的 `gpt-5-chat-latest`（论文标签 GPT-5）、5 candidates/5 trials/Pass@1/3/5，但尚未运行 |
| 当前 caption 盲评是否合理？ | 作为 v3.5 extension 合理；作为 HOMER exact reproduction 不合理 |
| 当前 latent bridge 是否属于 HOMER？ | 否，属于 v3.5 project-specific extension |
| 是否可以声称 100% 完整复现？ | **不可以** |
| 当前 adapted public-release 362 数据完整性是否已通过？ | **已通过**；使用 `verify_homer_public_release_362.py` 验证 manifest、allow-list、description、image、ranking 和 source rows |
| canonical 365-contest allow-list 是否已核验？ | **未核验**；仍是单独的 claim-level 限制 |
| 当前 pretrained Planner trace 是否已生成？ | **未生成**；trace/bridge gate 仍阻塞 |

## 10. 权威参考文献与公开实现

1. Shang, Sun, Ma, Huang. *On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation*. ICLR 2026, arXiv:2602.06423. [论文 HTML](https://arxiv.org/html/2602.06423) · [官方实现](https://github.com/Shang-hub/HOMER-Official-Implementation)
2. Zhang et al. *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*. NeurIPS 2024. [论文与数据集](https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf) · [Hugging Face 数据](https://huggingface.co/datasets/yguooo/newyorker_caption_ranking)
3. Hessel et al. *Do Androids Laugh at Electric Sheep? Humor “Understanding” Benchmarks from The New Yorker Caption Contest*. ACL 2023. [ACL Anthology](https://aclanthology.org/2023.acl-long.41/)
4. Bai et al. *Qwen2.5-VL Technical Report*. arXiv:2502.13923. [论文](https://arxiv.org/abs/2502.13923)
5. Peng et al. *StateBridge*. COLM 2026. [论文](https://arxiv.org/abs/2608.13317)
6. Du et al. *Enabling Agents to Communicate Entirely in Latent Space (Interlat)*. ACL 2026. [ACL Anthology](https://aclanthology.org/2026.acl-long.1248/)
7. Li et al. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. ICML 2023. [PMLR](https://proceedings.mlr.press/v202/li23q)
8. Dror et al. *The Hitchhiker’s Guide to Testing Statistical Significance in NLP*. ACL 2018. [ACL Anthology](https://aclanthology.org/P18-1128/)
9. Peyrard et al. *Better than Average: Paired Evaluation of NLP Systems*. ACL-IJCNLP 2021. [ACL Anthology](https://aclanthology.org/2021.acl-long.179/)
