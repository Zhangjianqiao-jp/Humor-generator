# HOMER 注释回溯与当前修正（v3.5）

更新时间：2026-09-06
范围：只针对当前 v3.5 的 HOMER 复现审计和预训练 7B + bridge 主线。旧 v2.5/v3.0
训练结果不被移植到本路线。

> **状态更新（2026-09-06）**：本文件中关于 v2 的“数据门禁 blocked”和 contest 878
> hash 不一致属于历史审计快照。当前路线已建立并封存
> `homer_pretrained_7b_public_release_362`；362-ID 公共发布版 population gate 已通过，
> 878 不再是当前版本的修复项。canonical 365-contest allow-list 仍没有公开、可逐项核验
> 的来源，因此只保留为单独的论文 claim，而不伪造补齐。当前剩余 gate 是预训练 Planner
> hidden-state trace 尚未生成，不是数据人口门禁。

## 结论

目前不能声称“100% 复现 HOMER”。可以准确地说：

```text
官方 HOMER public-code prompt/stage reproduction
+ pinned Qwen2.5-VL-7B model substitution
+ project-specific bridge extension
```

`official_prompts.py` 已逐字固定官方仓库 commit
`d1334f295cc1a8f8f6dc67ba7e846c5939dddcec` 的 prompt 文本、system 字符串形状、user
content block 边界和消息顺序（本地 prompt hash 记录在
`manifests/homer_official_assets.json`）；
`public_code_pipeline.py` 已覆盖 description、conflict、global/local imagination、
summary、retrieval、conflict/entity selection、DFS path selection 和 caption/explanation。
但是官方代码使用 `gpt-4o` API，而当前主线使用本地固定 revision 的
`Qwen2.5-VL-7B-Instruct`，因此这是一个可复现的 **adapted pretrained-7B track**，不是
论文权重级 exact reproduction。

## 逐条回应注释

### 1. 7B 仅靠预训练是否一定做不好幽默？

不能下“7B 无法完成”的结论。HOMER 论文自己的 Table 3 报告了 Qwen-VL (7B) 的
非零结果，例如 HIA #Top10 的 CoT Pass@1 为 16.76，HOMER 结构下为 24.06；这说明
预训练 7B 有可用的视觉/语言和有限幽默能力，但明显低于强闭源模型和 HOMER。论文同时
说明 Qwen-VL 只是低基线，不提供可核验的原始权重 revision。Qwen2.5-VL 技术报告证明
的是通用视觉理解和 grounding，不是 New Yorker 幽默能力。因此当前采用预训练 7B
作为两个冻结 policy、只训练 bridge，是一个能隔离通信变量的严谨实验，不是声称
预训练 7B 已足够达到 SOTA。

后续如果目标变成“最大化最终幽默质量”，可以另开有 SFT 的能力增强轨道；但它不能与
当前 bridge-only 因果比较混在一起。

### 2. Prompt 必须完全一致

当前主线的 prompt 文本、标点、system 字符串和 user 多模态 block 顺序来自官方固定
commit，并由 `tests/test_homer_public_code_protocol.py` 和
`scripts/homer_public_code_smoke.py` 检查。Qwen 本地图片 block
`{"type":"image","image":<local path>}` 是 provider 适配；官方 OpenAI 请求是
`image_url` + base64 data URL。两者不是字节级 HTTP 请求相同，但 prompt 语义文本和角色
结构保持不变，且已在 provenance 中区分。

旧 `homer/prompts.py` 不再作为当前 public-code route 的 prompt 来源；新结果必须记录
`official_prompts.py` 的 hash。

### 3. 必须完整执行 HOMER 的多角色通路

当前流程固定为：

```text
Extractor: image -> vivid description -> conflict scripts
Imaginator: image+conflict -> global JSON
            description+conflict -> local JSON
            local/global -> independent summary JSON
            summary -> humor retrieval/pruning
Generator:  select conflicts -> select entities -> DFS -> sample path
            -> caption + explanation
```

任何只读取旧缓存 trace、跳过 summary/selection、最后只调用一次 Generator 的作业，都
不得命名为 HOMER public-code。`standard_description_replay` 仅是使用官方发布描述、
跳过在线 description 请求的明确 replay；`online_description` 才包含 description 请求。

### 4. Successor 数目不能把论文和代码混为一谈

论文正文把链长度描述为由 LLM 自适应、经验平均长度约 4（含 root）；官方 prompt 的
可执行文本要求每个 root 生成 3 个 successor。当前主基线固定 `root + 3`，这是为了
复现公开 prompt，不声称复现论文的 adaptive length distribution。针对本地 7B 的选择
另设预注册 pilot：`n ∈ {2, 3, 4, 5}`，使用同一验证图片、seed、retrieval 和成本预算；
不修改 `public_code_fixed_3` 主基线，也不在看结果后调参。

还要区分“递归语义”与“在线请求粒度”：论文以
\(e_{\tau+1}=f_{\rm chain}(e_\tau)\) 形式描述每一步依赖，但固定公开代码的两个
imagination 函数各自只发送一次请求，要求模型在同一个 JSON 响应中返回三个 successor。
因此当前 exact-public-code 轨道保留的是“一次请求返回三步列表”，不是三次 API 调用的
逐步递归。若以后要实现三次独立调用，必须命名为单独的 `paper_recursive_association`
变体，并重新计量成本与结果，不能混入主基线。

### 5. Conflict/entity 选择与随机路径

官方 Generator 先通过独立 API 请求选择两条 conflict 和两个关键 entity，再 DFS 枚举
路径并随机采样一条 path。当前实现保留两个 selection 请求，并用
`random.Random(seed)` 固定路径随机流，同时保存 seed、候选集合和最终路径。这是为了
可复现而做的显式适配；不能声称与官方未固定的进程级随机状态完全相同。

### 6. Summary 阶段

Summary 是独立 API 调用，不是 Python 清洗。当前 pipeline 在 retrieval 前强制完成
summary JSON，并在 call ledger 中记录 `imaginator.summary`。缺少 summary 的旧缓存路线
被排除在当前结果之外。

### 7. Caption 输出必须含 Caption 和 Explanation

官方 caption prompt 要求：

```text
##Caption:
##Explanation:
```

当前保存 raw response，并分别解析 `caption` 和 `explanation`；评测只把 caption 字段
送入 humor evaluator，explanation 仅用于 provenance/审计。论文形式化变量 Ω 没有在
公开 prompt 中给出完整离散词表，但本项目已按要求加入一个明确隔离、可复现的
`project_omega_grid_v1` 扩展（6 个 narrative strategies × 6 个 linguistic styles =
36 组合，定义见 `src/humor_generator_v35/homer/omega.py` 和
`configs/homer_public_code_pretrained_7b.yaml`）。它用于研究可控幽默策略，不能被称为
官方 Ω 枚举；`public_code_exact` 仍默认不注入 Ω，只有显式 `--omega NS|LA` 才启用。

### 8. 数据必须保留官方全人口通路

已增加 `scripts/build_homer_population_manifest.py`，生成初始库存
`manifests/homer_population_v1.json`；随后用
`scripts/create_homer_population_v2.py` 生成不可变 v2 数据版本。它不按
`first_rows_by_cluster()` 丢弃 caption，
而是逐一盘点 ranking CSV、图片、描述、评测 JSON 和 hash。当前观测到：

| 对象 | 当前可验证数量 | 解释 |
|---|---:|---|
| HIA 论文声明 | 365 contests、365 cartoons、平均 6,044 captions/contest | 论文/NeurIPS 数据人口 |
| 本地 HIA ranking CSV | 385 contests（510–895；其中 530–895 全部存在，510–529 有 19 个） | 发布目录人口，不等于论文 365 allow-list |
| 本地 HIA ranking caption 行 | 2,297,153 | 完整本地 ranking 目录扫描，不是 bridge 子集 |
| HOMER 发布 GPT-4o descriptions | 362 | train 271 + validation 44 + test 47；缺少 605、644、657、890 |
| HOMER HIA evaluator JSON | 385 | IDs 510–895 的发布评测文件 |
| Electronic Sheep evaluator JSON | 198 | 每项含 Group A/B 三条样本 |
| joke corpus | 335,569 records | 335,570 physical lines 含 CSV header |
| 历史 v3.5 bridge manifest | 2,846 rows / 810 clusters / 每 source 最多 3 条 | 仅机制 pilot，不可称 HOMER full population |

“365”“362”“385”分别表示不同的发布物/allow-list概念，禁止静默替换。当前 adapted
public-release 路线已用 `(dataset, contest_number, image_sha256)` source-aware key 封存
362 个可核验记录，数据门禁为 ready；不能把它改名为 canonical 365。canonical 365 仍须
等论文作者或官方发布物提供逐项 allow-list。旧 v2 的 878 hash 不一致已被隔离在父 lineage，
不再阻塞当前 362 版本。

检索通路也已单独对齐：当前 public-code 入口使用
`OfficialCodeRetrieval`，而不是历史 latent pilot 的 `HomerRetrievalAugmenter`。前者保留
官方 `imaginator.py` 的 regex 优先检索、单 query TF-IDF、名词/动词抽取、WordNet
相关性/频率/POS 评分和 entity-tree 构造；后者只能作为历史 latent 分支。由于官方未锁定
Python/NLTK/sklearn 版本，仍应称 `official-code-compatible`，不能称字节级 exact。

### 9. 当前模型路由：只用预训练 7B，只有 bridge 训练

`configs/homer_public_code_pretrained_7b.yaml` 和
`configs/bridges/pretrained_7b.yaml` 已锁定：

```text
Planner:   Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5
Generator: Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5
planner_adapter:  null
generator_adapter: null
planner_frozen:   true
generator_frozen: true
bridge_trainable: true
```

旧 SFT adapter 只作历史 provenance 审计，不进入当前 Planner trace、Generator forward
或 caption 结果。这种设计与“研究 latent communication 而非两个模型能力差异”一致。

### 10. Evaluator 型号和评测协议必须分轨

HIA NeurIPS 2024 的原始 Group 评测是：Group Overall 使用 GPT-4-Turbo + Hessel
description，Group Best Pick 使用 GPT-4o-vision + raw image；每组提供十条候选并采用
5-shot prompt。HOMER 论文另外以 GPT-5 作为 primary humor evaluator，5 个候选、
Pass@1/3/5、5 次重复，并报告 evaluator reliability。当前项目的 Group-of-10、镜像
A/B、三个独立多模态 judge、good/weak/bad 和 image-clustered bootstrap 是扩展评测，不能
冒充 HOMER 的 primary Pass@K。

官方评测 JSON 与 evaluator 脚本已下载并用 SHA-256 记录；
`scripts/verify_homer_evaluation_assets.py` 现在会 fail-closed 检查：

```text
HIA sample5gt.json: 385 records, groups 1-9/200-209/1000-1009
ES sampled3_evaluate_data.json: 198 records, groups A/B
HIA benchmark evaluator: GPT-4-Turbo (Overall), GPT-4o-vision (Best Pick)
HOMER primary evaluator: GPT-5（官方脚本 ID `gpt-5-chat-latest`）, temperature=0, 5 candidates, 5 repeats
```

HIA 原始 benchmark 还有一个容易混淆的独立协议：其论文从 358 个可用 contests
留出 91 个做评测，每个 contest 生成 10 条 caption，与四组人类提交（`#1-10`、
`#200-209`、`#1000-1009`、median group）进行 Group Overall/Best Pick；Group Overall
使用 GPT-4-Turbo + Hessel 描述，Group Best Pick 使用 GPT-4o-vision + 原图。该协议
不是 HOMER 论文的 GPT-5/5-candidate Pass@K 主协议，不能把两者的候选数、holdout 或
模型名称互换。

## 当前文件和验证结果

- prompt/stage smoke：`scripts/homer_public_code_smoke.py`，online 8 requests，
  standard-description replay 7 requests，阶段顺序和 Caption/Explanation 均通过；
- route check：`scripts/check_pretrained_route.py` 通过，旧 adapter 不允许进入主配置；
- evaluation asset check：`scripts/verify_homer_evaluation_assets.py` 通过；
- population inventory：v1 初始库存、v2 parent lineage 和新的 public-release 362 manifest
  均已生成；当前 adapted route 使用 `manifests/homer_population_public_release_362.json`；
- A4 旧 bridge 机制 pilot：已完成但 `pilot_inconclusive`，不能当作 HOMER 结果；
- 新 pretrained-only Planner trace：**尚未生成**，因此不能提交正式 bridge training。

## 修改后的执行顺序

1. 使用已封存的 public-release 362 allow-list；canonical 365 只在获得官方逐项清单后另行
   建立，不能用算术范围补齐。
2. 以 source-aware population manifest 验证所有图片、ranking、description 和 evaluator
   asset hash；当前 362 版本已通过该门禁。
3. 用预训练 7B、无 adapter、完整 HOMER stage order 重新生成 trace；每条 trace 保存
   prompt/model/config/seed/tensor hash。
4. 在 5–10 张代表性图片上做 zero-adapter online/replay smoke；通过后才进入 bridge
   机制 pilot。
5. 先以 HOMER-comparable GPT-5 + Pass@1/3/5/5 repeats 作为主表，再把当前
   Group-of-10/mirror/multi-judge 作为扩展表。
6. 只有 baseline 和数据门禁稳定后，才运行 latent bridge 与 successor pilot。

## 权威依据

1. Shang et al., *On the Wings of Imagination: Conflicting Script-based Multi-role
   Framework for Humor Caption Generation*, ICLR 2026：
   [论文 HTML](https://arxiv.org/html/2602.06423)，
   [官方实现](https://github.com/Shang-hub/HOMER-Official-Implementation)。
2. Zhang et al., *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks
   for Cartoon Captioning*, NeurIPS 2024：
   [论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf)。
3. Bai et al., *Qwen2.5-VL Technical Report*：
   [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)。
4. Hessel et al., *Do Androids Laugh at Electric Sheep?* ACL 2023：
   [ACL Anthology](https://aclanthology.org/2023.acl-long.41/)。
