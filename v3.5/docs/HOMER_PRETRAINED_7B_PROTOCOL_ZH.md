# v3.5 预训练 7B + 仅 Bridge 的 HOMER 协议（当前执行版）

更新时间：2026-09-06

这份文档根据本轮逐项注释审计后建立，覆盖当前主路线；旧的 SFT-receiver、DPO 和
缓存后单次 generator 路线不属于本协议，也不作为本路线的结果来源。

## 结论先行

当前不能诚实地声称“100% 复现 HOMER”。原因不是算法概念不对，而是公开证据本身
不够使这个声明成立：

1. HOMER 论文只写 `Qwen-VL (7B)`，没有公开不可变的权重 revision；官方仓库的
   Extractor/Imaginator/Generator 代码实际调用 `gpt-4o` API。
2. 论文附录 B.12 把成本表写成 `Extractor 2 + Imaginator 3 + Generator 2 = 7`，
   但固定 commit 的 `generator.py` 明确执行“选 conflict、选 entity、生成 caption”
   三个请求，所以在线完整代码路径是 8 个请求（若使用已发布 description，跳过
   description 请求后是 7 个）。该矛盾必须保留在 provenance，不能靠改名消失。
3. HIA 论文宣称 365 contests，但当前公开、可固定 revision 的 GPT-4o description
   release 是 362 条；numeric contest 范围与 allow-list 也不能直接互换。本项目已经
   封存 `homer_pretrained_7b_public_release_362` 作为 adapted route 的机器可核验人口，
   而把缺少作者清单支撑的 canonical 365 作为独立 unresolved claim。它不再阻塞
   adapted route，但在获得权威 365-ID 清单前不得使用 canonical 365 标签。

因此本项目建立两个明确隔离的轨道：

| 轨道 | 目的 | 模型 | 能否称论文权重级 exact |
|---|---|---|---|
| `public_code_exact` | 逐字使用官方仓库 prompt、阶段和请求结构；需要 OpenAI 兼容后端 | 官方 `gpt-4o`（若按官方运行） | 不能，论文权重未公开；可称 public-code reproduction |
| `pretrained_7b_bridge`（当前主线） | 在同一 HOMER 阶段/数据/评测协议下测试 bridge 是否有增益 | Planner 与 Generator 都是本地 `Qwen2.5-VL-7B-Instruct@cc5948…`，无 adapter | 不能；称 adapted pretrained-7B protocol |

主线只训练 bridge：

\[
\theta_P=\theta_G=\theta_{\text{Qwen2.5-VL}}\quad\text{冻结},
\qquad \phi=\text{bridge 参数可训练}.
\]

旧 `planner_sft`/`generator_sft` 仍可用于历史 artifact 审计，但不得进入本路线的
Planner trace、Generator forward 或 caption 结果。

“模型可以变”只表示可以增加独立的待测 `model_variant`；每个 variant 都必须从自己的
固定 model/revision、完整 trace 和相同评测协议开始，不能跨模型复用候选或合并分母。
HOMER 论文 Table 3 比较 GPT-4o、Claude-4、Qwen-VL 和 LLaVA-1.5，说明这种跨模型对照
是合理的；当前主线先固定本地 Qwen2.5-VL-7B，其他模型作为单独对照轨道。

## 1. 与 HOMER 的逐阶段对齐

固定顺序为：

```text
Extractor:
  image -> vivid description
  description -> conflict scripts

Hierarchical Imaginator:
  image + conflict -> global imagination JSON
  description + conflict -> local imagination JSON
  description + conflict + two JSONs -> summary/merge JSON
  summary -> offline humor retrieval / pruning

Generator:
  description + conflicts -> select two conflict scripts
  selected conflicts + entity list -> select two entities
  DFS paths -> seeded random path per selected entity
  description + selected conflict + selected paths (+ optional Ω) ->
  ##Caption + ##Explanation
```

官方公开仓库的原始 prompt 已逐字放入
[`src/humor_generator_v35/homer/official_prompts.py`](../src/humor_generator_v35/homer/official_prompts.py)。
历史 `homer/prompts.py` 保持不变，避免旧 artifact 的 prompt hash 被事后重写。

`HomerPublicCodePipeline` 在
[`src/humor_generator_v35/homer/public_code_pipeline.py`](../src/humor_generator_v35/homer/public_code_pipeline.py)
中执行上述顺序，并保存每个阶段的 seed、温度、token budget 和请求日志。

### Description 的两种入口

- `online_description`：运行官方 description prompt，在线请求数为 8；用于新图/完整
  代码轨道。
- `standard_description_replay`：直接读取 HOMER 发布的 benchmark description，跳过
  description API，请求数为 7；适合与发布 benchmark 的固定输入做可重复比较，但
  必须标注为 replay，不能伪装成 online extractor。

两者不得混在同一个统计单元中。

## 2. 模型与训练边界

配置：

- [`configs/homer_public_code_pretrained_7b.yaml`](../configs/homer_public_code_pretrained_7b.yaml)
- [`configs/bridges/pretrained_7b.yaml`](../configs/bridges/pretrained_7b.yaml)
- [`configs/pilot/cross_attention_caption_pretrained.yaml`](../configs/pilot/cross_attention_caption_pretrained.yaml)

硬约束：

```text
Planner base:    Qwen/Qwen2.5-VL-7B-Instruct
Generator base:  Qwen/Qwen2.5-VL-7B-Instruct
Revision:        cc594898137f460bfe9f0759e9844b3ce807cfb5
Planner adapter: null
Generator adapter:null
Policy params:   frozen
Trainable:       bridge only
```

Qwen2.5-VL 技术报告证明该模型具有通用视觉理解、定位和长视觉上下文能力，但没有
证明它在 New Yorker 幽默上达到 GPT-4o。HOMER Table 3 则直接报告了 `Qwen-VL (7B)`
在 HIA/ES 上的非零 Pass@K，但明显低于 GPT-4o；这说明“7B 完全不能做幽默”没有
文献依据，同时“只靠预训练 7B 就足以达到 SOTA”也没有依据。故主线采用预训练 7B
作为无 SFT 混杂的严谨基线，并在训练前做 5–10 张图的 zero-adapter smoke。

## 3. Successor 数目

HOMER 论文正文描述 successor 链长度由 LLM 自适应，经验平均约 4（含 root）；固定
公开代码的 prompt 则要求每个 entity 生成 3 个 successor。当前主基线严格使用：

```text
public_code_fixed_3 = root + 3 successors
```

这不是把“3”误写成论文自适应结果。若要为本地 7B 找合适数目，先做低成本 pilot：

```text
n ∈ {2, 3, 4, 5}
5–10 张 validation 图片；相同图片、seed、检索和评测预算
```

只依据预注册的 caption 质量、grounding、有效 JSON 率和 token/时间成本选择扩展
配置；不改变 `public_code_fixed_3` 主基线，不在看到结果后调 n。

## 4. 随机选择与 summary

官方 Generator 会选择两条 conflict、两个实体，DFS 枚举路径后随机选择一条路径。
当前实现用 `random.Random(seed)` 复现这种选择，同时把 seed、候选全集和最终路径写进
输出；这样比使用进程级未设 seed 的 `random.choice` 更可复现，但会标记为
`seeded-random adaptation`，而不是声称官方运行的随机状态完全相同。

Summary 不是可选的清洗步骤。必须先完成 local/global 两路 JSON，再运行官方独立
summary/去重请求，然后才做 humor retrieval/pruning。任何只读取缓存 trace 后直接
调用 Generator 的作业都不能命名为 `HOMER public-code`。

## 5. Caption 输出与 Ω

官方代码的 caption prompt 要求：

```text
##Caption:
##Explanation:
```

因此输出必须保存 raw blob、解析后的 caption 和 explanation；评测时只将 caption
字段送入 humor evaluator，解释字段用于审计，不得混入候选文本。

论文形式化地写了：

\[
\Omega\in NS\times LA,
\]

其中 Ω 表示 narrative strategy 与 linguistic style。这里必须区分“论文中的形式化变量”
和“官方代码公布的离散词表”：论文/附录讨论了 GTVH 资源、叙事策略和风格控制，但没有
给出一个可逐项核验的有限 `NS × LA` 词表；固定 public prompt 也没有 Ω 参数。因此本
项目现在**有 Ω**，但它是显式、版本化的项目扩展，而不是冒充官方枚举：

```text
variant: project_omega_grid_v1
NS = {setup_punchline, deadpan_observation, question_answer,
      dialogue_voice, role_reversal, exaggerated_consequence}
LA = {plain_wit, pun_wordplay, idiom_twist, sarcastic_understatement,
      personification, metaphorical_comparison}
```

配置固定 36 个笛卡尔积组合、稳定顺序、prompt 文本和 provenance 字段。示例：

```bash
.venv/bin/python scripts/generate_homer_public_code.py \
  --config configs/homer_public_code_pretrained_7b.yaml \
  --omega 'setup_punchline|pun_wordplay' \
  --split validation --seeds 20260906 --output outputs/omega_smoke.jsonl
```

省略 `--omega` 才是 `public_code_exact` baseline；启用它的结果必须命名为
`project_omega_grid_v1`，单独报告并记录 Ω hash/seed，不能写成“官方 HOMER 已使用这套
词表”。这样既保留 Ω 控制，又不把不可核验的论文符号变成伪官方标签。

## 6. 数据通路

主评测使用官方发布资产与 hash：

- HIA GPT-4o standard description：train 271、validation 44、test 47（当前固定公开
  release 共 362 条；canonical 365 仍是单独 unresolved claim）；数据版本为
  `homer_pretrained_7b_public_release_362`，来源 revision
  `1cd70477b6a99a473690a25a2fed359f75184c64`；
- Electronic Sheep description：679 条；
- HOMER joke corpus：官方 CSV 物理行 335,570（含表头），实际 CSV records 335,569，
  SHA-256 `681059f0…d7d31`；
- HIA ranking/source：必须保留完整 caption population 和官方排名，不再使用“每个
  source 最多 3 条”作为 HOMER full-data 的无说明替代；任何 bridge pilot 子集都要在
  manifest 中显式写 sampling policy、row count、hash。

新版本 `data/processed/homer_pretrained_7b_v2` 复制并修复了历史行的字节 provenance；
`data/processed/latent_bridge_v35`（每 source 最多 3 条）仍只能作为历史 bridge 机制
pilot，不能充当 HOMER population。当前在线 public-code route 使用
`homer_pretrained_7b_public_release_362`：它以公开 HIA description release 的固定 revision
为 allow-list，按 `(dataset, contest_number, image_sha256)` 建立 population key，并保留
每个 contest 的三条源 caption。它是可复现的 **adapted 362-contest release**，不是论文
365-contest 的 canonical claim。v2 的 878 hash lineage 作为父版本保留，未原地改写旧
manifest；细节见 `docs/HOMER_PUBLIC_RELEASE_362_ZH.md`。

## 7. 评测协议

### HOMER-comparable 主轨道

严格使用论文披露的：

```text
HIA groups: #top10, #200–209, #1000–1009
Electronic Sheep: High-Humor / Low-Humor
5 candidates per image
Pass@1/3/5
5 repeated trials
caption temperature=1.0
primary evaluator=GPT-5 (official script model ID `gpt-5-chat-latest`), evaluator temperature=0
```

这与 HIA NeurIPS 2024 原始的 Group Overall/Best Pick evaluator 不同；后者只能作为
benchmark compatibility analysis（原 HIA 使用 GPT-4-Turbo 与 GPT-4o-vision）。所有
model ID、snapshot/date、prompt hash 和 API provenance 都要保存。主轨道默认使用
`gpt-5-chat-latest`（论文标签 GPT-5）。如果部署只能使用其他评审模型，允许做**预先
声明的 evaluator substitution**，但必须保持候选数、Pass@K、重复次数、参考组、温度和
评审 prompt 不变，并在配置、结果和表格中写出实际 model ID、snapshot/date 以及
`evaluator_substitution=true`；这类结果只能标为 HOMER-protocol adapted evaluation，
不能标为 canonical GPT-5 结果。

### 项目扩展轨道

现有 Group-of-10、镜像 A/B、三个独立多模态 judge、good/weak/bad、image-clustered
bootstrap 和 seed variance 继续保留，但单独命名为 `project_extension`，不与 HOMER
Pass@K 混算。

## 8. 执行门禁（当前不提交正式 GPU）

顺序固定为：

1. `official_prompts.py` 常量/哈希单元测试；
2. `HomerPublicCodePipeline` CPU fake-backend smoke，断言 7/8 请求数、阶段顺序、
   summary、selection、DFS、caption/explanation schema；
3. 运行 `scripts/create_homer_population_v2.py` 和
   `scripts/create_homer_public_release_362.py` 建立不可变父/当前 data version，并运行
   两个 verifier（不修改旧 v1 或旧 JSONL）；
4. 运行 `scripts/check_pretrained_route.py`：人口 `data_gate` 必须为 `ready`；
5. 5–10 图的本地预训练 7B zero-adapter smoke；
6. 按 `trace_inputs.jsonl` 生成当前预训练 Planner hidden-state traces，只有
   `trace_gate=ready` 后才提交单 GPU bridge-only pilot；
7. semantic/caption gate 通过后再扩展 successor pilot、完整 bridge 训练和最终评测。

当前已经完成的是代码与协议隔离以及 362-contest public population gate，不是新的训练
结果。不得把旧 SFT trace、旧 A5 caption 或旧 Group-of-3 结果移植到本路线。

`scripts/check_pretrained_route.py` 将在线基线和 bridge 输入拆成四个独立门禁：
`data_gate` 检查固定的 public population，`trace_gate` 检查 adapter-free Planner
hidden-state trace，`context_gate` 检查 362 条 summary/retrieval/selection 后处理上下文，
`bridge_data_gate` 还要求 source-caption bridge training view 存在并指向同一 context
hash。在线 public-code baseline 只需要 `data_gate`；`--require-data-ready` 要求
population 与 trace；bridge 作业必须显式使用 `--require-bridge-data-ready` 或等价的
`verify_pretrained_bridge_inputs.py`。这样既能在不占 GPU 的情况下运行文本基线，也不会
把 planner-input 清单、未完成 context 或旧 bridge 数据冒充可训练输入。

## 权威依据

1. Shang et al., *On the Wings of Imagination: Conflicting Script-based Multi-role
   Framework for Humor Caption Generation*, ICLR 2026：
   [arXiv HTML](https://arxiv.org/html/2602.06423)，[OpenReview](https://openreview.net/forum?id=SzaRhPom4o)。
2. HOMER 官方公开实现（固定 commit）：
   [GitHub](https://github.com/Shang-hub/HOMER-Official-Implementation)。
3. Zhang et al., *Humor in AI*, NeurIPS 2024：
   [论文与数据基准](https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf)。
4. Bai et al., *Qwen2.5-VL Technical Report*：
   [arXiv](https://arxiv.org/abs/2502.13923)。
5. Hessel et al., *Do Androids Laugh at Electric Sheep?*, ACL 2023：
   [ACL Anthology](https://aclanthology.org/2023.acl-long.41/)。
