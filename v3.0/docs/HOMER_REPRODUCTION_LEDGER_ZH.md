# HOMER 复现证据台账

参考版本：Shang et al., ICLR 2026，arXiv:2602.06423v2（2026-08-02）；官方实现固定到 commit `d1334f295cc1a8f8f6dc67ba7e846c5939dddcec`。

## 论文明确公开，已实现

| 项目 | 论文设定 | v3.0 |
|---|---|---|
| 角色 | Extractor / Hierarchical Imaginator / Generator | 已分离 |
| Conflict prompt | Appendix prompt 1 | 逐字保存 |
| Local/Global prompt | Appendix prompt 2 | 逐字保存 |
| Caption prompt | Appendix prompt 3 | 逐字保存并使用 system role |
| Conflict 数量 | two or more | 严格 `>=2` |
| Association | 每个 root 三个、逐步依赖 | 严格 JSON 与长度 3 |
| Local view | standard description | 已实现 |
| Global view | image | 已实现 |
| Deep imagination | first-order chains，经验均长约 4（含 root） | root + 3 steps |
| Broad imagination | 对 backbone 每个节点检索 top-k jokes，候选词作为 leaf | 官方代码确认 `TfidfVectorizer(max_features=1000, stop_words=english, ngram_range=(1,3))`，并使用 NLTK tokenizer/POS/WordNet lemmatizer；环境中的精确 NLTK resource revision 仍需记录 |
| Pruning | H_rel + H_freq + H_div | 公式已实现 |
| TSS | WordNet Wu-Palmer max | 已实现 NLTK adapter |
| CO | lexical-neighborhood Jaccard dissimilarity | 已实现 |
| Retrieval settings | k=5, delta=5 | 配置锁定 |
| Caption sampling | temperature=1 | 配置锁定 |
| Evaluation | n=5, pass@1/3/5, five trials | metric 与配置已实现 |

## 本轮找到并固定的公开数据源

### 1. Qwen-VL revision：仍未公开

论文只写 `Qwen-VL (7B)`。官方仓库的 extractor、imaginator、generator 和 demo 实际均通过 OpenAI API 使用字符串 `gpt-4o`，没有 Qwen loader、Hugging Face model ID、权重 hash 或 revision。因此不能从公开证据确定论文中 Qwen-VL baseline 到底是 `Qwen-VL`、`Qwen-VL-Chat` 的哪一个提交或快照，更不能把当前项目的 Qwen2.5-VL 当成同一 revision。

结论：`model_revision: null` 保持为硬 blocker；任何猜测只能作为新 baseline，不能标为 HOMER 精确复现。

### 2. benchmark standard-description：来源已确定

HOMER 官方仓库在固定 commit 中发布：

| 数据 | 上游文件 | 记录数 | SHA-256 |
|---|---|---:|---|
| Humor in AI train | `data/datasets/humorbench/gpt4o_description/train.jsonl` | 271 | `09151b799306b4dd2f6bbd5e67657cf988a4a5a96639fceba7625ad5cb8d9602` |
| Humor in AI validation | `data/datasets/humorbench/gpt4o_description/validation.jsonl` | 44 | `80d91d34dac4a00a0e976983329c94844ba05ec0372306bd80af1524815f7171` |
| Humor in AI test | `data/datasets/humorbench/gpt4o_description/test.jsonl` | 47 | `b7d9ff114f684d77bcf923780ba59ca67f0346ce90d551b4f52e218156273e8c` |
| Electronic Sheep | `data/datasets/electronic_sheep/description/description.jsonl` | 679 | `05f3c96a721c111f5d01fd6bc253f849e392db8b120e90817b26e30cd44b0597` |

Humor in AI 三个文件与本项目 v2.5 已下载的 `gpt4o_description` 文件逐字节一致。其原始 benchmark 是 `yguooo/newyorker_caption_ranking`，数据卡标注 CC-BY-NC-4.0。Humor in AI 论文说明这些结构化描述由 GPT-4o 在固定 five-shot examples 下生成。

Electronic Sheep 的 679 条 `canny` 字段来自该 benchmark 的人工 MTurk `image_description`。逐 contest 对照原始 annotation 后，679/679 都精确匹配三个描述者中的一个；但官方文件没有公开选择三者之一的规则，因此严格复现应直接使用 HOMER 发布的选择结果，不自行重选。

注意：官方 `extractor.py` 另公开了用于新图片的 `##Vivid Description:` prompt，模型为 `gpt-4o`、temperature 1.0、image detail high。它和已发布 benchmark JSONL 的结构并不相同；正式 benchmark 复现以固定 JSONL 为准，demo/新图片才使用 extractor prompt。

### 3. 清洗笑话语料：文件已公开，但计数有一条差异

官方文件：`data/Our_joke_database/humor_rag_large_database.csv`

- 固定 commit：`d1334f295cc1a8f8f6dc67ba7e846c5939dddcec`
- 文件大小：38,530,985 bytes
- SHA-256：`681059f010868c1021eeb9150828536b9cfe99ad5d81288acd98efb2c19d7d31`
- 物理行数：335,570（包括表头 `ID,Joke`）
- 实际 CSV 数据记录：335,569
- 唯一 ID：335,569
- 唯一完整 joke 字符串：335,569
- 空 joke：0

所以论文/README 的“335,570 jokes”与发布文件存在 off-by-one：它等于含表头的物理行数。复现必须保留官方文件和 hash，并在报告中披露实际样本数 335,569，不能修改数据以凑数。

完整机器可读清单位于 `manifests/homer_official_assets.json`；下载脚本为 `scripts/fetch_homer_official_assets.py`。由于 HOMER 官方仓库固定 commit 没有 LICENSE/COPYING 文件，原始资产只下载到 git-ignored 的 `data/external/`，不重新发布到本项目仓库。

## 论文/发布物仍未充分公开的 blocker

1. Qwen-VL (7B) 没有不可变 checkpoint revision，不能保证权重级复现。
2. 正文称 corpus 来自 12 个数据集，附录列出并称 11 个，存在内部不一致；发布 CSV 也没有逐条 source 字段。
3. 发布仓库没有独立 corpus cleaning manifest，无法逐步从原始 11/12 个源重建同一 CSV；当前只能对最终发布 artifact 做 byte-level 复现。
4. `Omega in NS × LA` 的 narrative-strategy 与 language-style 取值集合未公开。
5. 官方仓库没有 LICENSE/COPYING，二次分发权限不明确。

因此当前状态更新为：`official data artifacts and clean implementation pinned; weight-level result reproduction still blocked by undisclosed Qwen-VL revision`。

## 明确属于本项目扩展，不属于 HOMER

- StateBridge baseline；
- Learned/Typed latent bridge；
- text-teacher KL；
- matched/shuffled semantic margin；
- Base/SFT receiver 双桥实验；
- latent communication 的 Group-of-3 盲评。

这些实验只能在 HOMER 文本 baseline 门禁通过后启用，并必须以 extension/ablation 命名。

## 参考论文

1. Shang et al. *On the Wings of Imagination: Conflicting Script-based Multi-role Framework for Humor Caption Generation*. ICLR 2026. arXiv:2602.06423.
2. Zhang et al. *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*. NeurIPS 2024. arXiv:2406.10522.
3. Hessel et al. *Do Androids Laugh at Electric Sheep?* ACL 2023. ACL Anthology 2023.acl-long.41.
4. Du et al. *Enabling Agents to Communicate Entirely in Latent Space*. ACL 2026. ACL Anthology 2026.acl-long.1248.
5. Peng et al. *StateBridge*. COLM 2026. arXiv:2608.13317.

## 权威公开入口

- HOMER 论文：https://openreview.net/pdf?id=SzaRhPom4o
- HOMER 官方实现：https://github.com/Shang-hub/HOMER-Official-Implementation
- Humor in AI 论文：https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
- Humor in AI 数据：https://huggingface.co/datasets/yguooo/newyorker_caption_ranking
- Electronic Sheep 论文：https://aclanthology.org/2023.acl-long.41/
- Electronic Sheep 官方数据实现：https://github.com/jmhessel/caption_contest_corpus
