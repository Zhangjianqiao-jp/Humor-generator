# HOMER 复现证据台账

参考版本：Shang et al., ICLR 2026，arXiv:2602.06423v2（2026-08-02）。

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
| Broad imagination | 对 backbone 每个节点检索 top-k jokes，候选词作为 leaf | tree expansion、逐节点检索、score 与 DFS 路径已实现；精确 tokenizer/lemmatizer 未披露，仍为复现变量 |
| Pruning | H_rel + H_freq + H_div | 公式已实现 |
| TSS | WordNet Wu-Palmer max | 已实现 NLTK adapter |
| CO | lexical-neighborhood Jaccard dissimilarity | 已实现 |
| Retrieval settings | k=5, delta=5 | 配置锁定 |
| Caption sampling | temperature=1 | 配置锁定 |
| Evaluation | n=5, pass@1/3/5, five trials | metric 与配置已实现 |

## 论文未充分公开，当前为硬 blocker

1. 论文未公布 situation-description 的逐字 prompt。v3.0 严格模式使用 benchmark 已有 standard description；不允许自创 prompt 后声称复现。
2. 未发现 HOMER 官方代码仓库。
3. 未发布清洗后的 335,570-joke corpus、数据版本与 hash。
4. 正文称 corpus 来自 12 个数据集，附录列出并称 11 个，存在内部不一致。
5. “共享 80% English words”的分母没有定义；本项目暂以 overlap coefficient 记录为显式假设。
6. `f_emb` 论文允许统计或 LM embedding，但没有给出主实验的确定实现；当前 sparse TF-IDF 只是待消融的可复现选择。
7. Qwen-VL (7B) 没有不可变 checkpoint revision，不能保证权重级复现。
8. `Omega in NS × LA` 的 narrative-strategy 与 language-style 取值集合未公开。

因此当前状态是：`publicly disclosed algorithm implemented; full result reproduction blocked by undisclosed artifacts`。这里不再写成“完整复现”，因为 tokenizer/lemmatizer、模糊实体合并、embedding backend 与语料本身都不是论文可恢复的确定实现。

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
