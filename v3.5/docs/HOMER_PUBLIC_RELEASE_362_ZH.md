# HIA 公共发布版数据门禁（362 contests）

更新时间：2026-09-06

## 结论

原来的门禁阻塞是合理的：HOMER 论文写的是 365 个 HIA contests，但公开来源没有提供一个可逐项核验的 365-ID allow-list。直接把 `530..895`、本地目录全集或旧的 representative rows 当作 365 人口，会把论文计数、文件库存和实际 description population 混在一起。

现在已解决“可复现公共数据”的阻塞，方法是建立一个单独的数据版本，而不是篡改旧 manifest：

```text
data_version: homer_pretrained_7b_public_release_362
population_status: verified_public_release_362
contest IDs: 362
split: train 271 / validation 44 / test 47
source caption rows: 362 × 3 = 1,086
ID digest: 02e6596e8745569fe944a4a1a846a355758fc11230398280c72d7bfc4b931c62
```

该版本可以支持 **adapted public-release evaluation**。它不能被称为 canonical 365-contest HOMER evaluation；canonical 365 gate 仍明确为 `false/unresolved`，直到论文作者或官方发布物提供可核验的 365-ID 清单。

## 可核验来源

1. 公开数据集：`yguooo/newyorker_caption_ranking`，固定 revision
   `1cd70477b6a99a473690a25a2fed359f75184c64`（Hugging Face API 返回的完整 40 位
   commit；此前的 39 位近似字符串不是有效 revision，已在本版本修正）。
2. 本地固定描述来源：HOMER 官方实现 commit
   `d1334f295cc1a8f8f6dc67ba7e846c5939dddcec` 的
   `data/datasets/humorbench/gpt4o_description/{train,validation,test}.jsonl`。
3. 三个描述文件的记录数和 SHA-256：

   | split | rows | SHA-256 |
   |---|---:|---|
   | train | 271 | `09151b799306b4dd2f6bbd5e67657cf988a4a5a96639fceba7625ad5cb8d9602` |
   | validation | 44 | `80d91d34dac4a00a0e976983329c94844ba05ec0372306bd80af1524815f7171` |
   | test | 47 | `b7d9ff114f684d77bcf923780ba59ca67f0346ce90d551b4f52e218156273e8c` |

这些文件的 ID 并集是 362 个；在论文给出的数字范围 530–895 中缺少
`605, 644, 657, 890`。这不是推测出来的数字，而是从固定发布文件逐行读取、去重并做 digest 后得到的结果。

## 生成和验证

```bash
cd /home/pj26000152/ku60000936/projects/Humor-generator/v3.5

# v2 先固定 878.jpg 的历史 hash 与当前本地字节的 lineage
.venv/bin/python scripts/create_homer_population_v2.py
.venv/bin/python scripts/verify_homer_population_v2.py

# 从固定公开 description split 构造 362-ID source-aware 版本
.venv/bin/python scripts/create_homer_public_release_362.py
.venv/bin/python scripts/verify_homer_public_release_362.py

# 静态模型/协议 + 公共人口门禁
.venv/bin/python scripts/check_pretrained_route.py

# bridge 训练还必须等当前预训练 Planner 的 hidden-state trace 建好
.venv/bin/python scripts/check_pretrained_route.py --require-data-ready
```

生成的 ignored artifacts 位于
`data/processed/homer_pretrained_7b_public_release_362/`，包括：

- `population_rows.jsonl`：每个 contest 一行，正式在线 HOMER 输入；
- `source_rows.jsonl`：每个 contest 的全部三条源 caption，共 1,086 行；
- `train.jsonl`、`validation.jsonl`、`test.jsonl`：严格按公开 description split 的一行/contest 输入；
- `official_hia_unseen_test.jsonl`：旧工具名的显式、记录级等价 alias；
- `trace_inputs.jsonl`：362 行当前预训练 Planner 输入，尚不包含 hidden states。

allow-list 本身位于
`manifests/homer_public_release_362_allowlist.json`，完整 lineage 位于
`manifests/homer_population_public_release_362.json`。验证器会检查 manifest 自哈希、allow-list 哈希、description 文件哈希、图片和 ranking CSV 哈希、split 计数、三条 source caption/contest 以及 test alias；任何一项变化都会失败。

## 门禁语义

`scripts/check_pretrained_route.py` 现在返回两层状态：

```text
data_gate       = ready       # 362 公共人口可复现，允许在线 public-code baseline
trace_gate      = ready       # adapter-free Planner hidden-state trace 已完成
context_gate    = blocked     # post-trace summary/retrieval/selection 尚未封存
bridge_data_gate= blocked     # bridge view 尚未构建，因此暂不训练 bridge
```

这避免了两个相反错误：

1. 因为没有 canonical 365 清单而永远不能运行可复现的公共发布版 baseline；
2. 因为人口已就绪，就错误地把 planner-input 当成 hidden-state trace，绕过 bridge 训练门禁。

在线 `generate_homer_public_code.py` 的默认 dataset 已切换到这个 362 版本，并新增 `--split test`；它仍然拒绝历史 `latent_bridge_v35` representative-row 数据。

下一步是用 `trace_inputs.jsonl` 重新生成当前无 adapter 的 Qwen2.5-VL-7B Planner trace，并在 trace 完整性通过后才运行 bridge-only 训练。专用入口是
`scripts/cache_pretrained_homer_traces.py`，验证器是
`scripts/verify_pretrained_homer_traces.py`，作业模板是
`jobs/cache_pretrained_homer_traces.pjm`。该步骤不改变 362 人口，也不把它重新命名成 365。

trace 记录使用独立的 `data/cache/homer_pretrained_7b_planner_traces/` 目录，明确保存
`planner_adapter=null`、模型 revision、输入 manifest hash、官方 prompt hash、模型
manifest hash、Git commit、每通道生成 seed、原始文本、`token_ids`、hidden states 及
generation/replay alignment。所有三通道必须通过严格 schema；任何失败都会留在
`failures.json`，不会被补写或静默跳过。

## 数据质量与已知上游问题

图片和 ranking 文件使用当前本地公开源字节并记录 SHA-256；不对上游图像进行静默修复。NextML 的公开 issue 记录了 605/607、644/645、656/657 等历史数据问题，因此本版本只按固定 description allow-list 选取发布记录，保留原始字节和 issue 链接作为 provenance，不手工“修正”内容。

## 不能这样做

- 不能用 `530..895` 的算术范围补出三个缺失 ID；
- 不能把旧 `latent_bridge_v35` 的 3,633 行复制结果当成 HOMER population；
- 不能把 362 结果放入论文的 canonical 365 表格；
- 不能在 `trace_gate=blocked` 时把 `trace_inputs.jsonl` 当作 hidden-state `trace_index`；
- 不能修改旧 v1/v2 manifest 的 hash 来“凑”365。

## 权威参考

- HOMER 论文：[arXiv HTML](https://arxiv.org/html/2602.06423)；
- HOMER 官方实现：[固定 commit 的 GitHub 仓库](https://github.com/Shang-hub/HOMER-Official-Implementation)；
- HIA 数据集：[Hugging Face 固定发布](https://huggingface.co/datasets/yguooo/newyorker_caption_ranking)；
- HIA 基准论文：[NeurIPS 2024 数据与基准](https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf)；
- 上游数据问题记录：[caption-contest-data-api issue #38](https://github.com/nextml/caption-contest-data-api/issues/38)。
