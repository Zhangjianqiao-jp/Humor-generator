---
date: 2026-07-29
project: HIC humorous image captioning
tags:
  - humor-captioning
  - multimodal
  - qwen-vl
  - post-training
  - experiment-log
  - context-restore
---

# HIC Humor Captioning 本轮上下文与失败复盘

## 一句话任务目标

本项目的本质目标是：围绕 HIC / Bokete 风格图像幽默 caption 生成，建立一条可验证的后训练路线，让 `Qwen2.5-VL-3B-Instruct` 能生成更短、更贴图、更好笑、少解释腔的 caption。

当前最重要的问题不是“继续堆训练量”，而是判断哪种结构化幽默信号、哪种训练目标、哪种评估方式最能提升真实幽默生成质量。

## 当前研究路线

已经收敛到的主线是：

```text
image + compact humor context -> Qwen2.5-VL-3B caption generator
```

其中 compact humor context 暂时采用：

```text
hic-compact-json
```

这个 context 由 teacher/analyzer 从：

```text
image + gold caption
```

中提取出来，包含 humor point、visual anchors、viewpoints、primary viewpoint 等结构化字段。

重要定位：

```text
hic-compact-json 当前是 gold-caption-derived upper-bound 方法
```

它能回答：

> 如果模型知道 gold caption 暗示的结构化笑点，它能不能学会生成更像好 caption 的输出？

它不能直接回答：

> 没有 gold caption 时，模型是否能自己看图找笑点？

因此后续必须补 image-only humor hypothesis extractor 或 generate-and-rank pipeline。

## 本轮已经完成的工作进度

### 1. 明确评估原则

用户明确指出：判断 caption 是否好笑，必须参考数据集的 `gold caption`，不能只凭模型或人工主观感觉随意判断。

因此后续测试不再只看“输出是否像幽默解释”，而是至少要看：

- 是否贴近 gold caption 的幽默点；
- 是否短、自然、像 caption；
- 是否避免解释腔；
- 是否避免拒答；
- 是否 grounded 到图片内容；
- 多候选中是否有接近 gold humor 的候选。

### 2. Viewpoint taxonomy 收敛

通过一轮大范围分类，最终保留了 8 个 viewpoint：

```text
face_expression_crop
relation_crop
context_scene_view
text_region_crop
object_crop
full_image
pose_action_view
scale_reference_crop
```

这些 viewpoint 的作用不是直接生成 caption，而是回答：

> 理解 gold caption 的笑点，最小需要看图片中的哪类视觉区域？

这为后续图片内标注、crop/region prompt、image-only extractor 提供了分类基础。

### 3. Prompt 方案选择

比较过多种结构化笑点表达：

```text
plain
hic-humor-point
hic-viewpoint-tags
hic-anchor-viewpoint
hic-compact-json
```

当前选择：

```text
hic-compact-json
```

选择理由：

- 比自然语言解释更短；
- 字段稳定，便于训练；
- 更不容易诱导模型输出 “because / humor / contrast / mismatch” 这类解释腔；
- 能同时承载 humor point、anchor、viewpoint；
- 比 few-shot 更适合后续 SFT / DPO / reranking 数据构造。

### 4. Prompt style fix

早期 `hic-compact-json` prompt 会诱导模型输出解释说明，例如：

```text
because
this image
the humor is
contrast
mismatch
visual effect
```

后来 prompt 被改成严格 caption style：

- compact JSON 只是 clue，不是输出模板；
- 输出 exactly one caption；
- 最大约 12 words；
- 不要解释；
- 不要输出 JSON；
- 不要输出分析标签；
- 最终基础指令保持：

```text
Generate one short, natural, image-specific humorous caption for this image. Do not explain.
```

相关文件：

```text
src/analysis/guided_prompting.py
tests/test_hic_region_guided_prompting.py
```

### 5. 512 pilot LoRA 已证明方向有效

512 train / 128 val 的 `hic-compact-json` LoRA pilot 已完成。

输出目录：

```text
outputs/lora_sft_hic_compact_json_pilot_512/final_lora
outputs/lora_sft_hic_compact_json_pilot_512/best_val_loss
```

核心结果：

```text
steps: 64
train_loss: 2.257
eval_loss: 2.1428
eval_ppl: 8.52
best_val_loss: 2.1428 at step 64
```

held-out 200 evaluation 中，512 LoRA 相比 base hic-compact-json prompt 有明显提升：

```text
method                    gold_match  candidate_match  avg_max_sim  format_ok  avg_chars
base hic-compact-json        0.0550          0.0181       0.3768     0.9806       59.6
512-row LoRA pilot           0.1200          0.0694       0.4232     0.9963       28.7
```

解释腔下降：

```text
explains flag:        0.0194 -> 0.0031
generic-pattern flag: 0.0425 -> 0.0169
```

但拒答略有上升，需要继续监控。

### 6. 3000 档的决策

本地服务器上没有完整 3000 context / adapter 的可验证产物：

```text
outputs/analysis/hic_humor_viewpoints_sft_train_pilot_3000.jsonl: 1 line
outputs/lora_sft_hic_compact_json_pilot_3000: no checkpoint
```

但用户反馈自己已经用模型测过 `hic-compact-json-train-3000`，效果很好。

基于用户的外部测试结果，本轮决定：

```text
可以进入下一步，不必再在本地重复 3000 pilot
```

同时保留一个判断：

```text
不建议无脑直接全量训练，但如果 3000 已验证很好，可以把全量训练作为 upper-bound 后台实验。
```

### 7. Full run v2 脚本已升级

对 full run 支持做了脚本升级：

```text
scripts/analyze_hic_humor_viewpoints.py
scripts/run_hic_compact_json_pilot_train.sh
```

关键能力：

- `--limit 0` 或负数表示全量；
- context 生成可断点续跑；
- `OVERWRITE_CONTEXT=0` 避免覆盖已有 context；
- `CONTINUE_ON_OOM=1` 支持 OOM 后跳过继续；
- `RESUME_FROM_CHECKPOINT` 支持从 checkpoint 恢复；
- `LOGGING_STEPS` / `EVAL_STEPS` / `SAVE_STEPS` / `SAVE_TOTAL_LIMIT` 可通过环境变量覆盖；
- full run 不再沿用 pilot 的过密 eval/save cadence。

目标 full run：

```text
tmux session: hic-compact-json-train-full
train_context: outputs/analysis/hic_humor_viewpoints_sft_train_full.jsonl
val_context: outputs/analysis/hic_humor_viewpoints_sft_val_full.jsonl
output_dir: outputs/lora_sft_hic_compact_json_full_v2
```

### 8. GitHub 状态

当前本地仓库：

```text
/home/zhang.jianqiao/projects/Humor-generator
```

当前分支：

```text
codex/hic-viewpoint-ablation
```

本轮核心提交已经在远端：

```text
3353f1e Add full HIC compact JSON run support
63829e8 Add HIC compact JSON SFT workflow
```

截至写入本文时：

```text
local and origin/codex/hic-viewpoint-ablation are synchronized
```

## 本轮失败点与教训

### 失败点 1：全量任务被 GPU 资源阻塞

Full context / training 曾启动，但没有真正进入生成或训练阶段。

主要表现：

```text
train context lines: 0
val context lines: 0
checkpoint: none
final_lora: none
```

原因：

```text
GPU 被 ollama runner 或其他进程长期占用
free memory 低于 full run 要求
```

教训：

- 长任务必须用 `tmux`；
- 必须写日志；
- 必须能断点续跑；
- 必须有 GPU memory gate；
- 不要默认 overwrite；
- 每次恢复前先检查输出行数、checkpoint、日志尾部。

### 失败点 2：一开始容易把 gold-derived upper-bound 和真实部署混在一起

`hic-compact-json` 目前依赖 gold caption 来提取笑点。

这对训练和上限评估有价值，但不能直接用于真实推理。真实推理时没有 gold caption。

正确拆分应该是：

```text
upper-bound route:
image + gold caption -> compact JSON -> generator

deployable route:
image -> image-only humor hypotheses -> generator -> reranker
```

### 失败点 3：只看 SFT loss 不够

512 pilot 的 loss 明显下降，但幽默 caption 质量不能只靠 loss 判断。

必须同时看：

- gold similarity；
- best-of-N 是否命中；
- 解释腔；
- refusal；
- groundedness；
- diversity；
- Qwen7B / human preference win rate。

### 失败点 4：prompt 容易诱导解释腔

结构化 context 如果写得像“分析任务”，generator 就会学习解释，而不是写 caption。

修复方向：

```text
context 是 clue，输出仍然必须是 caption
```

并且要通过自动指标持续统计 explanation/generic pattern。

### 失败点 5：GitHub 上传曾被认证挡住

尝试过：

```text
git push -u origin codex/hic-viewpoint-ablation
```

失败原因：

```text
fatal: could not read Username for 'https://github.com': No such device or address
```

当时本机状态：

- 没有 `gh`；
- 没有 `GITHUB_TOKEN` / `GH_TOKEN`；
- 没有 credential helper；
- SSH key 也不可用；
- GitHub connector 不能直接使用本地 blob SHA 更新远端 tree。

后来远端已经同步到当前提交，说明用户或环境补上了认证/同步路径。后续如果再次上传失败，优先检查：

```bash
gh auth status
git remote -v
ssh -T git@github.com
git status -sb
git rev-list --left-right --count origin/codex/hic-viewpoint-ablation...HEAD
```

## 后训练相关的当前判断

### SFT

SFT 已经有价值，512 pilot 证明模型能学会：

- 更短；
- 更像 caption；
- 更接近 gold humor；
- 更少解释腔。

但继续扩大 SFT 的风险是：

- 学 gold-derived context shortcut；
- 学到数据集口癖；
- refusal 上升；
- 对真实 image-only 推理帮助有限。

### DPO

DPO 适合本任务，因为幽默 caption 本质上是偏好问题，不是唯一标准答案问题。

可行数据构造：

```text
chosen: 短、自然、贴图、接近 gold humor point、不解释、不拒答
rejected: 解释腔、泛泛描述、偏离图片、拒答、没抓住 gold humor point
```

但 DPO 需要先有可靠 preference pair。建议不要直接从复杂随机扰动开始，而是先用真实生成候选构造 pair：

```text
base candidates
hic-compact-json prompt candidates
512 LoRA candidates
3000 LoRA candidates
gold caption
```

然后用 gold caption + judge + 小规模人工校准产生偏好标签。

### Train-Inference Mismatch

如果推理时也会先用另一个模型生成笑点，再把笑点放进 prompt，那么训练和推理并不是“有没有 context”的 mismatch，而是：

```text
gold-derived context vs image-only predicted context
```

训练时 context 来自 gold caption，质量更高、更贴近目标笑点。

推理时 context 来自另一个模型，只能从图片预测，可能更噪声、更不确定、更多候选。

因此 mismatch 仍然存在，但形式更细：

- context source mismatch；
- context quality mismatch；
- viewpoint distribution mismatch；
- error propagation mismatch；
- confidence calibration mismatch。

解决方向不是放弃 context，而是：

- 用 noisy / predicted context 做训练增强；
- 训练 image-only extractor；
- 让 generator 接受 top-k hypotheses；
- 通过 reranker 选择候选；
- 用 DPO/IPO/ORPO 等偏好优化惩罚错误 context 下的坏输出。

## 下一步建议

### 最务实路线

1. 先不要把 full SFT 当唯一主线，让它作为后台 upper-bound 实验。
2. 固定评估集：

```text
sft_test first 200
sft_test random 1000
hard subset
```

3. 比较：

```text
base
base + hic-compact-json prompt
512 LoRA
3000 LoRA
full LoRA if available
```

4. 生成每图 8 个 candidates。
5. 用 gold caption + Qwen7B judge + 小规模人工检查建立 preference pairs。
6. 先训练 reranker 或做 best-of-N selection。
7. 再决定是否 DPO。
8. 并行训练 / 评估 image-only viewpoint extractor。

### 更像最终系统的路线

```text
image
 -> image-only humor hypothesis extractor
 -> top-k compact JSON hypotheses
 -> caption generator produces N captions
 -> reranker / judge selects best caption
```

这比单次生成更符合幽默任务本质，因为幽默生成高方差、多答案、强偏好。

## 推荐的上下文储存方式

我建议以后不要只存一份聊天总结，而是分成四层。

### 1. Stable instruction

文件：

```text
instruction.md
```

用途：

- 项目固定约束；
- 模型路径；
- 数据路径；
- 不能做的事；
- 标准 prompt；
- 标准采样参数；
- 新 Codex 窗口必须先读的内容。

这层应该少改，只记录长期有效的规则。

### 2. Obsidian-style project log

目录：

```text
docs/obsidian/
```

用途：

- 阶段复盘；
- 决策理由；
- 失败点；
- 下一步；
- 论文脉络；
- 人类可读的研究思路。

命名建议：

```text
YYYY-MM-DD 主题.md
```

这层适合你思考，也适合新对话恢复上下文。

### 3. Experiment card

建议目录：

```text
docs/experiments/
```

每个实验一个文件，例如：

```text
2026-07-29-hic-compact-json-512-eval.md
```

固定字段：

```text
commit
dataset split
sample ids / seed
model
adapter
prompt renderer
generation params
metrics
output paths
known caveats
decision
```

这层用于保证实验可复现。

### 4. Machine-readable run manifest

建议文件：

```text
outputs/manifest/runs.jsonl
```

每行记录一个 run：

```json
{"run_id":"...","commit":"...","model":"...","adapter":"...","dataset":"...","outputs":["..."],"metrics":{"gold_match":0.12},"status":"complete"}
```

这层给脚本读，不靠人类记忆。

## 给下一轮 Codex 的恢复提示

下一轮如果要继续本项目，优先读取：

```text
instruction.md
docs/obsidian/2026-07-27 HIC Humor Captioning 项目复盘与下一步.md
docs/obsidian/2026-07-29 HIC Humor Captioning 本轮上下文与失败复盘.md
README.md
```

然后检查：

```bash
cd /home/zhang.jianqiao/projects/Humor-generator
git status -sb
git log --oneline --decorate -5
```

训练/全量任务状态检查：

```bash
tmux ls
tmux capture-pane -pt hic-compact-json-train-full -S -80
wc -l /home/zhang.jianqiao/projects/v2.5/outputs/analysis/hic_humor_viewpoints_sft_train_full.jsonl
wc -l /home/zhang.jianqiao/projects/v2.5/outputs/analysis/hic_humor_viewpoints_sft_val_full.jsonl
nvidia-smi
```

如果继续研究，不要直接开始全量训练。先决定：

```text
是做 512/3000/systematic eval，
还是做 preference pairs / reranker，
还是做 image-only viewpoint extractor。
```
