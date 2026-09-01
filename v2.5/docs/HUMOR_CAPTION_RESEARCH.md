# 图像幽默规划与 caption 生成：论文依据和本项目映射

更新时间：2026-08-12。本文件只记录能够由论文原文或官方会议页面支持的结论，并区分“论文方法”和“本项目实现”，避免把工程选择误写成文献结论。

## 任务设计结论

本项目将一阶段直接生成拆成两段：

1. 7B planner 从图片输出 `ANCHOR / CONTRAST / ANGLE`。
2. 3B captioner 同时接收图片和该 plan，输出一句 caption。

这一拆分与 IRCoT 的核心观察一致：内容专属幽默需要先抽取图片细节和不协调点，再形成 resolution，直接生成容易产生跨图片复用的泛化笑话。这里的近似对应是 `ANCHOR ≈ unique image detail`、`CONTRAST ≈ incongruity`、`ANGLE ≈ resolution direction`；本项目不是对 IRCoT prompt 或实验的逐字复现。

Hessel et al. (ACL 2023) 的 New Yorker benchmark 将 locations/entities、图中异常之处和笑话解释作为独立标注，并报告即使给定高质量视觉描述，机器解释仍明显弱于人类解释。这支持让 planner 产生可检查的中间变量，而不是把不可观测推理全部压入最终 caption loss。

## 论文与实际决策

| 权威来源 | 论文结论 | 本项目中的对应实现 | 尚未声称的内容 |
| --- | --- | --- | --- |
| Tanaka et al., NAACL Findings 2024, IRCoT | 从图片独有细节构造 incongruity/resolution，并用 negative sampling 抑制通用笑点，可提高内容相关性 | 7B 输出可审计的三字段 plan；最终做跨图片 plan 替换测试 | 当前 SFT 没有实现其 logit bias，也不声称达到论文的人评结果 |
| Hessel et al., ACL 2023 Best Paper | New Yorker 幽默理解包含图文匹配、优胜 caption 识别和解释生成；视觉实体、异常和解释可分开评价 | `ANCHOR/CONTRAST` 分离常规视觉内容与异常；不把 gold caption 塞入 planner 输入 | 自动 bootstrap plan 不是人工 gold explanation |
| Hessel et al., NeurIPS 2024 Datasets & Benchmarks | 发布 365 场比赛、约 2.84 亿评分的 caption preference 数据；评分适合训练/评价幽默偏好 | 只使用用户指定的 New Yorker 数据；caption SFT 按源 rank 取各比赛前 3%，按 contest/image 隔离切分 | 高排名不等同于无噪声，SFT loss 也不等同于好笑程度 |
| Chandrasekaran et al., NAACL 2018 | 图像相关的机智文本可通过受约束生成/检索获得，并应通过人类比较评价 wit | 评价中同时检查图像特异性和简洁性；保留多候选接口 | 本项目目前不训练专门的 pun 模块 |
| Hu et al., 2021 / Dettmers et al., 2023 | LoRA 与 QLoRA 可在冻结基座上训练低秩增量，减少训练显存和优化器状态 | 7B/3B 均采用 NF4 QLoRA，分别只训练 0.1074%/0.3612% 参数 | 不进行全参数微调 |

## 数据预处理的可审计定义

- 原始来源：`yguooo/newyorker_caption_ranking` 的 cartoon、caption、rating 与 GPT-4o scene descriptions。
- 使用限制：本地 source manifest 记录为 CC BY-NC 4.0，dataset card 限制直接使用于学术研究，不应把当前 adapter 直接用于商业产品训练或发布。
- caption 筛选：每个 contest 内按数据源给出的 `rank` 升序（0 最优）取有效 caption 的前 3%；不是按跨 contest 不可比的原始评分做全局 top 3%，避免某些 contest 垄断数据。
- split：以 contest/image 为单位继承 train/validation/test，禁止同一图片跨 split。
- planner 标签：只使用发布数据中的 `canny`、`uncanny` 视觉描述；`ANCHOR`/`CONTRAST` 取各自完整首句，`ANGLE` 是粗粒度策略类别。
- 防泄漏：planner 的输入和标签均不包含高分 caption 或其释义；captioner 才使用 image + plan → high-score caption。
- clean-v2 规模：planner 79/24/24 张 train/validation/test 图片；captioner 13,190/3,990/4,415 行；预处理丢弃 0 行。
- 已知限制：planner 标签是自动 bootstrap 标注，尤其 `ANGLE` 为启发式类别，因此最终必须使用错配 plan 和无 plan 的对照实验判断中间变量是否真正有用。

## 训练与评价原则

SFT 的直接量化对象是 gold 输出 token 的条件负对数似然。对 7B，它提高生成合法三字段 plan 的概率；对 3B，它提高在给定图片和 plan 时生成高分 caption 风格文本的概率。validation loss/PPL 只能证明 held-out token prediction 改善，不能单独证明更幽默。

最低验收包括：

1. adapter 张量完整且全部有限，训练/验证无图像 token mismatch、OOM 或非有限 loss；
2. 7B 在未见图片上严格输出三行，并且字段能指向图片独有内容；
3. 3B 在未见图片上输出单句、非空、无 prompt 泄漏的 caption；
4. 真正级联运行 `image → generated plan → caption`，不能只用 gold plan；
5. 后续报告 `generated correct plan`、`swapped plan`、`no plan` 三组的人评胜率或配对偏好，量化 7B 的边际贡献。

## 权威参考

- Tanaka, K. et al. (2024). *Content-Specific Humorous Image Captioning Using Incongruity Resolution Chain-of-Thought*. Findings of NAACL 2024. https://aclanthology.org/2024.findings-naacl.152/
- Hessel, J. et al. (2023). *Do Androids Laugh at Electric Sheep? Humor “Understanding” Benchmarks from The New Yorker Caption Contest*. ACL 2023 Best Paper. https://aclanthology.org/2023.acl-long.41/
- Hessel, J. et al. (2024). *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*. NeurIPS 2024 Datasets and Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
- Chandrasekaran, A., Parikh, D., & Bansal, M. (2018). *Punny Captions: Witty Wordplay in Image Descriptions*. NAACL 2018. https://aclanthology.org/N18-2121/
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. https://arxiv.org/abs/2106.09685
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. https://arxiv.org/abs/2305.14314
