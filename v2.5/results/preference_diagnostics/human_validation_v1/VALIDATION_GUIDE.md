# Preference validation guide

## 推荐顺序

1. Quick gate：61 项，每张 pair-producing image 一项，选取该图最低 score-margin 的困难 pair。
2. Best-of-N audit：24 项，盲评两个系统各自的 best-of-32。
3. 只有 quick gate 通过后才做完整 485 项。

## 每个 pair 必须判断

- Preference：A/B 哪个对当前图片更好笑；没有明确差异就选 tie。
- Grounding：1=无关或与图片矛盾，3=大致相关，5=准确利用具体视觉细节。
- Hallucination：caption 是否依赖图片中不存在的对象、动作或关系。
- Generic/template：caption 是否可以不改动地套用到很多图片。
- Pair type：H1 humor-vs-literal；H2 strong-vs-weak/cliche；H3 grounded-vs-hallucinated；H4 image-specific-vs-generic。
- Use for training：只有偏好明确、winner grounded、negative 不属于低级语法错误时才选 yes。

## Quick gate 停止条件

解盲前不要看来源分数。解盲后若出现以下任一情况，不应直接训练：

- 原 chosen 的人工胜率低于 70%；
- tie/invalid 合计超过 20%；
- 超过 15% 的原 chosen 被判 hallucinated 或 grounding <= 2；
- 大部分 pair 仍只能归为 H2，无法测量 H1/H3/H4。

这些阈值是数据工程 gate，不是论文中的通用常数；最终应报告分子、分母和 bootstrap 95% CI。
