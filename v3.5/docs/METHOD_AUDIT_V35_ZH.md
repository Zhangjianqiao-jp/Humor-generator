# v3.5 方法审计与修正记录

## 结论

v3.5 当前是一个可检验的 bridge-only 实验，而不是“latent 必然更好”的展示工程。真实 GPU engineering smoke 已通过，但当前仍没有 held-out 模型收益，不能启动 preference learning。

## 已发现并修正的高风险问题

1. **Receiver 接口漂移**：旧设计让部分通信条件只看 description，但 7B SFT Generator 的真实接口是 image-conditioned。现统一为 `image + 原 SFT instruction + communication`；严格 HOMER 文本系统另列，不混作模态消融。
2. **视觉特征被绕过**：直接把 `input_ids` 转 embedding 后调用 latent generation 会丢失 Qwen-VL 的视觉特征替换。现由统一后端先运行冻结 vision tower、替换 image placeholders、计算原始 3D MRoPE，再插入 latent slots。
3. **latent 后位置错位**：插槽后 caption suffix 的三个 MRoPE 轴原先没有平移。现插槽使用连续 text-like positions，所有 suffix positions 同步平移，并有单元测试。
4. **predictive state 与 communication state 混淆**：generation hook 的最后状态用于预测下一个 token，它不是已经包含该 emitted token 的状态。现保留 predictive hook 只做 causal replay；bridge 实际读取 teacher-forced post-token states。trace 明确记录 state definition。
5. **伪 semantics**：旧 trace 曾保存占位字符串。现 trace 必须保存三个 Planner 调用的真实 raw output，`unspecified`/placeholder 一律拒绝。
6. **通道被截断**：先拼接再取尾部可能完全删掉 conflict/local。现 conflict/local/global 各保留 8 个 causal-tail positions。
7. **Typed 参数不公平**：Typed 曾因三个独立 pooler 拥有更多参数。现共享同一 pooler，只分三组 query；Learned-24 与 Typed-3×8 的 trainable params 完全相同。
8. **随机负例太容易**：现优先同来源、description 最相近但 conflict signature 不同的 image cluster，并记录相似度诊断。
9. **评测规模不对齐论文**：主评测从 Group-of-3 改为 Group-of-10；Group-of-3 只作 legacy sensitivity。
10. **位置偏差与多重比较混合**：正式 packet 生成 A/B 镜像项；primary 与 mechanistic secondary 是不同 Holm family；blind ID 包含 group size/family/orientation，禁止碰撞。
11. **盲评只做相对胜负**：新增 group 与 candidate 两级 `good/weak/bad`，防止“相对更好”被写成“真正好笑”。
12. **评审校准缺失**：增加五个非测试图片、crowd-ranked A/B calibration pairs；它对齐 Humor in AI 的 5-shot 思想，但不是其逐测试项随机抽样的逐行复现。
13. **多样性指标偏离 benchmark**：补入 Humor in AI 官方实现的 Average EAD 与 `all-mpnet-base-v2` SBERT diversity；Distinct/self-BLEU/TF-IDF/Vendi 只作补充。
14. **测试污染**：47 张官方图片中 23 张曾被冻结 SFT adapter 看过。主结论只用 97 张 internal unseen + 24 张 official adapter-unseen；23 张单列 diagnostic。
15. **trace 缺少代码身份**：先前正式 trace 作业在 v3.5 未提交时启动，只记录模型和 tensor hash，无法锁定实际 prompt/代码。该作业已取消、部分缓存已隔离；schema v2 现在强制记录唯一 Git commit、dataset manifest、prompt 与 adapter manifest hash。
16. **pilot 子集存在编号偏差**：旧实现取排序后的前 64/24 clusters。现改为固定 seed 的 SHA-256 cluster 抽样，且保存所选 cluster IDs 的 hash。
17. **loss 完成被误写成实验完成**：现在三个 pilot 训练后还会自动运行 24-image × 3-seed generation 并产生匿名评审 packet；自动化在独立评审前停止，不会自动进入 full training。
18. **无效 finalist 被当成 caption**：Electronic Sheep 用标量 `UNKNOWN` 表示缺失；旧 loader 把字符串逐字符迭代，产生 `U/N/K`，另有 `nan`。现只接受 list finalists，并拒绝 sentinel；共移除 133 个 train rows，所有图片簇划分不变。
19. **caption 清洗错误地牵连昂贵 trace**：旧 provenance 绑定整个 dataset manifest。现新增只描述 Planner 实际输入的 `trace_inputs.jsonl`；旧缓存仅能通过逐条 cluster/split/description/tensor-hash 验证后的原子迁移复用。
20. **learned bridge 读取的信息更多**：Learned/Typed 读取完整 hidden sequence 后压缩为 24 slots，而 budget text/token embedding/StateBridge 只读取每通道末 8 tokens。因此主结果应解释为“full-state learned compression”，不能仅据此声称 continuous 优于 text。新增 `typed_quantized`：同一 bridge 输出最近词表 embedding，与连续 Typed 的差异才更接近连续残差效应。
21. **pilot 内外验证混用**：24 张 validation subset 已用于 early stopping，不能再作为唯一生成证据。现把其余 40 张设为 outer-pilot generation/evaluation 集，并把 24 张仅保留为训练诊断。
22. **镜像 packet 被当作独立样本**：A/B 镜像用于检测位置偏差，不会把图片或评审数翻倍。聚合器现先在 `rater × image × comparison × mirror_pair` 内折叠两个方向，再进行 image-cluster bootstrap、显著性检验和一致性计算。

## 仍然不能过度声称的部分

- `statebridge` 是三通道、24-slot、异 adapter 的 StateBridge-style adaptation，不是论文默认 64-token 同质 agent 设置的原样复现。
- 固定 Qwen2.5-VL revision 的 HOMER 是 method/data reproduction，因为 HOMER 没公开其精确 Qwen-VL revision。
- `full_plan_text` 是语义上界，不是带宽匹配 control；真正的模态比较必须同时看 `budget_text` 与 `token_embedding`。
- EAD/SBERT 高只代表候选集合差异更大；只有 absolute quality 不降且 good-only angle coverage 提升，才能支持“更有用的幽默角度多样性”。
- 121 张图片 × 10 seeds 是每系统 1,210 条 captions，不是 1,210 个独立图片样本；推断单位始终是 image cluster。

## 当前执行门禁

1. CPU tests、环境、artifact hash、dataset leakage、v3.5 isolation 全部通过；
2. 两个真实 Planner traces 的 engineering smoke 已验证图像特征、MRoPE、post-token states、冻结参数、finite loss/gradient/update、峰值显存及所有生成路径；
3. 正式 trace 必须来自 clean Git commit 并通过完整 provenance gate；
4. 之后只串行运行 Learned+KL、Typed+KL、Typed-no-KL 三个小 pilot；
5. 三者训练结束后，在未用于 early stopping 的 40 张 validation 图片上，加入 budget-text 与 typed-quantized controls 生成 blind packet，并停下来等待独立评审；
6. pilot 有 validation 与真实生成信号后，才扩展到 3-seed full bridge training；
7. bridge 显示稳定 held-out 收益后，才重新讨论 preference learning。

## 权威依据

1. Shang et al., HOMER, ICLR 2026: https://openreview.net/pdf?id=SzaRhPom4o
2. HOMER official implementation: https://github.com/Shang-hub/HOMER-Official-Implementation
3. Du et al., InterLat, ACL 2026: https://aclanthology.org/2026.acl-long.1248/
4. Peng et al., StateBridge, COLM 2026: https://arxiv.org/abs/2608.13317
5. StateBridge official implementation: https://github.com/YanwenPneg/StateBridge
6. Zhang et al., Humor in AI, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/e297fb6cd1690ee5b39c5bb4c58ad801-Paper-Datasets_and_Benchmarks_Track.pdf
7. Humor in AI official implementation: https://github.com/yguooo/cartoon-caption-generation
8. Hessel et al., Electronic Sheep, ACL 2023: https://aclanthology.org/2023.acl-long.41/
9. Tevet & Berant, Evaluating the Evaluation of Diversity in NLG, EACL 2021: https://aclanthology.org/2021.eacl-main.25/
10. Friedman & Dieng, The Vendi Score, TMLR 2023: https://arxiv.org/abs/2210.02410
