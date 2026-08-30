# 幽默 Caption Preference Learning 当前工作计划

更新日期：2026-08-25（JST）

## 2026-08-28 Full DPO完成后的Objective决策

Quality64完整DPO已完成2,163 steps，但出现明确的相对优化风险：validation loss和reward margin持续改善，reward/policy accuracy基本不变；chosen logp/token从`-2.9062`降至`-3.0125`，24/24张validation图片均下降。随后7B固定Hint image-shuffle诊断作业`6631995`显示正确图像margin相对错图均值为`-0.0279`、12胜12负，未发现正确图片条件优势。

因此不继续扩大vanilla DPO。下一方向优先为`conditional image preference + chosen anchor`；IPO只保留为偏好噪声的后续对照。正式新训练前先完成SFT与四个DPO阶段checkpoint的24-image × 3-seed生成层validation。完整依据、数值和实验矩阵见`docs/7B_DPO_SCALING_OBJECTIVE_DECISION.md`。

## 2026-08-28 完整 Quality64 受控 Scaling 实验（正在运行）

不能把此前的1,264-pair/158-step Pilot解释为“DPO无效”，因为它没有回答完整17,297 pairs的规模效应。项目负责人因此授权一次受控完整数据实验。该实验仍从冻结的`outputs/7b-generator/best_val_loss`开始，保持DPO `beta=0.1`、MLP LoRA `r=3/alpha=6`、学习率`2e-6`、有效batch 8与seed `20260825`不变；只改变训练规模并加入阶段性监控。

阶段点为step `0/158/632/1264/2163`，分别对应baseline、原Pilot计算量、约4倍、约8倍以及完整遍历17,297 pairs。每个阶段保存：

- 独立LoRA checkpoint：`outputs/7b-generator-dpo/dpo_mlp_quality64_full/checkpoints/step_XXXXXX/`；
- 完整384-pair逐pair验证结果；
- 以24张validation图片为统计单位的mean、median和5,000次cluster bootstrap 95% CI；
- DPO loss、reward/policy accuracy、reward margin、chosen/rejected absolute log-probability。

Early stopping主指标预先固定为`eval_image_mean_loss`：相对历史最佳值至少改善`0.0002`才重置patience，连续2个阶段未达到该改善幅度则停止。该阈值是本项目的工程控制量，不声称为论文通用常数。Test47在整个训练和validation过程中继续封存。

可视化由`results/7b_generator/dpo_quality64_scaling_monitor/index.html`和`monitor.png`每15秒更新，展示训练loss、image-clustered validation loss/accuracy和chosen/rejected logp。登录节点的只读服务运行在tmux `dpo-scaling-monitor`、端口`8771`。本地查看命令：

```bash
ssh -N -L 8771:127.0.0.1:8771 <服务器SSH别名>
```

随后在本地浏览器打开`http://127.0.0.1:8771/`。

首个正式作业`6630888`提交到当时有空闲完整GPU的`c-batch`，只申请1 GPU、3小时；真实batch=4 optimizer smoke通过。正式训练前20步实测约`5.8秒/step`，加上baseline与四次validation，完整运行预计超过3小时。调度器不允许在线延长RUN状态作业；为了避免在约step 1,800被walltime强杀并丢失optimizer状态，作业在只完成20/2,163步时主动取消，并以实测所需的4小时重新提交为作业`6630912`。该调整依据实际吞吐，不是预留过量安全余量。实现依据包括DPO的reference-relative离线偏好目标（Rafailov et al., 2023）、IPO对确定性/噪声偏好过拟合风险的分析（Azar et al., AISTATS 2024），以及RPO对preference overoptimization使用SFT regularization的研究（NeurIPS 2024）。本轮先测DPO scaling，不把RPO/anchor提前混入主变量。

## 2026-08-27 Quality64 固定步数数据质量实验（当前执行项）

项目负责人已确认以下四阶段门控，顺序不得跳过：

1. 使用冻结的 `outputs/7b-generator/best_val_loss` 为 Quality64 训练池计算 7B reference log-probabilities；
2. 从完整 Quality64 reference 池按 tier 比例、跨图片 round-robin 固定抽取 1,264 对，以完全相同的 158 optimizer steps 训练 DPO；
3. 主验证集固定为旧 Pilot16 使用的同一组 384 pairs。只有相对旧 Pilot16 的 validation loss、reward accuracy/margin 等预注册指标改善，才讨论完整遍历 17,297 pairs；
4. 只有 validation Go 才允许重新运行封存的 47-image × 3-seed test 盲评。

严格控制不变的变量为：7B SFT 起点、DPO `beta=0.1`、MLP `gate/up/down_proj`、LoRA `r=3/alpha=6/dropout=0.05`、学习率 `2e-6`、batch size 1、gradient accumulation 8、cosine scheduler、seed `20260825`、图像/序列预算及 validation 数据。Quality64 同时改变 pair 质量与独立训练图片覆盖（79→271），因此本实验估计的是 **Quality64 数据方案整体效应**，不得表述为纯 pair-quality 因果效应。

当前配置与入口：

- reference：`configs/7b_generator/dpo_reference_quality64.yaml`；
- 固定步数 DPO：`configs/7b_generator/dpo_quality64_step_matched.yaml`；
- 作业：`jobs/genkai_7b_quality64_reference_and_step_matched_dpo.pjm`；
- reference 输出：`data/processed/newyorker_published_dpo_reference_7b_generator_quality64/`；
- pilot checkpoint：`outputs/7b-generator-dpo/dpo_mlp_quality64_step_matched/`。

提交记录：CPU/schema/配置等价性门禁共 11 项测试通过；Quality64 SHA-256 为 `98c5ff352edcdd2c6260f9c61a3b7a7e9f813b29a3905fb50a92730b7e400165`，冻结 SFT adapter SHA-256 为 `780f44d08f30f358c9a3fc994867dfde837b75ef973bf54de768d975891b5b7b`。2026-08-27 首次作业 `6606292` 直接启动，但 batch-size 2 的短样本 smoke 虽通过，全量首批仍触发 MIG allocator/NVML internal assert；按既往记录判定为临界 OOM。该作业 3 分 51 秒后以 exit code 1 结束，完整 reference 与 DPO 均未开始。修复为历史已验证稳定的 batch-size 1，并将 smoke 改为 Quality64 最长 chosen+rejected 文本压力样本；完整 reference 每 50 batch 刷新可恢复的 `.partial.jsonl`，避免长任务中断后从零计算。修复后作业 `6606436` 于 20:00:08 JST 直接启动，最长文本 smoke 通过。首个 100-pair 实测约为 50 pairs/分钟，完整 reference 约需 5 小时 45 分，再加历史 DPO 81 分钟会使原 7 小时串联作业在 checkpoint 保存附近撞 walltime。为避免末尾失败，该作业在 4 分钟时主动停止，安全保留 150-pair partial。正式流程拆成两个串行单-MIG 作业：`jobs/genkai_7b_quality64_reference_only.pjm`（6 小时，可恢复 reference）成功后才自动提交 `jobs/genkai_7b_quality64_step_matched_dpo.pjm`（2 小时，158-step DPO + validation gate）。两个作业不会并发占卡，也不会读取 test47。Reference-only 作业 `6606506` 于 20:05:56 JST 直接进入 `RUN`，从已验证的第 151 对继续。

Test 封存规则：当前 reference 任务只处理 Quality64 `train`，不处理 `test`；训练与门控不读取 47-image test。完整 Quality64 reference 的预计算不是完整 DPO 训练，只有 1,264 对进入本轮 optimizer。

### 2026-08-27 GPU 策略变更：完成时间优先

项目负责人明确将最短完成时间设为第一优先级。20:19 JST 实时资源为：`b-batch` 0/44 GPU 空闲、`b-batch-mig` 35/56 MIG 空闲、`c-batch` 3/8 GPU 空闲。MIG reference 已完成并落盘 900/17,297 对，但 batch=1 的预计剩余时间明显长于完整 H100。作业 `6606506` 因此主动停止并保留900行可恢复partial；后续切换到 `c-batch`，先以batch=8对八条最长文本进行压力smoke，通过后从第901对继续。Reference作业walltime收紧为2小时；随后自动提交的158-step DPO也使用`c-batch`、walltime 1小时。任何时刻仍只保留一个正式GPU作业，不以多个资源组race。

切换前14项测试、shell/config门禁及900行resume-prefix校验通过。`c-batch` reference作业`6606655`于20:21:53 JST进入`RUN`；它申请1张GPU，未与旧MIG作业重叠。8条最长文本的batch=8压力smoke通过；20:25 JST已从900推进至1,300，实测约400 pairs/50秒，按当前吞吐预计reference约21:00完成，随后自动提交DPO阶段。

Validation Go/No-Go 在看到新结果前固定为：`old_eval_loss - new_eval_loss >= 0.001`；reward accuracy 或 reward margin 至少一项不下降；reward accuracy 不下降超过 0.01；reward margin 不下降超过 0.001；chosen absolute log-probability 不下降超过 0.1。五项全部通过才标记 `GO_FULL_QUALITY64`。脚本为 `scripts/compare_dpo_validation_gate.py`，输出为 `results/7b_generator/dpo_quality64_step_matched/validation_gate.json`。按最新授权，GO会自动提交完整17,297-pair DPO，NO-GO则停止；两种情况都不会自动解封test47。

### 条件式完整DPO自动化（2026-08-27）

按项目负责人最新授权，gate行为更新为：`GO_FULL_QUALITY64`时自动提交`jobs/genkai_7b_quality64_full_dpo.pjm`；`NO_GO_FULL_QUALITY64`时明确停止且不提交。完整训练使用全部17,297个reference pairs、覆盖271张图片、1 epoch、有效batch 8、预计2,163 optimizer steps。为利用完整H100，micro-batch为4、gradient accumulation为2；DPO objective、SFT起点、MLP LoRA r=3、学习率、seed、validation均保持不变。正式训练前必须通过一次真实batch=4、accumulation=2 optimizer smoke。完整输出为`outputs/7b-generator-dpo/dpo_mlp_quality64_full/`。即使完整DPO成功，test47仍保持封存，等待单独授权评测。

### 2026-08-28 链式提交故障与修复

`c-batch` reference作业`6606655`成功生成全部17,297条reference，但计算节点不提供`pjsub`，因此作业最后一行链式提交报`command not found`并以exit 127结束；reference产物本身完整，pilot未被提交。原tmux监控器只观察、不代替提交，导致流水线停在reference完成处。修复后，GPU作业只生成结果和gate；所有后续`pjsub`由登录节点tmux监控器执行。监控器以显式pilot job ID启动，等待pilot结束并读取gate；只有GO时提交完整DPO，并把full job ID持久化到`results/7b_generator/dpo_quality64_step_matched/full_job_id.txt`以防监控重启后重复提交。

14:18 JST完整reference逐行审计通过：17,297个pair_id全部唯一、原始字段与Quality64逐字段一致、chosen/rejected reference log-probabilities全部有限且监督token数为正。此时`b-batch`与`c-batch`均为0 GPU空闲，`b-batch-mig`仍有5/56空闲；按完成时间优先，158-step pilot临时使用单MIG与2小时walltime（历史同配置实测81分钟），不等待完整H100。完整17,297-pair DPO仍使用`c-batch`的batch4配置。

Pilot作业`6629384`于14:20:12 JST直接进入`RUN`。登录节点tmux监控器以该显式job ID重启；轮询间隔从20分钟缩短为5分钟，以便pilot结束后尽快读取gate并在GO时提交完整DPO。

### 2026-08-28 Quality64 Pilot 与生成层 Validation 结论

Pilot `6629384` 正常完成158步训练。相对旧Pilot16，pairwise validation loss仅改善`0.000513`，未达到预注册的`0.001`门槛；reward accuracy从`58.07%`升至`61.72%`，reward margin从`0.005705`升至`0.006711`。自动gate因此判定`NO_GO_FULL_QUALITY64`，没有提交完整17,297-pair DPO，test47继续封存。

为判断log-probability门槛是否漏掉生成层收益，作业`6630138`在相同24张fixed-hint validation图片上生成SFT、旧DPO、Quality64三组各3条caption，并建立48项位置平衡的匿名Group-of-3 packet。作业使用`c-batch`单张H100，12分40秒正常完成。辅助Qwen judge对三组全部24项、所有维度均打5分，确认发生评分饱和；该结果无区分力，不参与gate。

Codex在不读取system key的条件下完成48项视觉盲评并冻结结果，随后解盲。Quality64相对旧DPO：overall为13胜10负1平，平局计0.5得分率`56.25%`，decisive win rate `56.52%`（95% Wilson CI `36.81%–74.37%`，双侧exact sign test `p=0.678`）；best-pick为14胜9负1平，平局调整得分率`60.42%`。绝对`good`为Quality64 `7/24`、旧DPO `2/24`，配对迁移为7张仅Quality64 good、2张仅旧DPO good（exact McNemar `p=0.180`）。相对SFT，Quality64 overall为15胜7负2平，平局调整得分率`66.67%`。

结论：Quality64呈一致的正向趋势，尤其提高了绝对good数量，但单一judge、24张图下置信区间仍跨过50%，尚不能称为稳定确认。维持`NO-GO`，不启动完整DPO、不读取test47。下一项低成本工作是增加至少一个独立盲评者复核同一冻结packet；只有多judge聚合仍为正向且无明显退化，才重新讨论完整DPO。评测产物位于`results/7b_generator/dpo_quality64_validation_blind/`。

独立复核材料已在`results/7b_generator/dpo_quality64_validation_independent/`生成。为形成可判定的多数票，准备了`llm_judge_2`与`llm_judge_3`两套独立重盲化PDF；每套48项、A/B各翻转24项，题序和blind ID不同，公开材料不含system identity。加上已冻结的Codex判断形成三评委。新增聚合器`report_consensus_group_eval.py`保留win/loss/tie/unresolved并报告neutral-imputed与resolved-only得分，避免对三方各不相同时强造赢家。相关7项单元测试、PDF页数及identity-leak检查全部通过。等待两个外部多模态模型返回完整JSON；此等待不授权GPU任务或完整DPO。

`llm_judge_2`的48项结果已冻结并通过ID/字段校验；原始A/B选择为25/22/1平，无显著位置偏差。解盲后，Quality64相对旧DPO为11胜12负1平（平局调整`47.92%`，good为5比6），相对SFT为10胜14负（`41.67%`，good为5比6）。这与Codex的正向判断冲突。两者overall原始一致率`77.08%`、Cohen kappa `0.527`，但absolute标签一致性较低（kappa约`0.25–0.29`）。继续维持NO-GO，等待`llm_judge_3`形成三评委多数结果。

`llm_judge_3`结果也已冻结并校验。其单独方向轻微支持Quality64：相对旧DPO为12胜10负2平（`54.17%`），相对SFT为13胜9负2平（`58.33%`）。三评委多数解盲后，Quality64相对旧DPO为13胜10负1平，平局调整`56.25%`，decisive 95% Wilson CI为`36.81%–74.37%`；absolute为Quality64 6 good/15 weak/2 bad/1 unresolved，对比旧DPO 4 good/20 weak/0 bad。相对SFT为13胜9负1平1 unresolved，neutral-imputed `58.33%`，但双方均为6 good，Quality64另有3 bad而SFT为0。overall Fleiss kappa仅`0.391`，absolute kappa为`0.112–0.187`。

最终决策固定为`NO_GO_FULL_QUALITY64`：预注册pairwise gate失败；三评委结果仅为方向性趋势且置信区间跨50%；绝对标签一致性弱并观察到bad退化；生成层结果还只有一个seed。不得用事后增加的多数趋势覆盖预注册门槛，不提交17,297-pair完整DPO，不解封test47。下一项建议不是继续对同一checkpoint反复检验，而是定位Quality64-only bad迁移，并设计带chosen-anchor/anti-regression约束的新受控pilot。机器可读决定见`results/7b_generator/dpo_quality64_validation_independent/final_go_no_go.json`。

进一步postmortem后撤回“chosen-anchor默认作为下一Pilot”的提法：两个bad案例都来自未见validation图片的单seed采样；Quality64 chosen per-token logp相对旧DPO只下降约`0.00107 nat/token`，不足以诊断概率坍塌；同seed下两个checkpoint仅1/24组候选完全相同，说明sampling分叉是重要混杂因素。完整17,297 pairs也只有271个图片cluster，中位每图64对，不能按17,297个独立样本理解。修订计划见`docs/QUALITY64_POSTMORTEM_AND_REVISED_PLAN.md`：先导出逐pair概率并做image-cluster统计、诊断固定Hint下的7B图像条件利用，再做DPO换seed复现与IPO；只有chosen likelihood系统下降才加入RPO/anchor，只有图像条件弱才加入conditional mDPO。

> **强制范围变更（2026-08-25）：从现在起，所有新训练、Preference Learning、生成评测和模块实验只允许在 7B Generator 上进行。3B Generator 方案正式废弃。**
>
> 3B checkpoint、日志与部分评测结果仅作为历史归档，不得再提交 3B 训练或评测作业，不得把未完成的 3B objective screening 当作最终结论。本文后半部分保留的 3B 配置只用于复现实验历史，不构成继续执行授权。

## 当前唯一有效的 7B 路线

| 项目 | 当前值 |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 任务 | image + fixed humor plan → humorous caption |
| SFT 数据 | `newyorker_compact_sft_v2/caption_train.jsonl` 与 `caption_validation.jsonl` |
| SFT module pilot winner | MLP：`gate_proj/up_proj/down_proj` |
| LoRA | rank 3、alpha 6、dropout 0.05 |
| 最佳 checkpoint | `outputs/7b-generator/best_val_loss` |
| 最终 checkpoint | `outputs/7b-generator/final_lora` |
| 最终 validation loss | `2.930127` |

7B module pilot 的 validation loss 为：MLP `3.6733`、Attention `3.7159`、All-linear `3.7239`。MLP 因此被选为本轮 SFT placement，但差距约 1%，且只有一个 seed；它是当前工程选择，不是“MLP 对所有 7B preference training 必然最优”的结论。

## 接下来的正常执行顺序（7B-only）

当前执行状态：按项目负责人最新指令，7B Base-vs-SFT held-out 评测作业 `6585267` 已在获得 GPU 前取消，不再等待该评测。下一步已直接切换为构建 7B SFT frozen reference；单 MIG 作业 `6585367` 于 2026-08-25 提交，当前处于 `QUE`。

作业 `6585367` 的硬性通过顺序为：

1. 验证 `outputs/7b-generator/best_val_loss` adapter 完整性；
2. 用 1 个 pair 完成真实 7B NF4 reference-logp 冒烟；
3. 为 train/validation/test 的 `1264/384/384` 个官方 preference pair 重新计算冻结 7B SFT reference log-probability；
4. 用同一个 7B SFT checkpoint、同一 NF4 配置和同一 MLP LoRA placement 完成一步 DPO forward/backward/optimizer smoke；
5. 只有以上步骤全部通过才进入 7B objective screening。

旧的 3B reference-logp 不可迁移到 7B。Reference 和 preference policy 均固定为 NF4、`max_seq_len=768`、相同 chat/image preprocessing，以确保训练初始 policy/reference 比值具有可解释性。

### 2026-08-25 运行更新

- `6585367` 已于 17:10:47 JST 正常结束，`EXIT CODE=0`，实际耗时 40 分 43 秒；
- 7B frozen reference 已完整生成并通过有限值检查：train/validation/test=`1264/384/384`；
- 一步真实 7B DPO forward/backward/optimizer smoke 已通过；
- 按“直接采用 DPO、跳过 7B objective screening”的决定，正式单-MIG 训练作业 `6587762` 已于 17:25:01 JST 提交；
- 正式配置：`configs/7b_generator/dpo_full.yaml`，DPO `beta=0.1`、MLP LoRA `r=3`、学习率 `2e-6`、1 epoch、全部 pair；
- 当前 `b-batch-mig` 已恢复 `ENABLE,START`；`6587762` 为普通 `QUE`，没有资源错误原因。

1. **建立 7B preference reference（正在执行）。** Preference pair 的 chosen/rejected 文本可以复用，但旧的 3B `reference_logps` 绝对不能复用；必须用冻结的 7B SFT checkpoint 重新计算 reference log-probabilities。
2. **直接训练 7B DPO。** 2026-08-25 项目负责人决定将 3B 的 objective 排名作为迁移先验，取消 7B DPO/SimPO/IPO/Anchored screening。固定 DPO、`beta=0.1`、MLP LoRA `r=3`，遍历全部 1,264 个训练 pair；配置为 `configs/7b_generator/dpo_full.yaml`。这是一项节省资源的工程决策，不等价于实验证明“DPO 在 7B 上优于其他 objective”。
3. **用生成与盲评验证 DPO 是否值得保留。** 主要指标为 7B DPO 对 7B SFT 的 image-level blind group win rate 与 95% CI；辅助 judge 只用于筛选。若 DPO 不能稳定超过 7B SFT，则停止 preference 扩展，转向数据质量、candidate reranking 或 best-of-N。
4. **只在 objective 显示正收益后做低成本 placement Go/No-Go。** 比较 7B Attention、MLP、All-linear、Gradient-selected、Random-selected，并近似匹配 LoRA 参数预算。没有稳定 selected-vs-random 优势就停止 Fisher/SVD/dynamic-rank 搜索。
5. **最终关键结果至少三个 seeds。** 报告 mean±std、image-bootstrap CI、训练参数量、显存、时间与吞吐。
6. **不恢复 3B 或双模型联合训练。** 除非项目负责人以后再次明确改变范围。

## 已归档的 3B Objective 结果

3B full-pair 训练本身成功完成，但 held-out 生成评测在范围变更时主动停止：SFT、DPO、SimPO 已完成 24-image 辅助 judge；IPO、Anchored 与正式盲评未完成。

| 3B 系统 | Qualified rate | Humor | Overall | 状态 |
|---|---:|---:|---:|---|
| SFT | 70.83% | 4.542 | 4.375 | 已完成辅助 judge |
| DPO | 62.50% | **4.583** | **4.583** | 已完成辅助 judge |
| SimPO | 58.33% | 4.542 | 4.542 | 已完成辅助 judge |
| IPO | — | — | — | 仅训练完成，生成评测未完成 |
| Anchored | — | — | — | 仅训练完成，生成评测未完成 |

结论：**DPO 是已评 preference 方法中的暂定领先者，但没有被证明是总体最优，也没有被证明优于 SFT。** 它的平均 Humor/Overall 更高，但 qualified rate 低于 SFT；样本仅 24 张，且缺少 IPO、Anchored 和盲评 win rate。因此不得写成“DPO 获胜”。这些 3B 结果不再驱动后续实验，只为 7B objective screening 提供候选优先级。

## 1. 历史 3B 目标与边界（已废弃，仅归档）

当前阶段只优化 3B Generator 的 preference learning，研究两个问题：

1. DPO、SimPO、IPO、Positive-Anchored Preference 中，哪一种 objective 更适合图像条件幽默 caption；
2. 在 objective 确定后，低成本判断 selective module placement 是否值得继续。

以下工作已经明确停止，不在当前计划内：

- 7B Humor Hint Model 与 3B Generator 的联合训练；
- `H0 → G1 → H1 → G2` 顺序/交替优化；
- Hint preference training；
- joint RL、GRPO 或端到端离散 Hint 反向传播。

现有 7B Hint checkpoint 和 Hint→Generator 接口只用于条件推理和评测，不再训练。

## 2. 固定资源与数据

| 项目 | 当前值 |
|---|---|
| 3B base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| 最佳 3B SFT adapter | `outputs/newyorker_compact_v2_captioner_3b_qlora/best_val_loss` |
| 7B Hint adapter（仅推理） | `outputs/newyorker_caption_aware_viewpoint_v3_7b_qlora/final_lora` |
| Preference train | `data/processed/newyorker_published_dpo_reference_3b/dpo_train.jsonl` |
| Preference validation | `data/processed/newyorker_published_dpo_reference_3b/dpo_validation.jsonl` |
| 训练规模 | 1,264 个精选 pair，全部遍历 |
| 验证规模 | 384 个 pair，全部遍历 |
| 当前 LoRA placement | Attention：`q_proj/k_proj/v_proj/o_proj` |
| LoRA 参数 | rank 16、alpha 32、dropout 0.05，共 7,372,800 个可训练参数 |

每个 preference pair 的 chosen/rejected 共享相同图像和相同 Hint prompt，避免把 caption preference 与 Hint 差异混为一谈。冻结 SFT reference log-probabilities 已预计算完成。

## 3. 已修复的问题

旧 pilot 无条件使用 `ImageBalancedPreferenceDataset`，导致每张图每个 epoch 只抽一个 pair：训练实际只有 79 pair/epoch，验证只有 24 pair，不能用于 objective 选择。

当前训练已改为配置驱动：

- `train_sampling: all_pairs`；
- `validation_sampling: all_pairs`；
- 旧的 `image_balanced` 模式仍保留，仅用于显式声明的廉价冒烟；
- 新实验写入 `outputs/preference_screen_fullpairs/`，不覆盖旧 pilot；
- 相关采样、loss 和 checkpoint 测试共 16 项通过。

## 4. 历史 3B Objective Screening 训练（已完成）

调度作业 `6579156` 已于 2026-08-25 02:11:45 JST 正常结束，`EXIT CODE=0`。后续 held-out 评测作业 `6585154` 因 3B 路线废弃，于 2026-08-25 13:03:29 JST 主动取消；已完成的 SFT/DPO/SimPO 辅助 judge 保留归档，未完成的 IPO/Anchored 和盲评不再补跑。

单个 MIG GPU 串行执行：

1. DPO；
2. SimPO；
3. IPO；
4. Positive-Anchored Preference。

控制变量：同一 SFT checkpoint、同一数据、同一 Attention LoRA placement、同一可训练参数量、同一 seed、同一训练 token 预算和同一评测流程。

每种 objective：

- 1 epoch；
- batch size 1；
- gradient accumulation 8；
- 158 optimizer steps；
- learning rate `5e-6`；
- cosine scheduler；
- 完整 384-pair baseline/final validation；
- 记录 chosen/rejected log-probability、reward margin、preference accuracy 和 loss。

四种 objective 的真实 GPU forward/backward/optimizer 冒烟已经全部通过。正式结果必须等四个 `best/final` checkpoint 均成功保存后才成立。

### 历史运行记录

- 作业开始：2026-08-24 23:46:55 JST；
- 2026-08-24 23:57 左右：四项冒烟全部通过，正式 DPO 到达 `20/158`；
- 首个真实训练区间约为 7–8 秒/optimizer step；
- 实际结束：2026-08-25 02:11:45 JST；
- 实际运行时间：2:24:51；
- 四个 `final` adapter 均保存并通过有限值检查。

## 5. 历史 3B Objective 选择规则（评测未完整执行）

训练完成后，先做固定 Hint held-out caption 生成，再做：

- 多维辅助 judge：Humor、Grounding、Originality、Specificity、Hallucination；
- 长度、模板化语言和多样性检查；
- 与 SFT baseline 的盲评 group-of-3 win rate；
- 以 image 为统计单位计算 bootstrap 95% confidence interval。

辅助 judge 只用于低成本筛选，最终结论以 held-out blind group preference 为主。Objective screening 后只保留最佳 1–2 个 objective。

## 6. Module Placement 低成本 Go/No-Go Pilot

固定最佳 objective，只比较五种近似预算匹配的 placement：

| Placement | 配置 | 可训练参数 |
|---|---|---:|
| Attention | `q/k/v/o_proj`, r=16 | 7.373M |
| MLP | `gate/up/down_proj`, r=5 | 7.050M |
| All-linear | Attention + MLP, r=4 | 7.483M |
| Gradient-selected | 30 个已选矩阵，r=73 | 7.400M |
| Random-selected | 同 family 数量、与 selected 分离，r=73 | 7.400M |

第一轮每种 placement 只跑一个 seed。只有 Gradient-selected 相对 Random-selected 出现正向 held-out 优势，才追加两个 seeds。

Go 条件必须同时满足：

1. 三个 seeds 下 Gradient-selected − Random-selected 的主要盲评指标均值为正；
2. image-bootstrap 95% CI 下界大于 0；
3. grounding 没有实质下降；
4. 每百万可训练参数的提升优于 Random 和简单 placement。

否则判定 No-Go：停止 Fisher、SVD、dynamic-rank 和大规模 layer-wise search，选择表现最好的简单 placement。Module gradient 只作为 analysis，不默认是主要创新。

## 7. 历史执行顺序（已废弃，禁止继续提交）

1. 完成作业 `6579156` 的四种 full-pair objective 训练；
2. 核验退出码、日志、四组 checkpoint 和训练稳定性；
3. 运行固定 Hint held-out 生成；
4. 生成多维辅助 judge 结果与盲评 packet；
5. 完成人工/盲评并选出最佳 1–2 个 objective；
6. 运行五种 budget-matched placement 的单 seed Go/No-Go pilot；
7. 仅在 gate 通过时追加两个 seeds，否则停止 module search；
8. 训练最终 Generator，并以至少三个 seeds 报告关键结果；
9. 不启动任何后续联合训练。

## 8. 完成判据与可复现记录

每个结果实验必须保存：

- resolved config、seed 和代码版本；
- 数据路径、行数和 hash；
- checkpoint；
- generation/evaluation 参数；
- trainable parameters、运行时间、显存和吞吐；
- chosen/rejected absolute log-probability 与 preference margin；
- 多维生成指标、盲评 win rate 和 image-level confidence interval。

不能仅凭训练 loss 或单一 judge 分数宣布 objective 更好。

## 9. 7B DPO 对 SFT 的正式三 Seed 盲评结果（2026-08-27）

确认性评测已完成。评测集为官方数据中未进入 SFT train、DPO train 或 DPO validation 的 47 张图片；每个模型使用 `20260827/20260828/20260829` 三个 generation seeds，每个模型、图片和 seed 生成三条 caption。共形成 141 个匿名 Group-of-3 trial。评审在读取模型映射 key 之前完成并锁定，决策文件 SHA-256 为 `72b9f98f99eeaeaaa33a8f284deb1d9a85aaba830285cf0e9d026fc3b29c70fb`。完全相同或无法可靠区分的组允许判为 Tie，统计时 Tie 计 0.5。

| 指标 | 结果 |
|---|---:|
| seed 20260827 DPO win score | 55.32%（19 win / 14 tie / 14 loss） |
| seed 20260828 DPO win score | 47.87%（20 / 5 / 22） |
| seed 20260829 DPO win score | 61.70%（26 / 6 / 15） |
| 三 seed 均值 | 54.96% |
| seed 标准差 / 方差 | 6.92 pp / 0.004791 |
| image-clustered bootstrap 95% CI | [46.45%, 63.12%] |
| 图片多数结果 | DPO win 21 / tie 11 / loss 15 |

绝对质量采用 `good / weak / bad`，避免把相对胜出误写成真正好笑。按每张图片三个 seeds 的中位标签统计：DPO 为 10 good、35 weak、2 bad；SFT 为 8 good、35 weak、4 bad。对应 majority-good rate 分别为 21.28%（Wilson 95% CI [11.99%, 34.90%]）和 17.02%（[8.89%, 30.14%]）。

结论：DPO 的相对点估计略高，并且绝对 `bad` 图片数更少，但主要 95% CI 仍跨过 50%，seed 间波动也明显。当前结果只能称为“有正向趋势”，不能称为 DPO 已稳定显著优于 SFT；尤其大多数图片仍为 `weak`。因此不据此启动大规模 module search，也不恢复任何 3B 或联合训练。下一步若继续，应优先增加独立盲评者或改进 preference pair 质量，再决定是否把 DPO 设为部署 checkpoint。

可复现文件：

- 匿名 trial：`results/7b_generator/dpo_test47_3seed_blind/trials.jsonl`
- 锁定判断：`results/7b_generator/dpo_test47_3seed_blind/codex_blind_decisions.json`
- 解盲报告：`results/7b_generator/dpo_test47_3seed_blind/codex_blind_report.json`
- 生成任务：`jobs/genkai_eval_7b_sft_vs_dpo_test47_3seed.pjm`（job `6603601`，exit code 0，运行 24 分 32 秒）

## 10. 独立盲评与 Preference Pair 改进（2026-08-27）

### 独立盲评

已为两位人工评审分别生成重新随机排序、独立 A/B 翻转、完全离线可用的 HTML 包。两份包使用不同 blind ID 与随机种子，不显示 SFT/DPO 身份；图片已内嵌，因此浏览器不需要读取仓库数据路径。每位评审完成后点击“导出 JSON”，不得交换答案，也不得查看 private key。

- `results/7b_generator/dpo_test47_independent_raters/public/human_1.html`
- `results/7b_generator/dpo_test47_independent_raters/public/human_2.html`
- private key 位于 public 目录之外：`results/7b_generator/dpo_test47_independent_raters/public_private_key.json`
- 局域网服务运行在 tmux `humor_blind_raters`，端口 `8766`。

收到两份 JSON 后使用 `scripts/report_independent_group_raters.py` 还原到统一匿名位置，报告 raw agreement、Cohen's kappa 与多数共识。当前 Codex 单评结果保留，但不能伪装成两名独立评审；确认性结论优先使用两名新增人工评审与预先锁定评审的共识，并明确报告 disagreement。

另外已生成两套可直接交给不同多模态大模型的材料，位于 `results/7b_generator/dpo_test47_independent_raters/llm_packages/llm_judge_1/` 与 `llm_judge_2/`。每套包含独立 `PROMPT.txt`、141 页完整 PDF、三个 47 页 PDF、对应 JSON response templates 与 API-friendly JSONL。两个 judge 的题目顺序、blind ID 和 A/B 翻转独立随机；rubric 保持相同，以免把 prompt 差异混入模型差异。使用说明见 `llm_packages/README_ZH.md`。大模型评审必须称为 multi-model judging，而不是独立人工评审。

`llm_judge_2` 已返回并通过 141/141 blind-ID、schema、类别和值域校验；冻结文件 SHA-256 为 `25a30f954ca2bda4a7ab608565ff6adb22fbdd5327309f17c9c331182424fbbd`。解盲后 DPO 的三-seed win score 为 53.19%、52.13%、47.87%，均值 51.06%，seed 标准差 2.81 pp；image-clustered 95% CI 为 [44.33%, 57.80%]，仍跨 50%。图片多数结果为 DPO win 22 / tie 5 / loss 20。

Judge 2 与已锁定 Codex 评审的 overall raw agreement 为 58.87%，Cohen's kappa 为 0.317；best-pick kappa 为 0.290，属于偏低一致性。其绝对标签在 282 个匿名 group 中给出 281 个 `weak/bad`、仅 1 个 `good`，出现明显量表压缩，因此目前不能用它的绝对 good rate 单独判断模型质量。该结果进一步说明 DPO 尚未被证明稳定优于 SFT。必须等待 `llm_judge_1`，再报告三评审多数共识；若总体 kappa 仍低，应把分歧项交给人工裁决，而不是平均掉主观差异。

`llm_judge_1` 随后也完成 141/141 项并冻结（SHA-256 `c0a6c5c2e429a1dffc137c7136db224e0db0a1303296f4a659e479e9de3ed0cd`）。三评审多数共识的 DPO score 为 53.90%，image-clustered 95% CI [47.87%, 59.93%]，仍跨 50%；overall Fleiss κ 只有 0.122。Judge 1 对匿名 A 有显著位置偏好，因此三评审多数票只能作为敏感性分析，不能当成人类共识。完整报告见 `docs/7B_DPO_MULTI_JUDGE_EVAL.md`。当前结论不变：Pilot16 DPO 未被证明优于 SFT；停止复杂 module search，只考虑固定配置的 Quality64 数据对照 pilot。

项目负责人随后委托 Codex 对 11 个无多数项进行重新盲化裁决。裁决前冻结哈希为 `12de2d38036b05b2cddbf6ca06316776188a173e7cb0d946478ad5a508adfa16`；裁决后 DPO score 为 53.19%，image-clustered 95% CI [46.81%, 59.57%]，图片多数为 22 win / 8 tie / 17 loss，DPO/SFT 都只有 2/47 majority-good images。结论不变，本轮 test47 盲评正式关闭。

### 改进后的官方 Preference Pairs

原 FullSplit64 每图优先选择最靠近 `z=3` 的 hard pairs，没有约束 chosen 自身必须位于排名头部。这有利于边界排序，却可能大量训练“两个 weak caption 哪个略好”。新版本不生成或修改标签，仍严格使用 NeurIPS 2024 官方 crowd ranking 方向，但改为：

1. chosen 必须位于该 contest 前 25%；
2. clear：chosen 前 10%、rejected 后 50%、`z>=4.5`，目标 24/image；
3. medium：chosen 前 20%、rejected 位于排名后 60% 且 `z>=3.5`，目标 24/image；
4. hard：chosen 前 25%、rank percentile gap 至少 20%、`z>=3`，目标 16/image；
5. 相对字符长度差继续不超过 0.35；caption 复用优先限制为 2 次；
6. validation/test 逐行保持原 FullSplit64 版本，避免同时改变训练和评测。

输出为 `data/processed/newyorker_published_dpo_pairs_quality64/`：17,297 pairs / 271 train images；clear 6,457、medium 6,473、hard 4,367（含 quota fallback）。contest 850 仍只有 17 对，不降低门槛补齐。数据 invariant、split isolation 与 surface-bias audit 均通过。

相对旧 FullSplit64，chosen 平均 crowd score 从 1.347 提高到 1.531，chosen rank 中位数从 638 提高到 91，平均 score margin 从 0.213 提高到 0.377；长度、emoji、网络模板和词汇多样性未出现明显 winner shortcut。该版本目前只是新的候选训练集，尚未提交 DPO。必须先完成独立盲评，确认现有 DPO 信号是否可复现，再决定进行一个固定配置的 Quality64 对照训练。

## 11. 论文依据

1. Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
2. Azar et al. (2024), *A General Theoretical Paradigm to Understand Learning from Human Preferences*, AISTATS 2024. https://proceedings.mlr.press/v238/azar24a.html
3. Meng et al. (2024), *SimPO: Simple Preference Optimization with a Reference-Free Reward*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/3a6bfa0d2b8cfce85f61f3c23c7f8b90-Abstract-Conference.html
4. Xu et al. (2024), *Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation*, ICML 2024. https://proceedings.mlr.press/v235/xu24h.html
5. Zhang et al. (2024), *Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
6. Hessel et al. (2023), *Do Androids Laugh at Electric Sheep? Humor Understanding Benchmarks from The New Yorker Caption Contest*, ACL 2023 Best Paper. https://aclanthology.org/2023.acl-long.41/
