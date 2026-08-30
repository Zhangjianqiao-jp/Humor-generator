# New Yorker compact-v2 SFT 结果

更新时间：2026-08-12 JST。本文只记录 SFT；本轮未运行 DPO 或其他偏好优化。

## 数据与可复现性

- 来源：`yguooo/newyorker_caption_ranking`，固定 revision `1cd70477b6a99a473690a25a2fed359f75184c64`。
- 使用限制：CC BY-NC 4.0；数据卡限制为学术研究用途，不得直接用于商业训练或产品。
- 选择：逐漫画按源 `rank`（0 最佳）保留有效 caption 的前 3%，不是跨漫画比较不可比的分数。
- planner：train/validation/test 为 79/24/24 张互斥图片。
- captioner：train/validation/test 为 13,190/3,990/4,415 条 caption；训练和验证按图片平衡采样。
- `scripts/audit_newyorker_compact_sft.py` 从 721,955 条原始有效记录独立重算，确认 21,595 个目标恰好是逐图 top 3%；127 张图片均可解码，split 无图像交叉，gold-caption prompt 泄漏为 0。
- 带各 JSONL SHA-256 的机器报告：`outputs/newyorker_compact_v2_data_audit.json`。

## 7B humor planner

| 项目 | 结果 |
| --- | --- |
| 基座 | Qwen2.5-VL-7B-Instruct |
| 方法 | 4-bit NF4 QLoRA，冻结基座 |
| LoRA | rank 8，q/k/v/o projections，5,046,272 个元素（约 0.1074% 可训练） |
| 正式作业 | `6455356`，单个 12 GB MIG |
| 训练 | 30 epochs，300 optimizer steps |
| runtime | 6,038 秒（约 1:40:38） |
| train loss | 1.094 |
| final validation loss / PPL | 0.879298 / 2.409208 |
| 峰值已分配显存（日志） | 9.661 GB |
| 截断 | train=0，validation=0 |
| adapter | best 与 final 均为 step 300；rank/A-B 配对正确，全部 tensor finite |
| 最终固定生成 | 5 张唯一 validation 图片，5/5 三字段 schema 合法，0 prompt 泄漏 |

定性限制：模型稳定学会 `ANCHOR/CONTRAST/ANGLE` 结构，并正确识别了巨浪人脸样例；但仍把巨大纸卷误认成轮胎、漏掉会议中的植物，对治疗椅上的汽车和钢琴场景也不完整。因此上述 loss 与 schema 结果证明 SFT 收敛和接口可用，不证明所有视觉笑点均已正确 grounding。

## 3B plan-conditioned captioner

| 项目 | 结果 |
| --- | --- |
| 基座 | Qwen2.5-VL-3B-Instruct |
| 方法 | 4-bit NF4 QLoRA，冻结基座；每个 image/epoch 随机抽取一个高排名 caption |
| LoRA | rank 16，q/k/v/o projections，7,372,800 个元素（约 0.3612% 可训练） |
| 正式作业 | `6455357`，单个 12 GB MIG，exit code 0 |
| 训练 | 80 epochs，800 optimizer steps |
| runtime | Trainer 约 12,800 秒；PJM wall time 12,858 秒（3:34:18） |
| train loss | 2.973 |
| final validation loss / PPL | 2.961170 / 19.320567 |
| best validation loss / PPL | step 400：2.956335 / 19.227375 |
| 峰值已分配显存（日志） | 4.534 GB |
| 截断 | train=0，validation=0 |
| adapter | best=step 400、final=step 800；两者 rank/A-B 配对正确，全部 tensor finite |
| 最终固定生成 | 8 张唯一 validation 图片，8/8 非空单句，0 prompt 泄漏 |

完整 validation 曲线（step: loss / PPL）：100: 2.981231 / 19.712067；200: 2.975635 / 19.602058；300: 2.989897 / 19.883644；400: 2.956335 / 19.227375；500: 2.964877 / 19.392324；600: 2.956532 / 19.231153；700: 2.968808 / 19.468699；800: 2.961170 / 19.320567。后半段未继续改善，因此部署和级联测试使用 step-400 best，而不是 final。

定性限制：captioner 能在部分例子中使用 plan 里的实体和场景（如接受心理治疗的汽车、伊甸园禁果、巨浪人脸），但也会输出流畅而泛化、与异常点联系较弱的句子。低 validation loss 和单句 schema 合法只说明条件语言建模收敛，不等于 caption 足够好笑。

## best 7B → best 3B 真实级联

- 作业 `6455911` 使用一个 12 GB MIG，wall time 2:52，exit code 0；未使用 gold plan。
- 7B 对 8 张唯一 test 图片生成 8/8 合法三字段 plan，0 prompt 泄漏。
- 3B 接收这些生成 plan，为每图生成 3 个候选，共 24/24 非空候选，0 prompt 泄漏。
- 文件：`outputs/newyorker_compact_v2_pipeline/planner_test_generations.jsonl`、`captioner_inputs_from_generated_plans.jsonl`、`captioner_test_generations.jsonl`。
- 定性上，planner 对客厅大象、末日废墟对话、骑士带领商务人士等场景抓住了主要异常；但把头顶微型办公室看成显示器头、把火箭车理解成被导弹击中的车，并漏掉餐厅 paparazzi。3B 候选中有图像相关且可用的句子，也有明显牵强或泛化的句子。因此本轮证明了两模型接口与训练产物可稳定运行，不宣称已经解决幽默质量。

## 方法依据

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022: https://arxiv.org/abs/2106.09685
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023: https://arxiv.org/abs/2305.14314
- Hessel et al., *Do Androids Laugh at Electric Sheep? Humor “Understanding” Benchmarks from The New Yorker Caption Contest*, ACL 2023 Best Paper: https://aclanthology.org/2023.acl-long.41/
- Hessel et al., *The New Yorker Caption Contest Dataset*, NeurIPS 2024 Datasets and Benchmarks: https://proceedings.neurips.cc/paper_files/paper/2024/hash/e297fb6cd1690ee5b39c5bb4c58ad801-Abstract-Datasets_and_Benchmarks_Track.html
