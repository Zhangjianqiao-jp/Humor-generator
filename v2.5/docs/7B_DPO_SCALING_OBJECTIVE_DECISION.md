# 7B Full Quality64 DPO Scaling：Objective 决策

日期：2026-08-28（JST）

## 决策摘要

完整Quality64 DPO已完成17,297 pairs / 2,163 optimizer steps。当前证据不支持继续放大vanilla DPO。下一项低成本训练应优先测试：

```text
conditional image preference + chosen anchor
```

即mDPO论文所针对的两个组成问题。IPO保留为偏好噪声的次级对照，不作为第一优先。任何新训练前，先对SFT与DPO各阶段checkpoint完成24-image × 3-seed生成层validation；pairwise log-probability指标不能代替真实caption质量。

## 一、Scaling结果

| Step | Image-mean DPO loss | Reward accuracy | Policy accuracy | Reward margin | Chosen logp/token | Rejected logp/token |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.692144 | 0.591146 | 0.596354 | 0.002056 | -2.906199 | -3.245959 |
| 158 | 0.688950 | 0.585938 | 0.601562 | 0.008769 | -2.912080 | -3.256367 |
| 632 | 0.678750 | 0.588542 | 0.606771 | 0.034363 | -2.945700 | -3.307288 |
| 1264 | 0.673504 | 0.588542 | 0.609375 | 0.055417 | -2.996282 | -3.371623 |
| 2163 | 0.670885 | 0.593750 | 0.601562 | 0.064474 | -3.012520 | -3.393785 |

从step 0到2163：

- image-mean loss改善`-0.021259`，24-image bootstrap 95% CI `[-0.035075,-0.008001]`；
- reward margin增加`+0.062418`，CI `[+0.034134,+0.090704]`；
- reward accuracy只变化`+0.002604`，CI `[-0.054688,+0.059896]`；
- policy accuracy只变化`+0.005208`，CI `[-0.010417,+0.020833]`；
- chosen logp/token下降`-0.106321`，CI `[-0.124138,-0.090440]`，24/24图片均下降；
- rejected logp/token下降`-0.147827`，CI `[-0.166650,-0.128621]`，24/24图片均下降。

解释：DPO主要通过让rejected下降得比chosen更快来扩大相对margin；它没有建立清晰的ranking-accuracy增益，而且chosen绝对似然发生系统性下降。这正是chosen-anchor/RPO类正则应进入下一Pilot的触发条件。不能仅凭DPO loss下降把step2163称为最终最佳生成模型。

## 二、7B图像条件诊断

只读作业`6631995`使用最终best adapter，在24张validation图片上各选一个pair，保持同一compact Hint与chosen/rejected caption，只把正确图片替换为确定性的错误图片。Test47未读取。

结果：

- mean `M(correct)-M(shuffled) = -0.0279`；
- median `-0.2351`；
- 正值12/24，负值12/24；
- image-bootstrap 95% CI `[-2.2840, 2.2603]`。

该低样本诊断不能证明模型完全忽略图片，但没有发现正确图片带来任何preference-margin优势。固定Hint已经携带图片信息，因此该结果具体说明：当前caption policy可能依赖Hint/语言shortcut，而没有独立核验原图。conditional image preference因此进入高优先级。

## 三、三种方向的选择

### IPO：暂不优先

官方crowd preference具有主观噪声，IPO理论上相关；但当前最直接的两个观测失败是chosen likelihood下降和正确图片条件不敏感。IPO本身不直接保证chosen概率上升，也不显式强化图片条件，因此只作为后续噪声鲁棒性对照。

### RPO / chosen anchor：需要

24/24图片的chosen per-token logp均下降，且下降随训练规模单调扩大。下一Pilot必须包含positive/chosen约束。项目当前`anchored`实现是DPO加chosen NLL的实验性CPO-style版本，不得误称为RPO或mDPO的逐行复现；正式实验需要明确数学定义并校准anchor梯度比例。

### Conditional mDPO：优先，并与anchor组合

当前7B在固定Hint下没有可检测的正确图片margin优势；mDPO正是通过response preference之外的image preference防止多模态模型依赖语言条件，并同时提出reward anchor处理chosen likelihood下降。两个触发条件同时出现，因此首选是组合方法，而不是只做其中一半。

## 四、下一步最小实验

1. 先生成并盲评SFT、step158、step632、step1264、step2163：24 validation images × 3 generation seeds × 每组3 captions；报告image-clustered win rate、CI、seed variance与`good/weak/bad`。
2. 如果生成层确认完整DPO或某中间checkpoint优于SFT，以相同SFT起点和固定MLP LoRA做最多632-step objective pilot：
   - vanilla DPO（现有step632作为已完成基线）；
   - DPO + chosen anchor；
   - DPO + conditional image preference；
   - conditional + anchor（mDPO目标方向）。
3. 固定数据、steps、LoRA、seed和训练预算，只改变loss组成。先做1 seed筛选；胜出的组合再做3 seeds。
4. IPO只有在上述方法仍显示pair噪声敏感、seed不稳定或过拟合时进入，不与第一轮四项同时膨胀。
5. Test47继续封存；只有validation生成层改善才允许一次最终测试。

## 参考论文

- Rafailov et al. (2023), *Direct Preference Optimization*. https://arxiv.org/abs/2305.18290
- Azar et al. (AISTATS 2024), *A General Theoretical Paradigm to Understand Learning from Human Preferences*（ΨPO/IPO）. https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html
- Wang et al. (2024), *mDPO: Conditional Preference Optimization for Multimodal Large Language Models*. https://arxiv.org/abs/2406.11839
- NeurIPS 2024, *Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer*（RPO）. https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa69e968b7319fd42524febd41475fb3-Abstract-Conference.html
