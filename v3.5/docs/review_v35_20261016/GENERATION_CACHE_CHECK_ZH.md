# 真实 Qwen 生成与 KV cache 工程检查

## 7097320 实测：数值门槛未通过

2026-10-17 01:09:25–01:12:02 JST，退出码 1；原始结果保留，不放宽门槛。

| Cluster | 条件 | max logits 误差 | max KL | 判定 |
|---|---|---:|---:|---|
| nycc_555 | no-memory | 0.207031 | 0.000511 | pass |
| nycc_555 | 初始 bridge | 0.281250 | 0.002046 | fail |
| nycc_788 | no-memory | 0.312500 | 0.001521 | fail |
| nycc_788 | 初始 bridge | 0.250000 | 0.001303 | fail |

32/32 步 argmax 相同，prefill 后均为单 token 解码，hook 清除检查均通过。18/18 输出通过严格格式解析，各 cluster/condition 三次生成内未出现完全重复的 caption/raw output，也没有 parser 识别的已知拒答。这不排除语义重复或未识别拒答，不构成幽默质量结果。

基线本身也失败，故尚不能把差异归因于 bridge。NF4/BF16 下不同前向形状的数值误差是一项待验证解释，不是已确认根因。每个条件重放自己的相同前缀；不同条件间的 KL 大小不是严格的同前缀配对因果比较。

### 新诊断 7100860

已提交未量化、FP32 计算、仅 cache 的对照，输出独立目录 `results/engineering_smoke/v35_generation_cache_fp32_20261017`，原门槛不变。它先按现有 loader 读取 BF16 权重再转 float32，**不恢复权重原始精度**；同时改变了量化与计算精度，只能定位这一联合因素，不能单独区分二者。

若仍失败，检查位置编码/attention 实现及同形状确定性；若通过，则记录当前量化部署路径未满足原数值门槛，不能反过来把 7097320 改成 pass。后续需要进一步分解 BF16 与量化影响再决定部署精度。

以下为原提交方案，排队状态是历史记录。

## 范围

GPU job 7097320 已提交，结果待验证；调度器预计 2026-10-17 01:10 JST 启动，时间可能变化。使用当前 v3.5 selected-plan 配置、固定 Qwen revision 和 4-bit/BF16 骨干；bridge 为固定随机种子的初始权重，没有加载训练 checkpoint。仅使用训练集 nycc_555、nycc_788，不使用测试集，不调用付费 API。

## 检查设计

- 对 no-memory 与 initial-latent 两条件，各生成最多 8 个 greedy token，保留未经过 sampling processors 的原始 logits。
- 对生成的每个相同 token 前缀关闭 cache 重算。检查全部 logits 有限、逐步 argmax 一致、最大绝对误差 <= 0.25、KL <= 0.001。这是预先设定的工程容差，不是数学恒等证明；基线和 bridge 使用同样容差。
- 检查实际 decoder 输入长度为完整 prompt 后逐步单 token；检查 bridge 注入非零有限，退出后 hook 数量恢复。
- 使用生产 backend.generate，三个条件（同一计划文本/no-memory/初始 bridge），两个训练 cluster，每个三个种子，最多 192 个新 token，共 18 次生成。
- 按严格 parser 记录格式、已知模式拒答，以及每个 cluster/condition 内有效 caption 和 raw output 的重复。已知拒答检测不等于完整语义拒答识别，长度上限也可能导致格式失败。

## 解释限制

cache 检查通过不代表已训练模型的质量；两个训练样本、八个解码步不能代表所有长度、batch 或视觉输入。格式和重复为描述统计，不能作为幽默质量评估。未训练 bridge 的输出不能用来否定或证明训练后效果。

## 运行与恢复

- 脚本：`scripts/check_selected_plan_generation.py`
- PJM：`jobs/v35_generation_cache_smoke.pjm`
- 结果：`results/engineering_smoke/v35_generation_cache_20261017/report.json`
- 原始生成逐条写入同目录 `generations.jsonl`，仅本地保存，不默认发布。
- 输出目录必须不存在，禁止覆盖；记录运行源码/配置哈希，并在结束时核对未发生修改。
- 日志：`v35_generation_cache_smoke.pjm.7097320.out`，genkai0002 上的 tmux 窗口 `v35-recovery:cache-7097320`。
- 已完成静态验证：Python 编译、模块导入及 --help、shell 语法检查。GPU 结果尚未计为通过。

## 论文参考

[Vaswani et al., Attention Is All You Need, NeurIPS 2017](https://arxiv.org/abs/1706.03762)：Transformer 自回归注意力背景。具体缓存实现是否正确必须以本次运行证据判断，论文不提供本脚本容差的理论保证。
