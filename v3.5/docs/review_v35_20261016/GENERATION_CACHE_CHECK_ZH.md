# 真实 Qwen 生成与 KV cache 工程检查

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
