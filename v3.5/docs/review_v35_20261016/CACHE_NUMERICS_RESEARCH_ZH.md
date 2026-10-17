# Cache 数值差异：文献、实现证据与修复决策

## 当前判断

7097320 的 3/4 条件未通过预设数值门槛，不能改记成功；32/32 greedy argmax 一致也不能保证长序列或采样分布不变。no-memory 也失败，因此不能单凭该结果归罪于 bridge。NF4 是权重量化，这次并未设置 Transformers QuantizedCache；不能把权重量化与 KV cache 量化混为一谈。

## 查阅依据

1. [QLoRA, NeurIPS 2023](https://arxiv.org/abs/2305.14314)：NF4 权重量化与低精度计算的背景。论文的任务质量结果不保证任意前向形状的 logits 逐位一致。
2. [PyTorch Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)：数学等价的批量/切片计算不保证浮点结果相同，矩阵乘法精度与 reduction 次序需要记录。
3. [Transformers cache 文档](https://huggingface.co/docs/transformers/en/kv_cache)：cache 复用历史 K/V；QuantizedCache 是独立配置，不等同于 NF4 模型。

## 本机源码证据，比泛泛的数值解释更具体

已检查安装环境的 `bitsandbytes/autograd/_functions.py` 与 `bitsandbytes/backends/cuda/ops.py`：

- `matmul_4bit` 根据梯度需求、量化状态等选择路径；当前版本不能直接套用旧版本“单 token 一定 gemv”的解释。
- CUDA `_gemm_4bit_use_custom_cuda(device_index,dtype,M,N,K)` 明确依赖输入行数 M、矩阵尺寸和设备。
- 自定义 kernel 内还可选择 SIMT/MMA；fallback 会反量化后调用 F.linear。
- 缓存解码 M=1，完整前缀重算 M 约 140–162，可触发不同路径。这里只证明源码存在形状依赖；尚无 GPU 实测证明它解释全部 Qwen 差异。

## 已安排的定位实验

| 作业 | 对照 | 能回答什么 | 不能回答什么 |
|---|---|---|---|
| 7100860 | 未量化 FP32 计算 vs 已有 NF4/BF16，真实模型 cache/replay | 差异是否随联合精度设置变化 | 同时改变权重量化与计算精度，不能单独归因；BF16 读取后转 FP32 不恢复权重原始精度 |
| 7100871 | 同一随机线性层/同一末行输入，M=1 vs M=140/155；NF4、显式反量化 BF16、同权重 FP32 | 不含 cache、位置编码和 bridge 时，是否仍出现形状差异；同形状重复是否一致 | 合成权重不是实际 Qwen 激活，不能直接推出最终 logits 误差来源 |

第二项设 3 个固定种子、512/3584 两种方阵宽度，FP32 参考关闭 TF32；记录版本、设备、dispatch 启发式和源码哈希。它仅输出诊断量，不把“有非零误差”自动判成 bug。已完成脚本编译与 shell 语法检查；GPU 实测尚待结果。

## 修复顺序

1. 两项结果返回后核对精度、源码与相同前缀。若 FP32 通过且线性层能重现形状差异，再增加未量化 BF16 对照，以区分权重量化与算术精度。
2. 若需定位具体内核，使用实际模型相同输入/同权重的局部 linear replay，而非全局 monkeypatch 第三方库后直接宣称修复。
3. 研究基准优先使用经验证的精度路径；在显存允许时，未量化 BF16/FP32 是候选，但当前尚未验证完成，不能先切生产默认。
4. 若高精度仍失败，逐层比较 position_ids、attention mask、cache 长度、Rope 状态与 hook 输出，查找最早发生差异的位置。
5. 无 cache 完整重算可作为诊断参考/临时慢路径，但启用它并不修复缓存，也不能据此作速度比较。当前没有改生产 generate 默认。

## 评价设计的补充

原 max-logit 0.25 / KL 0.001 gate 保留。后续若制定部署容差，必须另立版本、说明验证集依据，不能追溯性把旧 fail 改 pass。

同时记录居中 logits 误差（softmax 不受全词表共同平移影响）、KL、top-1/top-2 间距、逐步 token 与完整生成变化。若两个 logits 向量的逐元素误差上界为 epsilon，且参考 top-1 与 top-2 的间距 > 2 epsilon，可保证该步 argmax 不变；这只是该前缀下的判据，不证明任意未来前缀稳定。

训练桥接/无需训练接口都共用底层 Qwen 推理，换 StateBridge 或 CIPHER 不会自动消除数值路径差异。应先澄清此工程问题，再比较通信机制。
