# Bedrock 盲评接口小试：可调用，但判定不稳定

悉尼区域配置的 `au.anthropic.claude-sonnet-4-6` 连通性检查成功，随后执行 8 对候选、每对两种顺序的 16 次评价。模型配置保持本地；没有将密钥或原始候选公开提交。选取完整盲评包中按 pair_id 排序前八对，属于流程小试，不是平衡抽样或质量检验。

使用独立请求、temperature=0、maxTokens=256；每次只给描述、两个 caption 和格式有效性，不给方法标签、计划或生成 explanation。结果 16/16 JSON schema 有效、16/16 end_turn。用量 4222 输入 token、1654 输出 token。计费金额未查询，不推算。

映射回方法身份后：5/8 对顺序一致，3/8 对不一致。这个比例只描述八对小试，不估计普遍偏差；也不能把不一致全部归因于位置偏差，服务随机性和判断模糊都可能参与。它足以说明此处不能直接采用单次偏好作可靠标签。没有给方法排名，也没有宣布任何方法幽默质量更好。

原始记录本地路径：`results/preflight/v35_effectiveness_20261017/blind_full16/pilot_judgments/`。只发布带输入/结果哈希的 `blind_judge_pilot_validation.json`。脚本 `scripts/judge_interface_review.py` 默认离线校验，执行需显式 --execute，限制请求数，逐条落盘；目前不支持自动恢复，重跑必须使用新输出目录并明确新样本范围，避免重复收费。

后续先复核不一致案例、以独立人工偏好或适用的已知排序样本校准评审，再决定是否扩大；正式图像评审仍需接入原图，当前描述评价不等于 HOMER 官方协议。对齐 GPU 作业 7103017 仍独立排队，这次 API 小试没有改变其脚本。

参考：[Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, NeurIPS 2023](https://arxiv.org/abs/2306.05685)，讨论位置、冗长、自偏好等评审限制。交换顺序是诊断，不能保证自动消除偏差。
