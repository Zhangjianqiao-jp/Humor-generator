# Human validation packet / 人工校验包

1. 先打开 `preference_pairs_quick_gate_blind.html`：每张训练图抽一个最低 margin pair，共 61 项。
2. 若 quick gate 可接受，再打开 `preference_pairs_blind.html` 完成全部 485 项。
3. 打开 `best_of_n_blind.html`，复核 24 张 held-out 图片上的 joint/direct 结论。
4. 网页自动把进度保存在浏览器 localStorage；请定期点击 Export JSONL。
5. 冻结导出的标注文件以后，才能打开 `blind_key.json` 解盲。
6. 不要把辅助 judge 分数或原始排名直接当作 preference 标签。

不使用网页时，可以填写对应的空白 CSV。详细标准见 `VALIDATION_GUIDE.md`。
