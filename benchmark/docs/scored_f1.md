# scored-F1

用户批准以 **scored-F1** 替换原论文/新报告中的 C_all（全窗口内容召回）。名字不是同一指标的别名：分母发生了变化。

每个完整 Judge 结果先归一化：RTD/PR 为 response_score/2；Agent 为 (response_score+call_score)/4。缺失/非法必要调用的 call_score 仍为 0。原始五档分数不变。

令 Q 为原有一对一时间匹配得到的质量之和，N 为参考窗口数，M 为全部输出事件数：

```
scored_precision = Q / M
scored_recall    = Q / N
scored_f1        = 2 * Q / (N + M)
```

JSON 的 `scored_f1` 为 [0,1]，表格显示为百分数；`scored_f1_version=scored_f1_v1`。同时记录 `quality_sum`、`reference_count`、`prediction_count`，便于重算。N>0 且全沉默时为 0；无评测数据、缺少输出计数或相关 Judge 未完成时为 null，不伪造零分。

Realtime：M 包括所有新回复事件，包含重复、错误时间的额外回复。沿用既有时间匹配，不增加内容门控阈值，不重新选择最佳回复，不变更原始输出。单个回复跨秒续说仍只计一次；同一回复内部冗长不额外按 token 扣分。

Probe 使用独立的 **scored-Pair**，不再使用上述 F1 分母。严格配对公式、缺失策略与兼容说明见 [scored-Pair](scored_pair.md)。

Agent 保留现有 response/instruction 算术平均，不新增严格调用成功率门槛。scored-F1 是软质量 F1，不等于 OmniPro 原论文的内容正确性门控 F1。

## 使用与兼容

- 常规 `judge`、本地 AOI/OmniPro Judge、新导出的 portable Judge bundle 汇总包含 scored-F1。
- Realtime portable 输入包使用 `scored_f1_population` 冻结各组参考数与输出数。Probe 使用独立的 `scored_pair_population`，不能沿用旧的 Probe F1 分母。
- 旧包若缺少 M，不能只凭 Judge 均值推算 scored-F1。需用原预测重新导出含分母的包，或用 `paper/build_tables.py` 直接读取历史原始预测重算。不要重命名旧均值冒充新指标。
- 原始 `response_mean_all_windows`、`task_mean_all` 等字段暂留作兼容，仍是旧含义；用户可见的综合内容列改用 scored-F1。C_TP 继续用于诊断。
- `python paper/build_tables.py` 无需 API/GPU 即可重算当前论文全部表格与快照；原始预测与分数只读。

已有 Judge 结果可以纯离线重算，不重新请求 API（`--data` 必须是该次运行绑定的同版数据，而非索引已变化的新文件）：

```bash
python benchmark/scripts/recompute_scored_f1.py \
  --mode realtime --data /path/to/original_all_qa.json \
  --run /path/to/prediction_run --judgments /path/to/judgments.jsonl \
  --output /path/to/new_scored_f1.json
# Probe 改为 --mode probe，并加 --manifest /path/to/probe_manifest.jsonl；输出为 scored-Pair。
```

新输出校验预测与 Judge 的 sample/response 身份，保存全部输入 SHA256；已有文件不会覆盖。缺少相关 Judge 的位置显示 pending/null，而不是零分。
