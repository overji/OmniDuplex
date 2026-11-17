# scored-Pair：Probe 的配对内容综合分数

Probe 使用严格配对加权，不使用 Realtime 的 scored-F1 分母：

```text
scored_pair = Σ I(pre decision = NO) × I(post decision = YES) × q_post / N
```

`N` 是两侧推理均完成的配对数。RTD/PR 的 `q_post=response_score/2`；Agent 的
`q_post=(response_score+call_score)/4`，缺失或非法必要调用仍按 call_score=0。
pre 误答、post 沉默、无法解析任意一侧都使该对得 0。全沉默和始终回复均为 0；
内容全满分时等于 Pair 正确率，因此 scored-Pair 不超过 Pair 正确率。

缺一侧推理的配对只计入覆盖缺失，不补零。正确配对所需 Judge 缺失或失败时，
scored-Pair 为 null；错误配对上的 Judge 失败不阻塞此指标。没有完整配对时为 null。
Judge 总状态及条件内容均值仍可显示 incomplete，不代表已能计算的 scored-Pair 也必须缺失。

JSON 输出 `scored_pair`（[0,1]）、`scored_pair_version=scored_pair_v1`、`pair_count`、
`correct_pair_count` 和 `quality_sum`；报告显示百分比。各分组从配对和质量总和做微平均，
不平均组内百分比。既有 Pre/Post/Pair 和 TP 条件内容均值保持定义不变。

## 离线重算与导出

沿用现有命令名以兼容使用方式；`--mode probe` 的输出指标改为 scored-Pair：

```bash
python benchmark/scripts/recompute_scored_f1.py \
  --mode probe --data /path/to/original_all_qa.json \
  --manifest /path/to/probe_manifest.jsonl --run /path/to/probe_run \
  --judgments /path/to/judgments.jsonl --output /path/to/new_scored_pair.json
```

使用该次运行绑定的原版数据、manifest 和预测，不直接套用索引已变化的数据。
命令不调用 API、不加载模型、保存输入哈希，并拒绝覆盖已有输出。

新 portable 包为 Probe 保存 `scored_pair_population`：每组的完整配对数、正确配对数、
正确配对对应的 post probe IDs。原 Judge 请求和评分规则不变。
旧包只有 `scored_f1_population` 时无法恢复逐对正确性；新汇总输出 null 和重新导出提示，
不重命名旧分数，也不从旧总量推算。旧原始预测和 Judge 结果保留，可离线重新导出与重算。

Realtime、OmniPro 的 [scored-F1](scored_f1.md) 不变。
