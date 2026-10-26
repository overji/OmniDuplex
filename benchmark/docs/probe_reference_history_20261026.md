# Probe：已处理回复历史与MiniCPM负侧重评

本轮增加显式的 `--history-mode reference --history-data DATA.json`，模型上下文列出此前已经处理的参考回复正文。
默认 `count` 保留旧协议，不静默改变正在运行的历史条件。

这是一项GT历史辅助的诊断实验，不是实际模型回复历史下的闭环能力测试。只有当前目标之前、
属于可评分响应的用户侧参考正文会进入提示；不传当前目标、未来答案、工具结果或调用参数。
每个历史回复的window start、trigger/confirmed时间和证据available_time都必须严格早于负截断点；
否则排除整个配对，并记录原因。原数据与清单用数据指纹、pair ID及参考索引绑定。

新增参数：

| 参数 | 行为 |
| --- | --- |
| `--history-mode count` | 历史默认，只告知已处理次数 |
| `--history-mode reference --history-data ...` | 显式列出已处理参考回复 |
| `--probe-side both` | 默认，运行pre/post两侧 |
| `--probe-side pre` | 独立负侧诊断，不生成正侧，不与旧正侧拼接评分 |
| `--skip-failed` | 不补跑错误请求；成功及已有错误均跳过 |

`run_config`记录history模式、历史数据哈希和侧别。条件不同必须用独立目录。
`probe score`会按保存的侧别及`history_exclusions.json`恢复评分范围。

## 本轮运行

- 结果目录：`benchmark/results/minicpmo_probe_gt_history_negative_20261026/`。
- 模型：MiniCPM-o-4.5，八卡、每卡一实例；任务拆分提示词，512输出token，保持原媒体及解码设置。
- 数据：原full_2110清单，同一负截断规则 `[0, window_start - 1s)`。
- 可评负前缀4586条；其中2628条有非空既往回复；8对因历史时序不安全而排除。
- 原MiniCPM count模式结果保留，独立重评不覆盖旧预测，也不重跑正侧。
- 当前Qwen队列暂停。原始源码已保存于本轮目录下`frozen_count_runtime/`，并检查它能复现Qwen既有配置指纹及13个任务计划。
- 使用已有transition/watchdog衔接：历史诊断结束后，原count条件队列从Qwen进度继续；不会在同一结果中混入新历史条件。
- 失败项只保留为缺失覆盖，不自动补跑，不调用付费内容judge。

独立监督入口为 `benchmark/scripts/run_probe_history_trial.py`，先跑一个包含非空历史的负侧smoke，
再启动八卡分片；各分片写完整配置及prompt日志，根目录聚合 `probe_metrics.json`、`probe_report.md`、
`probe_predictions.jsonl` 和 `probe_errors.jsonl`。`history_plan.json`记录分片和预期请求数。

比较时应在两种条件都成功完成的同一probe ID上计算负侧准确率，并分别报告有/无历史的子集与覆盖率。
新实验没有正侧结果，不能报告新条件的paired或balanced accuracy。
