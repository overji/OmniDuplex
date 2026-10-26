# MiniCPM-o-4.5 与 Gander 对比

快照时间：2026-10-23T13:30:37.379989+00:00。

相同样本比较：1514 条。MiniCPM 已完成 2107 条；Gander 已完成 1514 条，仍在补跑。

Agent 使用同一批 723 条 task_start 重评；旧 Agent 预测全部排除。RTD/PR 只比较双方均已完成的相同样本。

| 分组 | 模型 | QA | 窗口 | 输出 | 命中 | Recall | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | minicpmo | 1514 | 3267 | 2985 | 783 | 23.97% | 26.23% | 25.05% |
| overall | gander | 1514 | 3267 | 1354 | 522 | 15.98% | 38.55% | 22.59% |
| mode:rtd | minicpmo | 395 | 1767 | 1018 | 390 | 22.07% | 38.31% | 28.01% |
| mode:rtd | gander | 395 | 1767 | 469 | 182 | 10.30% | 38.81% | 16.28% |
| mode:pr | minicpmo | 396 | 743 | 846 | 246 | 33.11% | 29.08% | 30.96% |
| mode:pr | gander | 396 | 743 | 351 | 144 | 19.38% | 41.03% | 26.33% |
| mode:agent | minicpmo | 723 | 757 | 1121 | 147 | 19.42% | 13.11% | 15.65% |
| mode:agent | gander | 723 | 757 | 534 | 196 | 25.89% | 36.70% | 30.36% |

Agent 调用：两模型均为 0/723 条有可解析调用；schema 合法调用也均为 0。MiniCPM 有 18 个被解析器识别为调用尝试的事件，Gander 为 0。Agent 时间命中不等于工具调用成功。

这里只评估时间；没有合入本轮内容 LLM judge。MiniCPM 原生文本返回存在已知的条件性丢首 token 问题，目前缺少本轮真实 token 流记录，不能仅凭解析失败判定底层 token 从未生成过调用。未对原始输出做修复或补造。

## 来源

minicpmo:
- `benchmark/results/full_2110_dp6_20261021/realtime/minicpmo/predictions.jsonl`
- `benchmark/results/agent_task_start_v1_20261023/minicpmo/predictions.jsonl`
gander:
- `benchmark/results/full_2110_dp6_20261021/realtime/gander/predictions.jsonl`
- `benchmark/results/agent_task_start_v1_20261023/gander/predictions.jsonl`
- `benchmark/results/full_2110_phased_8gpu_20261023/suite/realtime/gander/predictions.jsonl`
