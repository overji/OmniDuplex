# MiniCPM-o-4.5、Gander 与 JoyAI 对比

快照时间：2026-10-24T01:11:31.981397+00:00。

只比较三者共同完成的 1973 条 QA。MiniCPM、Gander 各完成 2107 条；JoyAI 完成 1973 条，因两个 worker 的 HTTP 502 中断缺少 134 条。

旧 Agent 预测全部排除，MiniCPM/Gander 使用 task_start 重评；JoyAI 同样使用 task_start schema。没有修复或补造调用。

**输入条件：MiniCPM 和 Gander 输入音视频；当前 JoyAI 原生适配器只输入视觉帧和文本，没有音频。** 因此这是当前适配条件下的系统比较，不是完全相同模态输入下的模型能力比较。

| 任务 | 模型 | QA | 窗口 | 输出 | 命中 | Recall | Precision | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | minicpmo | 1973 | 4634 | 4061 | 1135 | 24.49% | 27.95% | 26.11% |
| overall | gander | 1973 | 4634 | 1755 | 672 | 14.50% | 38.29% | 21.04% |
| overall | joyai | 1973 | 4634 | 12686 | 1331 | 28.72% | 10.49% | 15.37% |
| mode:rtd | minicpmo | 651 | 2766 | 1699 | 608 | 21.98% | 35.79% | 27.23% |
| mode:rtd | gander | 651 | 2766 | 722 | 261 | 9.44% | 36.15% | 14.97% |
| mode:rtd | joyai | 651 | 2766 | 9615 | 956 | 34.56% | 9.94% | 15.44% |
| mode:pr | minicpmo | 634 | 1146 | 1298 | 388 | 33.86% | 29.89% | 31.75% |
| mode:pr | gander | 634 | 1146 | 554 | 225 | 19.63% | 40.61% | 26.47% |
| mode:pr | joyai | 634 | 1146 | 1456 | 234 | 20.42% | 16.07% | 17.99% |
| mode:agent | minicpmo | 688 | 722 | 1064 | 139 | 19.25% | 13.06% | 15.57% |
| mode:agent | gander | 688 | 722 | 479 | 186 | 25.76% | 38.83% | 30.97% |
| mode:agent | joyai | 688 | 722 | 1615 | 141 | 19.53% | 8.73% | 12.07% |

## Agent 调用（相同样本）

| 模型 | Agent QA | 有可解析调用的 QA | 有 schema 合法调用的 QA | 调用尝试事件 |
|---|---:|---:|---:|---:|
| minicpmo | 688 | 0 | 0 | 18 |
| gander | 688 | 0 | 0 | 0 |
| joyai | 688 | 0 | 0 | 2 |

这里只评估回复发生时间和显式调用格式，不代表回复内容正确；未合入内容 LLM judge。时间命中不等于调用成功。MiniCPM 原生文本返回有已知的条件性首 token 丢失问题，尚不能从保存文本断言底层 token 从未生成调用。

## 数据来源

minicpmo:
- `benchmark/results/full_2110_dp6_20261021/realtime/minicpmo/predictions.jsonl`（nonagent）
- `benchmark/results/agent_task_start_v1_20261023/minicpmo/predictions.jsonl`（agent）
gander:
- `benchmark/results/full_2110_dp6_20261021/realtime/gander/predictions.jsonl`（nonagent）
- `benchmark/results/agent_task_start_v1_20261023/gander/predictions.jsonl`（agent）
- `benchmark/results/full_2110_phased_8gpu_20261023/suite/realtime/gander/predictions.jsonl`（nonagent）
joyai:
- `benchmark/results/full_2110_phased_8gpu_20261023/suite/realtime/joyai/predictions.jsonl`（all）
