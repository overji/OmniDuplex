# MiniCPM / Gander / JoyAI 总体结果对比

快照：2026-10-24T12:40:16.350119+00:00。JoyAI 仍未完成，本报告不是最终全量结果。

RTD/PR 使用已有完成结果，Agent 全部使用最新纯文本分隔符重评测。完全排除旧 JSON/native Agent，不以旧结果补缺。
三者共同完成 1912 条 QA，仅在相同交集上计算以下总分；不对各任务 F1 简单平均。

MiniCPM/Gander 输入音视频，JoyAI 输入视觉；未加入内容 LLM judge。

| 任务 | 模型 | QA | 窗口 | 输出 | 命中 | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | minicpmo | 1912 | 4573 | 3835 | 1078 | 23.57% | 28.11% | 25.64% |
| overall | gander | 1912 | 4573 | 1426 | 554 | 12.11% | 38.85% | 18.47% |
| overall | joyai | 1912 | 4573 | 11875 | 1231 | 26.92% | 10.37% | 14.97% |
| mode:rtd | minicpmo | 651 | 2766 | 1699 | 608 | 21.98% | 35.79% | 27.23% |
| mode:rtd | gander | 651 | 2766 | 722 | 261 | 9.44% | 36.15% | 14.97% |
| mode:rtd | joyai | 651 | 2766 | 9615 | 956 | 34.56% | 9.94% | 15.44% |
| mode:pr | minicpmo | 634 | 1146 | 1298 | 388 | 33.86% | 29.89% | 31.75% |
| mode:pr | gander | 634 | 1146 | 554 | 225 | 19.63% | 40.61% | 26.47% |
| mode:pr | joyai | 634 | 1146 | 1456 | 234 | 20.42% | 16.07% | 17.99% |
| mode:agent | minicpmo | 627 | 661 | 838 | 82 | 12.41% | 9.79% | 10.94% |
| mode:agent | gander | 627 | 661 | 150 | 68 | 10.29% | 45.33% | 16.77% |
| mode:agent | joyai | 627 | 661 | 804 | 41 | 6.20% | 5.10% | 5.60% |

时间 Recall=命中窗口/参考窗口；Precision=命中/输出事件；闭区间内回复开始时间一对一匹配。时间分数不代表回复或 instruction 语义正确。

整体包含 RTD/PR 旧运行与 Agent 新运行，协议按任务分开；这不是全任务同一新提示协议重跑的结果。后续新 probe 和消融不混入本表。

## 来源

### minicpmo
- `benchmark/results/full_2110_dp6_20261021/realtime/minicpmo`
- `benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/minicpmo_task_text_agent_full`

### gander
- `benchmark/results/full_2110_dp6_20261021/realtime/gander`
- `benchmark/results/full_2110_phased_8gpu_20261023/suite/realtime/gander`
- `benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/gander_task_text_agent_full`

### joyai
- `benchmark/results/full_2110_phased_8gpu_20261023/suite/realtime/joyai`
- `benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/joyai_task_text_agent_full`
