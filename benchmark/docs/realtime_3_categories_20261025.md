# Realtime 三大认知层级结果

采用用户指定的 T1–T12 分类。沿用 `realtime_12_tasks_20261025.json` 的三模型共同完成2008条QA、4669个窗口快照。

| 编号 | 任务（中 / 英） | 认知层级 |
| --- | --- | --- |
| T1 | 属性感知 / Attribute Perception | 感知 |
| T2 | 时间定位 / Grounding | 感知 |
| T3 | 动作感知 / Action Perception | 感知 |
| T4 | 物体感知 / Object Perception | 感知 |
| T5 | 情绪识别 / Emotion Recognition | 感知 |
| T6 | 事件理解 / Event Understanding | 理解 |
| T7 | 文本内容理解 / Text-Rich Understanding | 理解 |
| T8 | 物体计数 / Object Counting | 理解 |
| T9 | 动作计数 / Action Counting | 理解 |
| T10 | 空间理解 / Spatial Understanding | 推理 |
| T11 | 前瞻推理 / Prospective Reasoning | 推理 |
| T12 | 因果推理 / Causal Reasoning | 推理 |

各类先累计窗口、输出和命中，再计算时间 Recall、Precision、F1；不平均子任务F1。仅评开口时间，未做内容 LLM judge。

| 类别 | 模型 | QA | 窗口 | 输出 | 命中 | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 感知 | minicpmo | 863 | 2027 | 1707 | 446 | 22.00% | 26.13% | 23.89% |
| 感知 | gander | 863 | 2027 | 705 | 266 | 13.12% | 37.73% | 19.47% |
| 感知 | joyai | 863 | 2027 | 5463 | 529 | 26.10% | 9.68% | 14.13% |
| 理解 | minicpmo | 749 | 1830 | 1562 | 492 | 26.89% | 31.50% | 29.01% |
| 理解 | gander | 749 | 1830 | 459 | 219 | 11.97% | 47.71% | 19.13% |
| 理解 | joyai | 749 | 1830 | 4797 | 562 | 30.71% | 11.72% | 16.96% |
| 推理 | minicpmo | 396 | 812 | 707 | 156 | 19.21% | 22.07% | 20.54% |
| 推理 | gander | 396 | 812 | 285 | 80 | 9.85% | 28.07% | 14.59% |
| 推理 | joyai | 396 | 812 | 1731 | 147 | 18.10% | 8.49% | 11.56% |

MiniCPM/Gander 输入音视频，JoyAI 仅视觉。RTD/PR沿用原运行，Agent使用最新task-text重评测；LiveStar覆盖不足，未纳入。
