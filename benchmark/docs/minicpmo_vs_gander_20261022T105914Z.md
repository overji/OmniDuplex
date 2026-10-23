# MiniCPM-o-4.5 与 Gander：同样本阶段对比

快照时间：2026-10-22T10:59:14.756063+00:00。共同完成 999 条 QA；原始样本索引交集，无新增筛选。

原始预测保持不变，逐条核对 task/video/question 和数据哈希，使用当前官方 timing scorer 重新统计。

## 同样本总体与三种交互类型

| 分组 | 模型 | QA | 窗口 | 输出 | 命中 | 命中率 | 时间精度 | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | minicpmo | 999 | 2424 | 2202 | 605 | 24.96% | 27.48% | 26.16% |
| overall | gander | 999 | 2424 | 976 | 370 | 15.26% | 37.91% | 21.76% |
| mode:rtd | minicpmo | 315 | 1432 | 792 | 306 | 21.37% | 38.64% | 27.52% |
| mode:rtd | gander | 315 | 1432 | 384 | 141 | 9.85% | 36.72% | 15.53% |
| mode:pr | minicpmo | 328 | 614 | 715 | 205 | 33.39% | 28.67% | 30.85% |
| mode:pr | gander | 328 | 614 | 289 | 121 | 19.71% | 41.87% | 26.80% |
| mode:agent | minicpmo | 356 | 378 | 695 | 94 | 24.87% | 13.53% | 17.52% |
| mode:agent | gander | 356 | 378 | 303 | 108 | 28.57% | 35.64% | 31.72% |

## 按能力分组（同一批样本）

| 能力 | 模型 | QA | 窗口 | 输出 | 命中 | 命中率 | 时间精度 | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attribute_perception | minicpmo | 83 | 184 | 172 | 45 | 24.46% | 26.16% | 25.28% |
| attribute_perception | gander | 83 | 184 | 67 | 36 | 19.57% | 53.73% | 28.69% |
| grounding | minicpmo | 93 | 239 | 210 | 49 | 20.50% | 23.33% | 21.83% |
| grounding | gander | 93 | 239 | 81 | 30 | 12.55% | 37.04% | 18.75% |
| action_perception | minicpmo | 127 | 328 | 288 | 78 | 23.78% | 27.08% | 25.32% |
| action_perception | gander | 127 | 328 | 127 | 59 | 17.99% | 46.46% | 25.93% |
| object_perception | minicpmo | 55 | 153 | 133 | 44 | 28.76% | 33.08% | 30.77% |
| object_perception | gander | 55 | 153 | 94 | 41 | 26.80% | 43.62% | 33.20% |
| event_understanding | minicpmo | 128 | 303 | 297 | 68 | 22.44% | 22.90% | 22.67% |
| event_understanding | gander | 128 | 303 | 102 | 31 | 10.23% | 30.39% | 15.31% |
| text_rich_understanding | minicpmo | 134 | 313 | 325 | 125 | 39.94% | 38.46% | 39.18% |
| text_rich_understanding | gander | 134 | 313 | 114 | 66 | 21.09% | 57.89% | 30.91% |
| object_counting | minicpmo | 51 | 171 | 101 | 31 | 18.13% | 30.69% | 22.79% |
| object_counting | gander | 51 | 171 | 38 | 18 | 10.53% | 47.37% | 17.22% |
| spatial_understanding | minicpmo | 72 | 206 | 164 | 53 | 25.73% | 32.32% | 28.65% |
| spatial_understanding | gander | 72 | 206 | 68 | 27 | 13.11% | 39.71% | 19.71% |
| prospective_reasoning | minicpmo | 57 | 114 | 107 | 19 | 16.67% | 17.76% | 17.19% |
| prospective_reasoning | gander | 57 | 114 | 55 | 16 | 14.04% | 29.09% | 18.93% |
| causal_reasoning | minicpmo | 78 | 132 | 135 | 22 | 16.67% | 16.30% | 16.48% |
| causal_reasoning | gander | 78 | 132 | 62 | 17 | 12.88% | 27.42% | 17.53% |
| emotion_recognition | minicpmo | 57 | 127 | 122 | 27 | 21.26% | 22.13% | 21.69% |
| emotion_recognition | gander | 57 | 127 | 116 | 10 | 7.87% | 8.62% | 8.23% |
| action_counting | minicpmo | 64 | 154 | 148 | 44 | 28.57% | 29.73% | 29.14% |
| action_counting | gander | 64 | 154 | 52 | 19 | 12.34% | 36.54% | 18.45% |

## 各自已完成数据（分母不同，不作直接排名）

| 分组 | 模型 | QA | 窗口 | 输出 | 命中 | 命中率 | 时间精度 | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | minicpmo | 2107 | 4935 | 4486 | 1230 | 24.92% | 27.42% | 26.11% |
| overall | gander | 999 | 2424 | 976 | 370 | 15.26% | 37.91% | 21.76% |

## 解释与可比性限制

命中率衡量参考窗口覆盖，时间精度惩罚额外/重复输出，F1 综合二者。时间匹配使用包含端点的一对一输出起始时间，不包含内容正确性判分。

两者使用同一数据、基础提示词、1 FPS、输入区间结束时间戳、关闭滑窗；没有工具返回文本输入或后续完成答复评分。

MiniCPM 的 Agent 输出协议为 structured，Gander 保留官方原生工具调用语法；每步 token 预算分别为 20 和 11（Gander 原生 Unit-8 语法）。因此这属于当前各自适配配置的比较，不能解释为相同解码格式和预算下的消融。

Gander 尚未全量完成；Agent 指标只比较回应时间，不代表调用成功率，也没有 LLM judge 内容或调用正确性分数。

复算 JSON 和交集索引：`benchmark/results/comparisons/minicpmo_vs_gander_20261022T105914Z`。
