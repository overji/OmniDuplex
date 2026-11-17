# Counting 人工标注审查报告

已从远程 gallery 下载原始标注及对应源 QA。本报告检查标注完整性、response 指纹和评分统计，未重新观看全部视频；失败原因依据人工二元评分分类，不推断未记录的修改理由。

## 完成度

- 208 条 QA，127 个视频，539 个可评分 response；208 条 QA、539 个 response 均完成两项评分。
- 539 条评分全部匹配当前远程源数据的 response 指纹，失效记录 0 条。
- 仅发现标注员 A 的此批记录，无 B 的配对评分，不能计算标注员间 Kappa 或相关性。

## Response 通过率

| 分组 | QA | Response | 时间通过 | 内容通过 | 两项均通过 |
|---|---:|---:|---:|---:|---:|
| 全部 | 208 | 539 | 507/539（94.1%） | 502/539（93.1%） | 495/539（91.8%） |
| object_counting | 92 | 260 | 233/260（89.6%） | 225/260（86.5%） | 221/260（85.0%） |
| action_counting | 116 | 279 | 274/279（98.2%） | 277/279（99.3%） | 274/279（98.2%） |
| rtd | 72 | 338 | 322/338（95.3%） | 316/338（93.5%） | 314/338（92.9%） |
| pr | 62 | 124 | 111/124（89.5%） | 113/124（91.1%） | 110/124（88.7%） |
| agent | 74 | 77 | 74/77（96.1%） | 73/77（94.8%） | 71/77（92.2%） |

## 问题分布

- 495 个 response 两项均通过；44 个至少一项失败：仅时间失败 7、仅内容失败 12、两项均失败 25。
- 186/208 条 QA 的全部可评分 response 均通过（89.4%）；22 条 QA 至少一项失败，涉及20个视频。
- 22 条问题 QA 中，物体计数 17 条、动作计数 5 条；RTD 10 条、PR 6 条、Agent 6 条。
- 这反映该计数子集的参考标注质量，不是模型推理正确率，也不能外推整个 benchmark。

## 与当前 AOI-Bench 对齐

- 当前 canonical AOI-Bench 共 1,960 条；按视频、问题、任务、能力、交互类型匹配，远程208条中187条仍存在，且responses完全一致，无歧义匹配。
- 另21条当前已不存在，不能使用旧sampleIndex直接回写。
- 当前仍存在的187条中，168条全部通过，19条至少一项失败。
- 本次只下载和分析，没有修改canonical数据、远程评分或实验结果。

## 需要复核的 QA

索引均为远程gallery的0起始索引。完整44条问题response及原文见 [flagged_responses.json](flagged_responses.json)。

| Gallery QA索引 | 能力 | 交互类型 | 时间失败response索引 | 内容失败response索引 | 当前Benchmark索引 |
|---:|---|---|---|---|---|
| 0 | object_counting | rtd | [] | [0, 1] | [17] |
| 2 | object_counting | agent | [] | [0] | [19] |
| 9 | object_counting | rtd | [] | [1] | [] |
| 38 | object_counting | agent | [0] | [] | [318] |
| 39 | object_counting | rtd | [0] | [] | [319] |
| 42 | action_counting | rtd | [1] | [1] | [350] |
| 44 | object_counting | agent | [] | [0] | [377] |
| 46 | object_counting | pr | [3] | [] | [389] |
| 51 | action_counting | agent | [0] | [0] | [424] |
| 62 | object_counting | pr | [0, 1, 2, 3, 4, 5, 6, 7, 8] | [0, 1, 2, 3, 4, 5, 6, 7, 8] | [521] |
| 63 | object_counting | rtd | [] | [0, 1, 2] | [562] |
| 67 | action_counting | pr | [1] | [] | [577] |
| 71 | action_counting | rtd | [2] | [] | [599] |
| 72 | object_counting | rtd | [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | [] |
| 109 | object_counting | pr | [] | [0] | [877] |
| 115 | object_counting | pr | [0] | [] | [932] |
| 122 | object_counting | rtd | [2] | [2, 3] | [996] |
| 128 | action_counting | agent | [0] | [] | [1082] |
| 132 | object_counting | rtd | [5] | [5] | [1118] |
| 138 | object_counting | pr | [0] | [0] | [1136] |
| 155 | object_counting | rtd | [] | [1] | [] |
| 186 | object_counting | agent | [] | [0] | [1687] |

## 建议

优先复核物体计数的数量定义、遮挡、相机切换和触发窗口；这些是待视频核验的方向，不是本次已证实的错误原因。如果沿用“全部response通过才保留QA”的规则，应针对当前数据中的19条失败QA处理，另外21条已不存在的QA不要重复操作。
