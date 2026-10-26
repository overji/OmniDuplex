# MiniCPM / Gander / JoyAI Agent 纯文本格式重评测对比

统计时间：2026-10-24 12:33:35 UTC。这是固定快照，JoyAI 当时尚未跑完。

结论：Gander 的开口时间 precision/F1 最好，但几乎不生成 instruction；JoyAI 格式遵循最好，但大量提前开口；MiniCPM 在召回与调用格式之间居中。

## 条件与口径

三者采用 `回复<|task_start|>instruction`。本轮仅比较 Agent，全部为正样本；不是 RTD/PR 全任务总分。
MiniCPM 与 Gander 使用音视频，JoyAI 使用视觉，因此这不是控制输入模态后的模型对比。
时间评分复用仓库官方 scorer：回复开始时间落入闭区间参考窗口，一对一匹配。Recall=命中窗口/参考窗口，precision=命中/输出事件。
格式合规指至少一次非空回复、单个分隔符、非空 instruction、内部 task_start schema 合法，且没有明确截断；不代表内容或 instruction 语义正确。
JoyAI 的结束状态为 unknown，格式统计不要求有已核实的 EOS；它的格式结果不能被解读为“已确认自然结束的合法调用”。

## 相同样本上的公平比较

三者共同完成 **557 条 QA / 586 个参考窗口**。只按此交集排名，完整索引保存在同名 JSON。

| 模型 | 输出事件 | 时间命中 | Recall | Precision | F1 | 合规双字段 QA | 命中时间且双字段合规的回复 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniCPM-o-4.5 | 740 | 67/586 | 11.43% | 9.05% | 10.11% | 54/557 (9.69%) | 11 |
| Gander | 136 | 61/586 | 10.41% | 44.85% | 16.90% | 0/557 (0.00%) | 0 |
| JoyAI | 709 | 39/586 | 6.66% | 5.50% | 6.02% | 537/557 (96.41%) | 37 |

最后一列要求命中的同一个回复事件同时具备双字段格式；它不要求分隔符生成时刻也在窗口内。
若另外用调用分隔符首次出现时刻匹配窗口，三者分别命中 9 / 0 / 37 个窗口。以上均未评判语义。

## 行为诊断

| 模型 | 有回复 QA | 首次回复早于首个窗口 | 提问后首个完整秒即开口 | 首次回复相对窗口开始的中位差 |
| --- | ---: | ---: | ---: | ---: |
| MiniCPM-o-4.5 | 494/557 | 181/557 | 113/557 | +6.5s |
| Gander | 129/557 | 21/557 | 0/557 | +2s |
| JoyAI | 557/557 | 540/557 | 365/557 | -24s |

时间中位差只对有回复 QA 计算，负数表示提前。JoyAI 540/557 条首次回复过早，中位提前 24 秒；格式遵循高不等于触发正确。
Gander 在交集中 557 条只有 129 条开过口，输出较保守；136 个输出中 127 个状态 absent、9 个明确截断，没有分隔符。
MiniCPM 在交集中 740 个输出中，612 个状态 absent、61 个明确截断、9 个字段为空、58 个合规；主要损失仍是模型没有输出双字段。

## 全量/当前进度（覆盖不同，不用于三者直接排名）

| 模型 | 完成 QA | 参考窗口 | 输出 | 命中 | Recall | Precision | F1 | 双字段合规 QA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniCPM-o-4.5 | 723/723 | 757 | 979 | 98 | 12.95% | 10.01% | 11.29% | 70/723 |
| Gander | 723/723 | 757 | 173 | 79 | 10.44% | 45.66% | 16.99% | 1/723 |
| JoyAI | 557/723 | 586 | 709 | 39 | 6.66% | 5.50% | 6.02% | 537/557 |

Gander 全部 723 条中有 1 条双字段合规，但该条不在此快照的三者共同完成交集中，且未命中时间窗口。

## 复核

依照 benchmark-score-audit 流程，用冻结快照中的原始预测重新运行 timing scorer；MiniCPM、Gander 完整分数与保存的 metrics 一致。
对每个已有事件的 text 和结束标记重新执行 task-text parser，三者均为 0 个 tool_calls 不一致。没有进行猜测恢复，没有修改原始输出。
未调用内容 LLM judge，不能据此排名回复内容或 instruction 的语义正确率。

原始运行目录：
- `/root/jimodels/OmniDuplex/benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/minicpmo_task_text_agent_full`
- `/root/jimodels/OmniDuplex/benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/gander_task_text_agent_full`
- `/root/jimodels/OmniDuplex/benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/joyai_task_text_agent_full`
