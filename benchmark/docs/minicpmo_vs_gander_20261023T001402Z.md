# MiniCPM-o-4.5 与 Gander：当前同样本对比

快照时间：2026-10-23T00:14:02.026853+00:00。共同完成 1318 条 QA。

RTD/PR 使用原全量实验已完成结果；Agent 仅使用 protocol_v2 重评结果，旧 Agent 全部排除。只在相同原始 sample_index 上比较；Gander 未完成的样本不计入双方分母。

| 类型 | 模型 | QA | 窗口 | 输出 | 命中 | 时间召回率 | 时间精度 | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | minicpmo | 1318 | 3010 | 2767 | 728 | 24.19% | 26.31% | 25.20% |
| overall | gander | 1318 | 3010 | 1142 | 435 | 14.45% | 38.09% | 20.95% |
| mode:rtd | minicpmo | 380 | 1702 | 968 | 375 | 22.03% | 38.74% | 28.09% |
| mode:rtd | gander | 380 | 1702 | 441 | 174 | 10.22% | 39.46% | 16.24% |
| mode:pr | minicpmo | 389 | 731 | 837 | 244 | 33.38% | 29.15% | 31.12% |
| mode:pr | gander | 389 | 731 | 345 | 140 | 19.15% | 40.58% | 26.02% |
| mode:agent | minicpmo | 549 | 577 | 962 | 109 | 18.89% | 11.33% | 14.17% |
| mode:agent | gander | 549 | 577 | 356 | 121 | 20.97% | 33.99% | 25.94% |

## 新协议 Agent 调用（同样本）

| 模型 | Agent QA | 输出事件 | 有输出 QA | 调用尝试事件 | 有可解析调用 QA | 有 schema 合法调用 QA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| minicpmo | 549 | 962 | 493 | 19 | 0 | 0 |
| gander | 549 | 356 | 304 | 0 | 0 | 0 |

## 能力分组（同样本，含新 Agent）

| 能力 | QA | MiniCPM R | Gander R | MiniCPM P | Gander P | MiniCPM F1 | Gander F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attribute_perception | 111 | 26.20% | 18.78% | 24.19% | 51.19% | 25.16% | 27.48% |
| grounding | 119 | 22.42% | 12.81% | 27.04% | 30.51% | 24.51% | 18.05% |
| action_perception | 172 | 20.20% | 17.00% | 20.20% | 48.59% | 20.20% | 25.18% |
| object_perception | 78 | 29.91% | 25.23% | 33.68% | 46.55% | 31.68% | 32.73% |
| event_understanding | 165 | 21.21% | 7.99% | 23.91% | 25.44% | 22.48% | 12.16% |
| text_rich_understanding | 185 | 33.87% | 18.66% | 36.48% | 58.70% | 35.13% | 28.32% |
| object_counting | 61 | 16.41% | 9.74% | 24.81% | 50.00% | 19.75% | 16.31% |
| spatial_understanding | 94 | 27.67% | 14.62% | 33.49% | 44.58% | 30.30% | 22.02% |
| prospective_reasoning | 70 | 14.06% | 11.72% | 13.74% | 26.79% | 13.90% | 16.30% |
| causal_reasoning | 109 | 16.85% | 11.80% | 16.67% | 26.58% | 16.76% | 16.34% |
| emotion_recognition | 72 | 20.55% | 7.53% | 21.43% | 10.19% | 20.98% | 8.66% |
| action_counting | 82 | 30.05% | 10.93% | 31.25% | 30.30% | 30.64% | 16.06% |

## 口径与限制

- R = 命中数 / 参考窗口数；P = 命中数 / 输出事件数；F1 为两者调和平均。窗口包含端点，每条输出与窗口进行一对一 onset 匹配。
- Agent 只评工具委托触发窗口，不输入工具回复，不评工具返回后的总结。时间命中允许普通口头回复，因此不代表有工具调用。
- 工具调用仅统计模型明确输出，未执行工具、未运行语义 LLM judge；没有内容正确性分数。
- 两者采用各自适配协议：MiniCPM structured JSON、Gander 原生工具动作；token 预算不同，不构成相同解码条件消融。
- 总体为当前评测配置的阶段性结果，RTD/PR 与 Agent 来自上述不同轮次；不能与旧版全 Agent 总体直接解释为模型能力变化。

## 核验

- minicpmo/original: stored groups exactly match recomputation (2107 QA)
- minicpmo/agent_v2: stored groups exactly match recomputation (723 QA)
- gander/original: stored groups exactly match recomputation (1204 QA)
- gander/agent_v2: stored groups exactly match recomputation (549 QA)

原始预测未修改；快照、交集索引、来源目录和复算指标见：`benchmark/results/comparisons/minicpmo_vs_gander_20261023T001402Z`。
