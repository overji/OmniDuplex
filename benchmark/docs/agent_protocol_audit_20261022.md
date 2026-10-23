# Agent 工具调用协议审查与重评（2026-10-22）

> 2026-10-23 更新：本文的七字段 Agent 实验已停止并保留快照（MiniCPM 723 条、Gander 624 条）。
> 后续按用户要求仅在 benchmark 内改为 `task_start(instruction)`，新实验及自动续跑目录见
> [benchmark README](../README.md#简化-agent-调用2026-10-23)。原始生成数据与训练协议未迁移。

原全量 benchmark 队列已按用户要求暂停，保留 MiniCPM 的 2,107 条完整结果及 Gander 暂停时的 1,204 条完整结果。没有停止其他用户或其他项目的训练进程。旧 Agent 结果只用于对照，不与本轮新协议结果合并。

## 协议审查

| 路径 | 实际协议 | 发现及处理 |
|---|---|---|
| 数据生成与合并 | `agent_call` 七字段参数；参考三段交互展开为 responses | 保留原 schema、available 工具列表和正样本；不将工具结果或最终总结输入评测模型 |
| MiniCPM 原始 checkpoint | 原生 listen/speak；Agent 回复用结构化 JSON 表达 | 原示例只有 instruction，现补全全部字段、禁止纯口头委托、说明跨媒体 chunk 继续 JSON；没有强制模型选择调用 |
| Gander | 原生 `<tool_call>{"name":...,"arguments":...}</tool_call>`，调用与口语分 unit | 补回原生动作/工具格式说明；直接保留 tool_calls 与错误状态，不再拼入口语；重评显式将工具预算从 96 提至 1,024 tokens，口语 Unit-8/11-token 预算不变 |
| MiniCPM 训练导出 | `<|agent|>` 原子 token 分隔回复和调用 | 保留训练权重/词表一致性校验；原始 checkpoint 不冒用这一协议 |
| LiveStar | 原生静默/回复动作，回复正文可使用 structured JSON | 共用修正后的结构化提示与解析器；不声称模型有经过训练的原生函数通道 |
| JoyAI | 原生 `</silence>` / `</response>`；可能返回 message.tool_calls | 保留文本解析，并补齐 native tool_calls 和 null content 情况的保存，避免静默丢失调用 |
| Qwen Realtime | 手动媒体步、文本回复；当前未在 session 注册原生工具 | structured 文本走统一解析；若最终 response.output 返回 function_call，直接保留，call-only 输出也不会丢失；未新增或实测原生工具注册 |
| Gemini Live（本仓库 SDK 适配器） | 音频输出与转写；当前没有注册原生函数 | 当前以 structured 文本提示为主，原生 tool_call 会显式报错；不把它当作支持原生函数的已验证实现。若要启用原生函数，仍需单独设计不执行工具时的会话继续策略 |
| 外部 Vertex 结果包 | 外部 vertex_live.py，非当前仓库适配器 | 只审计已有结果，未修改其代码或重跑远程 API；不能以本仓库修复声称该外部实现已修复 |
| Probe | YES/NO 回复机会判断 | 不生成工具参数，不在本轮 Agent 参数重评范围 |

MMDuet2 不运行实验。现有时间窗口、逐秒先输入媒体再生成、未提供工具返回等规则不变。

## 统一解析和事件保存

- 明确支持 OpenAI function 包装、原生 name/arguments 对象及完整 tool_call 标签；只归一化实际输出，不从自然语言承诺或参考答案补参数。
- Native tool-only unit 即使 text 为空、end_of_turn 为 false，也单独保存调用事件；错误和截断尝试保留供审计。
- 普通回复跨秒拼接，完整 JSON 在结束时解析；回复起始时间与工具字段出现/完成时间可以分别记录。
- 原生工具 schema 与提示词共用过滤逻辑，只传 agent_call 及实际 available 后端能力；不传额外函数。
- 新结果目录与协议指纹隔离，避免用旧格式结果跳过本轮重评。

## 验证与解释

修复后的三条真实 Agent 视频（原索引 2、171、273）均跑完。两模型在这组预检里仍没有产生非空调用。因此协议链路缺陷确实存在，但尚无证据表明修复提示就能解决模型的调用能力问题。

CPU 回归覆盖原生/包装格式、标签解析、非法与截断调用、跨秒 JSON、tool-only unit，以及 Qwen/JoyAI/cloud 调用字段保存。测试使用模拟模型或模拟网络，不调用收费 API。

## Agent-only 重评

使用原始 2,110 条合并数据，筛选有视频的正 Agent 样本，共 723 条，保留原始 sample_index。缺失视频的 Agent QA 不补回。GPU 为 0、1、2、3、6、7；依次跑 MiniCPM 和 Gander。

```bash
benchmark/.venv/bin/python benchmark/scripts/run_agent_recheck.py \
  --commands benchmark/probes/full_2110/run_commands.json \
  --output-dir benchmark/results/agent_protocol_v2_20261022 \
  --gpus 0 1 2 3 6 7 \
  --max-tool-tokens 1024
```

结果目录中的 `recheck_status.json` 报告实时进度和调用统计；各模型保留普通 timing 指标及 `agent_call_metrics.json`。分别报告语法可解析调用、schema 合法调用、仅使用 available 能力的调用，以及覆盖参考 required_tools 的调用。

这些检查不等于语义正确性、实际执行成功或完整任务成功。没有运行收费 LLM judge；即使仍为零调用，也按实际模型输出保留结果，不通过规则补出调用。

## 自动继续全量队列

按后续指示，使用 `benchmark/scripts/resume_after_agent.py` 等待上述两个 Agent 任务结束、原监督进程退出后，再启动剩余全量任务。同一时间不额外占用 GPU。Agent 任务若以 incomplete 结束仍继续其他模型，保留其不完整状态；若监督进程崩溃、被中断或协议文件变化，等待器明确报错，不擅自抢占卡。

续跑目录为 `benchmark/results/full_2110_dp6_after_agent_v2_20261022`。顺序为 Gander 剩余 615 条 RTD/PR、LiveStar 全量、JoyAI 全量、原定 8 个 probe 模型。已完成的 MiniCPM RTD/PR 和 Gander 769 条 RTD/PR 复用，旧 Agent 结果不混入新协议报告。`continuation_plan.json` 记录分来源复用规则，`continuation_status.json` 记录等待/运行状态，进入全量阶段后查看 `suite/suite_status.json`。这几个目录中的分片结果需要按原始 sample_index 与任务模式汇总，不应把新目录单独当作所有模型的全量报告。

```bash
benchmark/.venv/bin/python benchmark/scripts/resume_after_agent.py \
  --wait-for benchmark/results/agent_protocol_v2_20261022 \
  --commands benchmark/probes/full_2110/run_commands.json \
  --previous-root benchmark/results/full_2110_dp6_20261021 \
  --output-dir benchmark/results/full_2110_dp6_after_agent_v2_20261022 \
  --gpus 0 1 2 3 6 7
```

添加 `--dry-run` 只生成及校验续跑计划，不启动推理。恢复等待器也使用相同命令，已有输出继续按现有分片机制复用；不应同时启动两个等待器，目录锁会拒绝重复启动。
