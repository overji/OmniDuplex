# JoyAI / Gander 原生 Agent 协议适配

新建 realtime 运行时，JoyAI 与 Gander 默认使用 `--agent-output-protocol native`，
配合 `--agent-call-schema task-start`。仅 Agent 样本改变协议；RTD、PR、Probe、
采样配置和训练行为沿用原有实现。此变更不自动启动任何实验。

## 模型实际输出

JoyAI 使用原生交互标记：

```text
</silence>
</response> 看到了，我会记录这次事件。 </delegation> 记录刚才观察到的事件及其相关信息。
```

原生 runtime 的 `<delegation>` 拼写也接受。读取 HTTP 返回的
`streamingharness.raw_content`，避免只取归一化后的用户回复而丢掉委托内容。
没有 delegation 标记的承诺不构成调用；空 instruction、重复标记以及明确的
length 截断均保留为失败尝试，不补全参数。服务未提供可靠完成状态时记录为未知。

Gander 保留原生 speaking / tool-call 动作通道，分别生成面向用户的回复和调用。
仅当 available 后端能力非空时向官方 OnlineRunner 传入以下工具 schema：

```json
{"type":"function","function":{"name":"task_start","parameters":{"type":"object","properties":{"instruction":{"type":"string","minLength":1}},"required":["instruction"],"additionalProperties":false}}}
```

这是 benchmark 的 instruction schema 适配，**不是严格照搬官方 name-only schema**。
不要求 Gander 在 spoken text 中输出 JSON 或 `<|task_start|>` 文本分隔符。
它仍需在原生工具动作中生成合法 JSON 参数；适配不能保证模型本身一定会使用它。

## 配对、时间和计分

- Gander 的一个完整 speech turn 与紧邻的一个 native call 可按任意顺序配成
  一次逻辑 Agent 事件。多单元的同一 speech turn 持续累积。每个组件只消费一次。
- listen、interrupt 和无法识别的空动作阻断配对；不跨间隔猜测关系，不合并重复调用。
- 逻辑 onset 取实际两个组件最早发生的时间，并分别保存
  `response_onset_seconds`、`agent_call_onset_seconds`、`agent_call_complete_seconds`。
  时间仍为输入完当前媒体单元后的区间终点，不回填参考窗口时间。
- 未配对的 speech / call 各自保留；缺失的回复或 instruction 不由标注或 prompt 补齐。
- 原始输出照常写入 worker 结果目录的 `sample_*.chunks.jsonl`（Gander）或
  `sample_*.json`（JoyAI）；事件附带原始调用、解析状态和完成状态。
  只记录调用，不执行工具、不输入工具结果。
- 归一化为已有的 `response_text` 和 `tool_calls: task_start(instruction)`，继续使用
  原有时间匹配、LLM Judge 与 scored-F1。新增协议不修改参考答案或评分窗口。

仅 available 后端能力及其说明进入模型提示词。前端遗留的格式指令、参考 response、
目标窗口均不进入新增原生提示词。完整模板为
[`agent_native_joyai.txt`](../configs/prompts/agent_native_joyai.txt) 和
[`agent_native_gander.txt`](../configs/prompts/agent_native_gander.txt)。

## 运行与复现

沿用现有 realtime / `parallel joyai` / `parallel gander` 的环境、模型路径、服务配置和
GPU 参数。在原命令中显式选择：

```text
--agent-call-schema task-start --agent-output-protocol native --output-dir <全新结果目录>
```

需要复现旧文本协议时显式传入 `--agent-output-protocol task-text` 或 `text`。
已有 suite/队列显式指定的协议继续生效，不会被新默认值覆盖。

`run_config.json` 记录 `native_agent_protocol_version`、协议实现/模板/schema 的摘要；
逐样本保存实际 system prompt 摘要。旧结果目录不能混入新 native 产物。
代码升级也可能改变旧 task-text 实现摘要；出现协议不匹配时使用新的结果目录，
或用旧运行时继续旧结果，不绕过恢复校验。

适配层的 CPU 测试：

```bash
python -m pytest -q benchmark/tests/test_native_agent.py \
  benchmark/tests/test_agent_protocols.py benchmark/tests/test_task_text.py
```

CPU mock 测试不代表真实模型能正确生成调用；正式结果应报告所选协议。
