# MiniCPM task_start 零调用诊断（2026-10-23）

本次固定快照：262 条已完成 QA，418 个回复事件。运行目录为 `benchmark/results/agent_task_start_v1_20261023/minicpmo`。仅审查现有结果和做 CPU 模拟，不停止或修改正在运行的实验。

## 结论与确定性

1. **已确认输出未稳定遵循协议。** 53 个事件以 JSON 对象开头，其中 28 个可以整体解析；4 个合法 JSON 明确给了空 tool_calls。已保存的逐秒原始文本中，task_start 出现于 0 条 QA；tool_calls 出现于 12 条 QA。没有从自然语言承诺补造调用。

2. **已确认原生文本返回存在丢首 token 的条件性缺陷。** 当前官方 streaming_generate 只在 j != 0 时把生成 token 加入返回文本。首 token 是控制符时符合预期；首 token 为普通内容时会丢失。CPU 模拟调用真实官方函数：同样的 JSON token 流，以 speak 开头时完整返回，直接以左花括号开头时返回文本缺少左花括号。未加载模型权重，不能以此推断真实发生频率。
```json
[
  {
    "first_is_speak": true,
    "raw_ids_text": "{\"name\":\"task_start\"}",
    "returned_text": "{\"name\":\"task_start\"}",
    "equal": true
  },
  {
    "first_is_speak": false,
    "raw_ids_text": "{\"name\":\"task_start\"}",
    "returned_text": "\"name\":\"task_start\"}",
    "equal": false
  }
]
```

3. **当前原始日志不足以排除 token 返回层损失。** 本轮 structured 模式只保存 native 返回的 text，没有保存 total_ids 的逐步增量。因此可以确认文本解析器未发现完整调用，但不能证明模型 token 流从未生成过调用。必须新增独立诊断记录，不能直接把全部失败判为模型能力问题。

## 代表样本

- sample 84：完整输出 `{"response":"First knockdown occurred, recording note now.","tool_calls":[]}`。这是主动省略调用，不是 JSON 被截断。
- sample 291：从第 108 秒跨到 112 秒持续输出，途中切换为 `tool_call: web_search` 的另一种结构，最终不构成完整 JSON；说明多秒收集确实存在，但模型格式不稳定。
- sample 684：直接尝试 `product_search`，而非 task_start；包含 function 后多余空格、字段名前空格和未闭合结构；有回合在预算未用满时结束。
- sample 318：视频最后 60.005 秒才开始 JSON，视频结束时未完成。EOF 是这类个别样本的额外限制，不能解释全部零调用。

## 已排除与尚未验证

- 真实 system prompt 已包含 task_start 唯一参数 schema、available 列表和结构化示例；不是没传工具定义。一个真实提示约 2,200 tokens，能力描述有重复。长提示与“只宣布委托”的旧自然语言描述可能干扰格式遵循，但尚未做消融，不能断言因果。
- tokenizer bad_token_ids 虽有 6,965 项，但 task_start、tool_calls 和测试最小调用 JSON 的 token 均未命中禁用列表。最小内层调用仅 15 tokens；不能把每秒 20 token 上限视为唯一原因。外层 JSON 与较长任务指令仍需跨秒。
- 当前为原始 MiniCPM-o-4.5 的 as_duplex 路径，无训练后的 Agent adapter；schema 只是文本提示，没有独立函数解码/语法约束。模型在该路径上的格式遵循需要实际验证。
- 原生默认 sampling、temperature 0.7、top_k 100、top_p 0.8、文本重复惩罚 1.05；它们可能影响 JSON 稳定性，尚未独立验证。

## 下一步优先级

先独立保存每步 total_ids 增量，同时记录 native text，核对首 token 和特殊 token；只修复文本返回中的内容丢失，不改变模型决策和 KV。随后在小样本上分别对比短而一致的提示、greedy 与当前 sampling、当前预算与较大预算。每项使用新目录和协议身份，不覆盖当前结果；未经验证不直接重跑全量，也不将错误函数名改写成 task_start 来抬高分数。

固定快照、CPU 复现代码和版本哈希：`benchmark/results/diagnostics/minicpmo_agent_20261023T032456Z`。
