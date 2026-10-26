# 三模型 Agent 零调用排查（2026-10-24）

本次只读检查原始日志、真实提示、tokenizer、官方运行时代码及 CPU 解析测试；没有修改运行协议，也没有启动 GPU 消融。

## Gander：工具动作从未进入，不是 JSON 校验失败

检查范围：`benchmark/results/agent_task_start_v1_20261023/gander` 的全部 723 条 Agent QA。

- 全部 723 条 QA 均有非空的可用函数 schema；实际运行使用 Gander Thinker checkpoint 和 checkpoint tokenizer，并将 tools 传入官方 OnlineRunner.prepare。
- 共检查 135,866 条逐秒原始日志：134,665 条首记录 token 为 listen，1,112 条为 speak，89 条为 tts_bos。这里统计的是原始 generated_token_ids 的首元素，不把 tts_bos 自动解释为一个独立动作。
- 全部 generated_token_ids 中 `<tool_call>`（ID 151657）出现 0 次；原生 is_tool_call 事件 0 次、tool_error 0 次。因此不是生成了完整工具调用后被 benchmark 解析器丢弃。
- 实际 checkpoint tokenizer 的 bad_token_ids 为空；tool_call、speak、listen 等控制 token 均未被该列表禁止。官方首动作规则在 tool_schemas 非空时允许 tool_call。
- 工具 token 预算已设为 1024，但只有进入工具动作后才使用；本轮没有进入该分支，增大该预算没有解决动作选择问题。

例如 sample 2：用户要求发现乐器盖被移除后检查替换盖是否可买；模型在 67 秒只输出移除事件提醒，没有搜索或委托调用。sample 8 用户要求起身离开时保存笔记；模型在 14 秒仅输出事件提醒。

## 已确认的协议差异，因果尚待对照

官方 `mcpmft/tool_protocol.py` 的 LEAN_TASK_START_SCHEMA 使用 `arguments.name`：简短任务名称，完整任务内容由 Runtime 绑定当前用户 turn。本 benchmark 按统一约定使用 `arguments.instruction`，要求简短、自包含任务指令；原生提示还显式说明该差异。函数名相同不代表与官方协议完全一致。

真实 system prompt 又包含较长的能力列表，以及基础层的 “announce the grounded delegation” 和原生层的工具动作要求。模型可能将请求理解为事件提醒或口头宣布委托，而没有进入工具分支。这里是待验证解释，不能仅凭现有结果证明由某一处提示造成，更不能断言原生 schema 一换回去就一定有效。

没有在本轮证明 Gander 在其官方任务协议下也不能调用工具；当前结果仅证明本 benchmark 适配条件下没有生成工具动作。

## 另外两个模型不是同一种失败

- MiniCPM：723 条 Agent 的 18 个事件被解析器识别为调用尝试，但没有可解析调用；抽查见空 tool_calls、多余括号、错误字段等。先前 CPU 排查还确认原生返回文本可能丢掉普通首 token；本轮没有保存完整真实 token 增量，因此不能把文本层零调用完全归因于模型本身。
- JoyAI：688 条已完成 Agent 中有 2 个失败尝试。sample 215 原始服务返回的 JSON 自身未闭合，而且调用 note_write 而不是 task_start；sample 837 为混合自然语言及破损字段，arguments 也不是要求的对象。没有从这些文本补造调用。

## 验证及下一步

现有 AgentProtocolTests 共 5 项 CPU 测试通过，覆盖跨秒 JSON、原生工具元数据、无文本工具事件和保留截断失败。

后续应在独立输出的小样本上比较：官方 task_start(name) 与 benchmark task_start(instruction)、短且一致的提示与当前提示；记录每秒 listen/speak/tool_call 的决策概率和原始 token。保持媒体、随机种子与问题一致。先确定是动作选择还是协议适配问题，再考虑重评全量；不得通过强制工具动作或改写自然语言承诺提高正式调用分数。
