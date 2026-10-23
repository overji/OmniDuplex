# MiniCPM / Gander Agent 调用输出检查

快照：20261022T114826Z。只审计已完成 Agent QA，原始预测不修改。

| 模型 | Agent QA | 输出事件 | 非空可解析调用 | 已检查原始视频日志 | 原始 chunk 调用证据 |
|---|---:|---:|---:|---:|---:|
| minicpmo | 723 | 1287 | 0 | 723 | 0 |
| gander | 422 | 355 | 0 | 422 | 0 |

未观察到可解析的非空调用，因此当前没有可进一步验证完整 schema 或语义正确性的参数实例。

MiniCPM 使用 structured 提示，Gander 保留原生协议。时间窗口内的自然语言回复可以命中 timing，但不代表生成了工具调用。此处没有运行付费 LLM judge，也没有根据参考答案重建参数。

## 时间命中但没有调用的原始回复

### minicpmo

- Sample 39，87.0s；问题：Whenever a Como player celebrates a goal with joyous excitement, send a message to my soccer fan channel on Discord sharing who scored and the updated match score.
  输出： Moise Ken scores again for Como! The match is tied at 1-1.
- Sample 171，74.0s；问题：Whenever a fish is pulled out of the ice hole, look up winter handling guidelines for northern pike and save a reference note for my ice fishing trips.
  输出： A northern pike has been pulled from the ice hole. I will now look up winter handling guidelines for this fish and save a reference note.

结构化/调用尝试：sample 274，1.0s。
```text
 {"response": "The video creator is talking now.", "tool_calls": []}
```
### gander

- Sample 171，74.0s；问题：Whenever a fish is pulled out of the ice hole, look up winter handling guidelines for northern pike and save a reference note for my ice fishing trips.
  输出：a fish is pulled out of the ice hole appeared.
- Sample 273，45.0s；问题：When the host mentions where she bought her new Santa figurine, add that store and item to my holiday shopping list.
  输出：Noticed "The host mentions where she bought her new Santa figurine" starting.
