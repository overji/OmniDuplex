# MiniCPM probe：无示例提示与输出问题的观察

快照：2026-10-25T05:44:22Z，完成 3630/9188 请求；推理持续运行。本次仅检查已有输出，未改 prompt、输入或解码设置，未调用付费 judge。

没有有无示例的受控对照，不能把当前失败因果归结为 zero-shot。观察上存在格式边界不清，但更突出的退化与前缀长度相关。

| 类型 | 已完成 | 含 think 标签 | 带 task_start 分隔符 | 达到上限截断 |
| --- | ---: | ---: | ---: | ---: |
| rtd | 2135 | 478 (22.39%) | 334 (15.64%) | 542 (25.39%) |
| pr | 881 | 172 (19.52%) | 151 (17.14%) | 189 (21.45%) |
| agent | 614 | 72 (11.73%) | 101 (16.45%) | 108 (17.59%) |

Agent 共 226 次非沉默输出、78 次合法格式调用。其中负侧 26 次合法格式调用、正侧 52 次；格式合法不代表触发正确或 instruction 正确。
RTD/PR 合计 3016 次输出中，485 次误带分隔符（16.08%）。系统和 user prompt 对所有任务重复 Agent 格式说明，缺少明确的分任务示例，可能加重这种混淆，但尚未做因果验证。

| 前缀长度 | 请求 | 截断 | 含 think 标签 |
| --- | ---: | ---: | ---: |
| 0–60 | 1480 | 10 (0.68%) | 0 |
| 60–120 | 1092 | 26 (2.38%) | 1 |
| 120–inf | 1058 | 803 (75.90%) | 720 |

超过 120 秒的 1058 次请求中，803 次触及生成上限、720 次含 think 标签；120 秒以内 2572 次仅 36 次截断、1 次含 think 标签。这个关联很强，但不能据此断定存在模型的 120 秒硬限制。应排查长输入的组织、模板与实际解码。
已确认适配器传 enable_thinking=False；原生 chat 未显式覆盖 num_beams，按本地模型代码使用 3-beam、repetition_penalty=1.2。未调整本轮运行。

## 最新例子

### sample 868 / response 2

Let me know whenever the drummer appears on screen.

- pre，前缀 42.0s，结束状态 True：
```text
<silent>
```
- post，前缀 47.0s，结束状态 True：
```text
The drummer appears on screen. <|task_start|>Continue monitoring for further appearances of the drummer.
```

### sample 835 / response 0

Let me know whenever the print head carriage is positioned above the mascot's head on the paper sheet.

- pre，前缀 78.0s，结束状态 True：
```text
The print head carriage is positioned above the mascot's head on the paper sheet.
```
- post，前缀 84.0s，结束状态 True：
```text
The print head carriage is positioned above the mascot's head on the paper sheet.
```

### sample 840 / response 0

Whenever the streamer expresses frustration over making a tactical mistake with his equipment or scorestreaks, save a note summarizing his reaction and what happened.

- pre，前缀 68.0s，结束状态 True：
```text
<silent>
```
- post，前缀 72.0s，结束状态 True：
```text
Note: At 00:40, the streamer expresses frustration for mistakenly using his Shredder scorestreak. <|task_start|>Create a note titled "Shredder Mistake at 00:40" summarizing the streamer's regret over the unnecessary use of the Shredder scorestreak.
```

### sample 876 / response 0

When Raphael and his instructor jump out of the airplane into freefall, let me know and check the current weather conditions in Portimão.

- pre，前缀 156.0s，结束状态 False：
```text

<think>

<think>
<think>
<think>

<think>
<think>
<think>
<think>
<think>
<think>
<think>
<think>
<think>
<think>
<think>

<think>
<think>

<think>
<think>
<think>
<think>
<think>
<think>
<think>
<think>

<think>
<think>
<think>
<think>
<think>
<think>
<think>




<think>
<think>
<think>
<think>
<think>
<think>



<think>
<think>


<think>
<think>

<think>

<think>






<think>
<think>




<think>



<think>
<think>
<think>















































<think>


<think>


<think>


<think>

<think>





















<think>
<think>


<think>

<think>















<think>








<think>





```

鼓手样本：正侧正文正确表达鼓手出现，却额外委托 Continue monitoring；这是 PR/Agent 边界混淆。
打印头样本：正常生成简短 PR 回复，没有分隔符，说明 zero-shot 并非完全无法遵守。
游戏主播样本：格式完整，但参考应记录 01:09 的 Shock RC 误用，输出写成 00:40 的 Shredder；按参考文本看有事件/装备混淆。尚未逐帧复核或调用内容 judge。
跳伞样本：156 秒负侧大量重复 think 标签并触及 256-token 上限，属于解码退化；不应直接归结为没有展示 task_start 示例。

建议先定位长前缀退化，再在固定样本、相同输入与解码配置下做有无示例对照。若添加示例，应仅展示该任务的前后两侧行为，另设输出目录与协议身份；本次没有启动新实验。
