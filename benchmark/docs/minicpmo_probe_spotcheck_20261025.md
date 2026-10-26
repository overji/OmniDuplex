# MiniCPM probe 早期运行与输出抽查

固定抽查快照：132/9188 次独立请求（1.44%），共 64 个完整前后缀配对。当前仅 MiniCPM 在八卡运行，其他模型尚未开始。该样本很早且类别不平衡，不是最终成绩。

## 判断结果

| 类型 | 完成请求 | 正侧应回复准确率 | 负侧应沉默准确率 | 整体准确率 | 配对全对 |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 132 | 67.19% | 38.24% | 52.27% | 14.06% (64 对) |
| mode:rtd | 109 | 71.70% | 33.93% | 52.29% | 15.09% (53 对) |
| mode:pr | 13 | 50.00% | 42.86% | 46.15% | 0.00% (6 对) |
| mode:agent | 10 | 40.00% | 80.00% | 60.00% | 20.00% (5 对) |

当前判断只区分精确 `<silent>` 与非空输出。35/132 次输出因达到 256 token 而截断；39 次含 `<think>` 标签，非 Agent 任务中 15 次带 `<|task_start|>`。这些是输出质量问题，不是推理进程报错。
非空乱码、推理标签和重复数字也可能被当前判断口径算作开口，因此决策命中不能代表内容正确。当前非 Agent 的格式有效性仅检查非空、无调用分隔符、无明确截断；没有完整过滤推理标签，这个统计也存在局限。

## Agent 输出

Agent 仅 10 次请求/5 对：负侧 4/5 正确沉默；正侧 2/5 被判为开口，但只有 1 次有完整回复和 instruction，另一次是截断乱码。
所有 Agent 开口输出共 3 次，合法调用格式 1 次；其余 2 次均为 256-token 截断。80% 的 Agent 格式有效率包括合法 `<silent>`，不等于 80% 生成了正确调用。

### 可用格式示例：sample 5 / response 0

问题：人群对街头片场事件大声欢呼或发笑时，记录时间和原因以便检索。
前缀到 57 秒：`<silent>`，按参考正确；前缀到 62 秒：
```text
At 00:57, the crowd cheers loudly as the man in the gray shirt falls over the barricade and lands on the sidewalk. This was triggered by his clumsy attempt at a Spider-Man-like leap. <|task_start|>Record crowd cheer at 00:57 due to man falling over barricade during "Spider-Man" stunt attempt.
```
回复、分隔符和 note instruction 都存在，目标方向与记笔记请求一致。但参考欢呼触发为 00:58，模型写成 00:57；这是格式合规与细节正确性的区别，尚未跑内容 judge 或逐帧人工核验。

### 漏回复示例：sample 2 / response 0

问题：乐器浅色端盖被取下时通知，并查询替换盖。64 秒前缀与 80 秒前缀都输出 `<silent>`。
前者按参考正确，后者漏触发；参考在 65 秒已看到端盖分离，窗口 65–80 秒。这一对失败在应不应该开口，而不是 JSON/分隔符解析。

## RTD 输出示例：sample 0 / response 1

问题要求持续报告簧片状态和位置的变化。66 秒前缀输出 `<silent>`，80 秒前缀描述老师讲解簧片组件，随后输出：
```text
 <|task_start|>Explain the function of the bridle on the duduk reed and how it affects sound production.
```
这不是 Agent，不应带调用。参考重点是“端盖已取下，裸露的双簧片和深色箍被露出”；生成偏向泛化介绍，还擅自增加解释任务。二分类配对能对，内容和输出类型仍有问题。

## PR 输出示例：sample 1 / response 0

问题：簧片放进透明水杯时提醒。141 秒前缀应沉默，却输出：
```text
02:17 done already handled: 1, output: <|task_start|>Place the duduk reed into the clear glass of water.
```
145 秒前缀应回复，输出：
```text
duduk reed is placed into the clear glass of water.
```
正侧简短且与参考事件吻合；负侧提前开口，并把提醒任务变成执行“放入水杯”的 instruction。不能只看正侧输出，要看整对是否全对。

## 异常与后续判断

在较长前缀中观察到重复 `<think>`、复述提示词和长串 `00:50.000...`。sample 5 / response 3 的两个侧面都输出退化文本并触及 256-token 上限。
本地适配器确实设置 `enable_thinking=False`，模型远程代码也接收此参数，因此不能简单归因为忘记关闭 thinking。具体原因还需用受控短/长前缀复现，检查输入组织、原生模板与解码；本次没有改正在运行的配置。
建议将“是否开口”“输出格式”和“内容正确性”分开看。当前分隔符协议已出现成功例，但 Agent 样本只有 5 对，不能据此判断最终能力。

原始结果：`benchmark/results/task_text_agent100_20261024_v2/task_text_reply_queue_v2/suite/probe/minicpmo_reply_task_text/`。同名 JSON 保存本次统计、样本 ID、前缀时间与完整原始输出。

补充配置核验：MiniCPM 适配器传入 `do_sample=False`，但未显式设置 `num_beams`；本地模型 `prepare_generation_config` 默认 `num_beams=3`，并在此分支使用 `repetition_penalty=1.2`。因此本轮按原生默认是确定性的 3-beam 搜索，不能称作单 beam greedy。这也是需要在解释退化输出前确认的配置差异；本次未修改运行参数。
