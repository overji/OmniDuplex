# Benchmark Comparison：AOI-Bench 与相关流式交互评测

核对日期：2026-10-25。外部条目以论文当前可访问版本为准；AOI-Bench 以当前发布数据和仓库实现为准。这里比较任务定义与评测协议，不比较不同协议下的数值成绩。

## 1. 定位

AOI-Bench 最值得强调的定位是：在相同的多模态视频能力体系下，同时评价持续更新、条件提醒、事件触发的后台任务委托，区分是否应当开口、说什么、应当委托什么任务。

双模式 Probe/Online、音视频输入、主动响应、多次触发和内容与时间联合评价，在相关工作中已有先例，不能单独作为首创贡献。相对下列六项工作的主要差异，是事件触发的 Agent 委托：用户侧回复和面向后台的任务 instruction 必须同时生成，并受到可用能力与用户授权范围约束。它目前是委托指令生成评测，不是工具执行成功率或执行后闭环评测。

## 2. Benchmark Comparison

V=视频，A=音频，T=文本问题。“未设专门任务”指论文任务体系没有把该能力列为独立评测目标，不表示模型本身不支持该能力。各行的 QA、sample、response slot 统计单位不同，规模不可直接当作覆盖度排名。

| Benchmark | 规模与任务体系 | 输入 / 问题形式 | 何时回复的评测方式 | 主要输出与评分 | Agent 委托专项 |
| --- | --- | --- | --- | --- | --- |
| [StreamingBench](https://arxiv.org/html/2411.03628) | 900视频、4500 QA；18任务；实时视觉、全源、上下文理解 | V+A / T | 主体以前缀离线QA模拟在线；PO子任务逐秒询问是否应输出 | 主体选择题准确率；PO首次响应时间误差 | 未设专门任务 |
| [OVO-Bench](https://arxiv.org/html/2501.05510) | 644视频、2814样本/标注；12任务；回溯、实时感知、前向主动响应 | V / T | 支持在线模型及离线前缀模拟；前向任务使用密集时间点查询，原文另讨论在线时效指标 | 回溯/感知为选择题；前向任务准确率及时间相关指标 | 未设专门任务 |
| [OmniMMI](https://arxiv.org/html/2503.22952) | 1121视频、2290问题；6子任务 | V+A / T或合成语音问题 | 按任务采用时间点问答、PA主动告警、PT噪声抑制；PA统计首次主动响应是否在指定时段 | 任务相关准确率、LLM内容判分、首次告警时间与保持沉默 | 未设专门任务；Action Planning是下一步动作预测，不能等同工具调用 |
| [OmniPro](https://arxiv.org/html/2605.18577) | 1262视频、2700人工复核样本；9子任务、3认知层级、6基础能力 | V+A / T；区分speech与非语音sound | Probe：触发前后两次累积前缀查询；Online：持续流式输入、自主多次响应 | Probe精确匹配且前后都对；Online ±3秒匹配且内容正确的F1，开放任务用LLM judge | 未设专门任务 |
| [OmniInteract](https://arxiv.org/html/2605.26485) | 250视频、1430 response slots；1062个1Q1A slots、368个1QnA slots | V+A / 问题直接位于原音轨 | 原生在线推理；实时、主动、嵌套、连续监控及打断 | IA-QTF1结合质量、时效和无效输出；另测打断及嵌套恢复 | 未设专门任务 |
| [Omni-DuplexEval](https://arxiv.org/html/2605.17360) | 660视频/样本；9子任务；RTD与PR两类交互；视频均不足1分钟，平均34秒 | V+A / 用户指令 | 流式连续描述和主动提醒，包含目标事件重复出现 | RTD内容一致性+时间敏感度；PR时间窗口与LLM内容判断 | 未设专门任务 |
| **AOI-Bench（本项目）** | **852视频、2107 QA；12能力×RTD/PR/Agent，共36个已覆盖组合** | **V+A / 按question_time插入T；视觉模型仅V** | **Realtime按时间顺序接收输入后生成；Probe对每个参考回复构造前后前缀** | **独立时间R/P/F1；开放回复内容judge；Agent另评instruction；Probe决策与TP内容分开评** | **回复内容 `<\|task_start\|>` 后台instruction；提供可用能力；记录委托、不执行工具** |

说明：StreamingBench论文称总规模4500 QA，但PO子集的构建数与当时实际评测子集不同；正式报告应锁定下载版本。OVO-Bench的2814通常作为样本/元标注规模引用，不能等同所有密集查询次数。AOI-Bench的2107是当前发布集，不是三模型共同完成的2008条QA分析子集。

## 3. 最接近工作的区别

| 对照对象 | 已有覆盖，不能当作我们的独有贡献 | 本项目可明确强调的差异 |
| --- | --- | --- |
| OmniPro | 音视频主动理解、三认知层级、多次触发、Probe/Online、模态诊断 | 12语义能力与3交互类型交叉；开放回复与后台委托instruction同时生成；委托内容单独评价 |
| Omni-DuplexEval | RTD、PR、开放生成、持续更新、时间与内容评价 | 增加Agent委托交互；统一12能力体系；当前数据覆盖更长的视频区间；同时提供普通Omni模型的Probe入口 |
| OmniInteract | 原生流式、多响应监控、响应窗口、质量及时效评价 | 侧重同一任务能力在更新/提醒/委托中的表现，以及触发后后台任务指令的生成；我们目前没有它的语音提问、打断及嵌套交互专项 |
| OmniMMI | 流式交互、主动告警、多轮依赖、噪声下保持沉默 | 对每次参考回复建立窗口并统计额外输出；在能力×交互矩阵中单列Agent委托 |
| OVO-Bench / StreamingBench | 前缀限制、动态时刻问答、主动响应相关任务 | 持续自主回复与实际回复内容评测；一条指令可对应多个回复机会；额外评价事件触发的任务委托 |

这是对本次比较范围的分析，不构成对所有已发表工作的“首次”声明。

## 4. 当前实现边界

- **Realtime输入**：用户问题按其时间插入，模型先接收该秒音视频再生成；例如接收 `[43,44)` 后，该轮输出时间记作44秒。视觉模型不输入音频。输入推进和原生listen/speak控制的具体方式依适配器而异。这里的时间分数主要按视频时间轴计算，不能直接解释成墙钟实时吞吐、语音首包延迟或打断能力。
- **时间评分**：回复起始时间在闭区间 `[window_start, window_end]` 内作一对一匹配。R=命中/参考窗口，P=命中/输出事件；单独时间F1不验证语义。不能与OmniPro内容正确性门控的Online F1或OmniInteract IA-QTF1直接横比。
- **内容judge**：代码已实现。RTD/PR评回复正确性；Agent分别评回复与委托instruction。分项分值为0、0.5、1、1.5、2。Probe仅对TP做内容judge，Agent任务分取两项平均。此前三模型对比表尚未加入这些内容分数，不能称作已完成内容验证。
- **Probe前缀**：当前代码为负侧截至 `window_start - 1秒`、正侧截至 `window_end`，不是统一触发点±3秒；实际裁切及无效对排除由manifest决定。每侧独立推理；只告知先前已处理机会数量，不提供真实先前回复文本。二元决策准确率、成对准确率与TP内容质量分别报告，不能把开口判为YES就当作内容正确。
- **Agent范围**：当前发布集723条Agent均为正样本。Probe的事件发生前负前缀是时机测试，不是恢复被删除的missing_tool/no_context负QA。后端能力列表用于约束委托，工具不会被实际执行；不输入工具返回，不评价完成回复。不覆盖调用链成功率、环境状态变化或执行后纠错。
- **数据质量**：已具备生成和人工标注工具，但本次未核验全量人工复核、标注者一致性及数据泄漏检查结果；论文中不能直接宣称达到其他工作的人工验证标准。

## 5. 建议的论文定位句

> AOI-Bench evaluates event-grounded response and task delegation in streaming audio-visual environments. It crosses twelve semantic capabilities with three interaction modes—continuous updates, proactive reminders, and backend task delegation—and separately assesses response timing, user-facing content, and delegated task instructions through realtime and paired-prefix probe protocols.

这里的delegation指生成并记录后台任务指令，不意味着已经执行工具。

## 6. 适合支撑区别的后续实验

1. 对相同语义能力分别报告RTD、PR和Agent，验证委托不只是给普通回复增加格式要求。
2. 同时报告Agent窗口命中、完整双段输出比例、回复分与instruction分；补人工抽样核验judge。
3. 对比格式协议与示例提示，区分模型能力不足和输出格式失败。
4. 按工具必要性、参数事实充分性及授权范围分析instruction错误；与空泛的“继续监控”等无执行意义指令区分。
5. 补充长视频和模态消融时控制任务构成；明确训练/评测视频及语义近重复隔离。

以上为建议，未在本次启动实验或修改现有队列。

## 7. 核对依据

- StreamingBench：论文§3–4及评测附录，https://arxiv.org/html/2411.03628 。
- OVO-Bench：论文§3–4、§6，https://arxiv.org/html/2501.05510 。
- OmniMMI：论文§3.2及统计表，https://arxiv.org/html/2503.22952 。
- OmniPro：论文§3.1–3.2，https://arxiv.org/html/2605.18577 。
- OmniInteract：论文§3–4，https://arxiv.org/html/2605.26485 。
- Omni-DuplexEval：论文§3及评测附录，https://arxiv.org/html/2605.17360 。
- 本地发布数据：`/remote-home/yushengliao/speech/jimodels/datasets/AOI-Bench/all_qa.json`，本次核验2107 QA、852视频、36个能力/交互组合、723条positive Agent。
- 本地实现：`benchmark/src/evaluation/timing.py`、`benchmark/src/evaluation/judge.py`、`benchmark/src/evaluation/probe_judge.py`、`benchmark/src/probe/data.py`、`benchmark/src/probe/prompts.py`、`benchmark/src/config/defaults.py`、`benchmark/configs/prompts/agent_task_text.txt`。
