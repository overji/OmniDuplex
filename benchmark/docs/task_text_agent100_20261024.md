# 三模型 Agent 纯文本分隔符试验（2026-10-24）

## 协议与实际提示词

模型生成 `回复内容<|task_start|>agent instruction`，不生成 JSON。
分隔符在三个原始 tokenizer 中均为 6 个普通 token，不新增词表、不训练权重。
程序只在解析成功后转换为内部 `task_start(instruction)` 结构；不据参考答案补造调用。

完整提示由以下固定正文、模型补充和一次 available 能力列表构成，旧 JSON schema 和旧应用委托格式不进入本轮提示：

- [共用提示](../configs/prompts/agent_task_text.txt)
- [MiniCPM 补充](../configs/prompts/agent_task_text_minicpmo.txt)
- [Gander 补充](../configs/prompts/agent_task_text_gander.txt)
- [JoyAI 补充](../configs/prompts/agent_task_text_joyai.txt)

共用要求：观察到触发后，生成一句面向用户的回复，再输出一次精确分隔符，之后生成一句包含已观察事实、工具目标及授权范围的 instruction。不得使用 JSON、Markdown，不声称工具执行成功，不重复委托；缺少必要信息时只澄清。

MiniCPM/Gander 保留原生 listen/speak 决策，并从原始 token 增量跨单位解析，避免只依赖原生返回文本。Gander 仅在此显式实验中不注册原生工具 schema，关闭原生 JSON 工具动作；正文按新格式输出。JoyAI 保留 `</silence>` 和 `</response>`，后者之后的正文为新格式；整条内容保持单行，避免服务端单行归一化丢失 instruction。

实际渲染提示的首个样本为 sample 8，已保存到
`benchmark/results/task_text_agent100_20261024_v2/prompts/{minicpmo,gander,joyai}_sample_8.txt`。
每个执行样本的最终提示还会保存在 worker 的 `sample_system_prompts.jsonl`。

## 样本与执行

配置：[task_text_agent100.json](../configs/suites/task_text_agent100.json)。
从三者旧版共同完成的 688 条正 Agent QA 中，以 seed=42 按 ability 比例抽取同一组 100 条，覆盖全部 12 类；保留原始全局 sample_index，不泄漏参考回复或窗口到提示。

顺序：MiniCPM-o-4.5 → Gander → JoyAI，模型间串行；每个模型先在 0 号卡运行所选第一条完整视频，之后八卡 DP 运行其余 99 条，合计恰好 100 条，不重复计算 smoke。JoyAI smoke 和剩余分片失败时各允许一次原命令续跑重试。

用户已确认立即插队。原 LiveStar 及所属队列已暂停；已保存预测和原计划保留。三组试验结束后监督器自动恢复 LiveStar，使用原结果目录、原八卡分片、原视频/标注及旧协议身份；随后继续原 probe 和消融顺序。不会自动重开已完成的 Gander 或本轮范围外的旧 JoyAI 缺失补跑。

当前监督器入口：

```bash
benchmark/.venv/bin/python benchmark/scripts/run_task_text_trial.py \
  --config benchmark/configs/suites/task_text_agent100.json \
  --output-dir benchmark/results/task_text_agent100_20261024_v2
```

已有后台实例运行时不要重复启动；只生成清单及快照使用 `--dry-run`。改变配置、代码或提示需新结果目录。

## 输出与解释

根目录：`benchmark/results/task_text_agent100_20261024_v2/`。

- `status.json`：监督器 PID、当前模型、阶段、失败与重试状态。
- `plan.json`、`indices.json`：模型命令、数据/协议指纹、共享样本和类别配额。
- `baseline/`：仅这 100 条旧结果的只读快照；旧目录不改写。
- `<model>/smoke/`、`<model>/full/`：真实原始输出、提示、推理配置、分片预测。
- `<model>/predictions.jsonl`：合并后的 100 条结果；`format_metrics.json` 为格式统计。
- `report.md`、`comparison.json`：定期更新的新旧时间指标、格式统计与原始示例。
- `continuation/`、`continuation.log`：试验后的原队列续跑。

严格记录分隔符数量和两侧文本。缺失、重复、空字段、明确截断分别标记；明确截断不形成有效内部调用。对于 JoyAI 未提供可信底层终止信号的情况，完成状态记录为 unknown，不宣称已证明语义完整。

报告区分“文本双字段格式合规”“内部 schema 合法”与“语义正确/原生工具执行成功”。不额外调用付费 judge。试验结束前两版本覆盖率不同，不据进行中的汇总下最终结论。MiniCPM/Gander 仍为音视频输入，JoyAI 仍为视觉输入，此差异不在本轮修改。

## 验证

已完成新协议和既有协议相关 CPU 检查，包括跨秒分隔符、中英文、重复/空字段/截断、真正 raw IDs 的首 token 保留、Gander 新分支关闭 schema、JoyAI 单行输入和旧 LiveStar 协议指纹不变。各模型真实 GPU 推理按队列先 smoke 再全量，实际完成情况以 status.json 为准。

首轮真实 smoke 已生成回复和 instruction，但暴露了 tokenizer 未将原生 turn_eos 标为 special 的问题。已补齐原生控制 ID 过滤；28 项 CPU 检查及真实 smoke token 流回放通过，修正后的独立运行目录为上述 v2。首轮原始结果保留在未带 v2 的旧目录，不计入新试验。

## 追加全量 Agent 重评

用户在 100 条测试期间要求：JoyAI 完成后插入三模型全量 Agent 重评。该修改已生效：JoyAI 的 100 条已全部完成，随后已自动启动 MiniCPM 的全量 Agent 任务。

- 顺序：MiniCPM-o-4.5 → Gander → JoyAI → 原 LiveStar → 其余既有 probe/消融。
- 三模型各 723 条正 Agent QA，均使用八卡 DP 和相同 `task-text` 协议；不改变提示、模态或生成预算。
- 这是独立的全量重新生成，包含先前抽样的 100 条；旧 100 条和旧 JSON/native 结果均保留，不覆盖。
- 单样本 smoke 已由先前 100 条试验验证，本次全量不重复做额外 smoke。
- 全量目录位于 `benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/{minicpmo,gander,joyai}_task_text_agent_full/`。
- 完整运行顺序与监督状态：同一 `continuation/` 目录下的 `suite_plan.json`、`suite_status.json`。
- `full_agent_indices.json` 固定 723 个原始索引，`full_agent_schedule.json` 保存追加配置，原续跑命令备份为 `continuation_commands_before_full_agent.json`。

调度器会保留显式指定的 `--agent-output-protocol task-text`，不会被默认 structured 模式覆盖。CPU 调度/模态测试 10 项通过；实际运行计划也已逐一验证三模型均为 723 条、八卡，之后第一项为原目录 LiveStar。
