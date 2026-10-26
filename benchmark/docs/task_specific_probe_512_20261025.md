# Probe 任务提示词拆分与 512-token 队列

2026-10-25：暂停旧看护和旧推理后，保留原始结果、配置、日志，并保存修改前代码快照。
新队列重新生成全部 probe；没有导入旧共享提示、256-token 条件的预测。
已完成的 MiniCPM/Gander/JoyAI realtime Agent 结果保留，后续 realtime 命令未改变。

## 提示词与参数

- 公共提示 `configs/prompts/probe_common.txt`：因果前缀、已处理机会、实际回复和 `<silent>` 规则。
- RTD：新增状态变化的简短正文；附玩具火车位置变化的一正一负示例。
- PR：条件确认后的简短提醒，已在前缀中发生但不在最后一帧的事件仍有效；附指示灯示例。
- Agent：`回复<|task_start|>instruction`，仅提供当前 available 能力；`note_write` 可用时附包裹投递记笔记示例。
- RTD/PR 的 system 和 user prompt 均没有 Agent 分隔符或工具列表。

所有常规和消融 probe 显式设置 `--prompt-layout task-specific --max-new-tokens 512`。
CLI 默认仍是历史 shared/256，避免改变旧命令的行为。队列参数集中在 `configs/suites/reply_task_text.json`。
1 FPS、像素上限、完整前缀、音频输入、停止条件及模型原生 beam 设置维持不变；没有做截断根因 A/B。
512 仅是更宽松的生成上限，不保证消除重复输出、显存不足或上下文超限；达到上限仍记录 truncated。
输出协议未变，仍可用 `probe-judge` 对 TP 单独评分；本次没有调用付费 API。

## 当前状态入口

根目录：`benchmark/results/task_text_agent100_20261024_v2`。

- `active_queue.json`：当前队列、看护和计划的指针。
- `queue_transition_task_specific_512.json`：新交接配置，跳过已完成的三个 realtime Agent 重跑。
- `task_specific_probe_512_v3/transition_status.json`：新监督器状态。
- `task_specific_probe_512_v3/suite/suite_status.json`：实际模型队列进度。
- `task_specific_probe_512_v3/suite/suite_plan.json`：展开后的命令和 GPU 分片。
- `task_specific_probe_512_v3/watchdog/status.json`：每 15 分钟的看护结果。

顺序保持：前三个 probe（MiniCPM、Qwen3-Omni、Qwen2.5-Omni）→四组消融→LiveStar→后五个 probe。
使用八卡 DP，Qwen3-Omni 为四实例、每实例两卡。不运行 MMDuet2。
每个新任务先跑既定 smoke，推理成功后才展开全量；不按模型是否答对筛选实验。

## 历史结果

旧结果仍在 `task_text_reply_queue_v2/`，旧看护留有 `watchdog/PAUSE`，不会重新抢占 GPU。
修改前的 benchmark 实现、配置、脚本和测试在 `task_text_reply_queue_v2/code_before_task_specific/`。
旧运行因显式暂停显示 interrupted；这不代表预测文件被删除。
新旧运行的结果应分开报告，不能把新参数当作原条件的断点续跑。

## 验证

64 项 CPU 回归通过，覆盖任务隔离、示例能力过滤、无参考泄漏、shared 行为、模板指纹、
512 预算、队列顺序、realtime 命令一致、TP judge 及看护。
13 个任务的全部完整/worker/smoke CLI 参数通过解析检查；GPU smoke 由后台队列实际执行。
MiniCPM 的一对 Agent smoke 已实际完成：两侧均正常返回、双字段格式合规且无截断；
负侧误回复、正侧回复，说明 smoke 确认推理流程可用，不证明准确率提升。随后已进入八卡全量生成。
