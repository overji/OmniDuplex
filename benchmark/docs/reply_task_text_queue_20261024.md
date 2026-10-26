# 后续回复／沉默协议与运行队列

Realtime Agent 使用 `回复<|task_start|>instruction`，支持 MiniCPM、Gander、JoyAI 和 LiveStar；
沿用各模型原生开口策略。MiniCPM realtime 两组消融也使用相同格式。
Probe 使用 `--output-protocol reply-task-text`：沉默输出 `<silent>`，Agent 正判输出上述双字段，
RTD/PR 正判输出正文；不再生成 YES/NO 标签。不扩词表，不生成 JSON，不执行工具。

## 实际队列

先完成现行 MiniCPM、Gander、JoyAI 各 723 条全量 Agent 重跑，再自动运行：

| 顺序 | 模式 | 任务 |
| --- | --- | --- |
| 1 | Probe | MiniCPM-o-4.5 |
| 2 | Probe | Qwen3-Omni-Instruct |
| 3 | Probe | Qwen2.5-Omni-7B |
| 4 | Realtime 消融 | MiniCPM 仅视觉 |
| 5 | Realtime 消融 | MiniCPM 仅音频 |
| 6 | Probe 消融 | Qwen2.5-Omni 仅视觉 |
| 7 | Probe 消融 | Qwen2.5-Omni 仅音频 |
| 8 | Realtime | LiveStar 全量补齐，全部 Agent 重新生成 |
| 9–13 | Probe | Ming、Qwen3-VL、Baichuan、HumanOmniV2、OmniVinci |

使用八卡数据并行，Qwen3-Omni 为四实例、每实例两卡。不运行 MMDuet2。
新任务先跑一个 Agent 样本或一对 Agent probe 的 smoke；推理失败时跳过该任务的全量，最终队列状态为 incomplete。
Smoke 不要求模型答对或生成合法调用，避免按结果筛选模型。

根目录为 `benchmark/results/task_text_agent100_20261024_v2`：

- `frozen_agent_runtime/` 保存前三个 Agent 实验的旧协议代码和提示词快照；当前 MiniCPM 已加载旧代码继续运行。
- `queue_transition.json` 指定交接进程、冻结任务命令与新队列入口。
- `task_text_reply_queue_v2/transition_status.json` 是交接监督器状态；`transition.log` 为监督日志。
- `task_text_reply_queue_v2/commands.json` 是新队列命令；`suite/suite_plan.json` 是展开后的实际 GPU/worker 计划。
- `task_text_reply_queue_v2/suite/suite_status.json` 在交接后记录新队列进度。新结果均保存于该 suite 下。

旧 dispatchers 暂停派发，当前 MiniCPM 的 worker 继续工作；交接监督器等它完成后关闭旧 dispatchers，
使用冻结代码运行 Gander、JoyAI，随后启动新 suite。失败时保留结果和日志并停止交接，避免多队列争抢 GPU。
旧顶层 `status.json` 仅代表原监督器，交接后请以上述 transition/suite 状态为准。

## LiveStar 结果复用

`--agent-output-protocol task-text` 启用新格式；`--reuse-non-agent OLD_PARALLEL_DIR` 仅导入旧目录中
已完整完成的 RTD/PR。导入核验数据和基础提示词 SHA256、模型路径和版本、视频目录、官方代码路径、
原生采样/生成设置及每条系统提示词指纹；每条记录保存来源和原 plan SHA256。
本次核验通过 114 条非 Agent，Agent 一律不导入，共 723 条重新生成；剩余非 Agent 继续生成。
这是配置及记录身份核验，不是对旧视频和全部模型权重字节的历史校验。
时钟保持原生帧 t 输入后在 t+1 秒计回复；工具返回不输入模型。

## 准备其他同类队列

顺序及协议集中于 `benchmark/configs/suites/reply_task_text.json`。从已准备好的模型命令构造新队列：

```bash
python benchmark/scripts/prepare_reply_queue.py \
  --source-commands PATH_TO_EXISTING_COMMANDS.json \
  --output PATH_TO_NEW_COMMANDS.json \
  --reuse-livestar PATH_TO_OLD_LIVESTAR_PARALLEL_RUN \
  --agent-indices PATH_TO_AGENT_INDICES.json

python benchmark/scripts/run_prepared_suite.py \
  --commands PATH_TO_NEW_COMMANDS.json --output-dir PATH_TO_NEW_SUITE \
  --gpus 0 1 2 3 4 5 6 7 --dry-run
```

去掉 `--dry-run` 才启动推理；模型和环境路径沿用 source commands，不自动下载或修改采样参数。
旧 JSON/native Agent 及历史 YES/NO probe 结果保留原目录，不混入新协议。

## 每 15 分钟自动看护

`benchmark/scripts/watch_benchmark_queue.py` 是独立进程，当前队列已挂载后台看护。
参数集中在 `benchmark/configs/watchdog.json`：每 900 秒检查一次，连续 3600 秒无有效结果或原始流式 chunk 进展才按卡死处理；已完成 worker 不参与卡死判断。
进程退出、worker 非零退出、服务失败和 suite 未完成都会触发原命令断点重启。它不修改推理参数、提示词或评分，也不删除预测文件。
重启时先核验父/子进程的命令身份及 PID 启动时间，只清理该队列的进程树；其他任务占用 GPU 时等待，不终止其他任务。
没有新增完成记录时，同一任务最多自动重启 3 次；预算耗尽或不能安全处理的情况记录 `needs_attention`，不会无休止重启，也不会自动改源码、重装环境、降低输入质量或跳过失败数据。

前台手动启动方式（同目录由文件锁禁止重复看护）：

```bash
benchmark/.venv/bin/python benchmark/scripts/watch_benchmark_queue.py \
  --transition-config benchmark/results/task_text_agent100_20261024_v2/queue_transition.json
```

只读诊断、不重启任务：

```bash
benchmark/.venv/bin/python benchmark/scripts/watch_benchmark_queue.py \
  --transition-config benchmark/results/task_text_agent100_20261024_v2/queue_transition.json \
  --output-dir /tmp/omniduplex-watchdog-check --once --dry-run
```

看护状态和历史位于 `task_text_reply_queue_v2/watchdog/status.json`、`checks.jsonl`；
`state.json` 保存重试预算，`restarts.log` 保存自动恢复日志，重启前的 transition 状态另存档。
队列全部完成后看护自动退出。硬件故障、协议/配置身份不一致等不能只靠重启解决的问题会保留诊断记录。

有意暂停实验时，先创建 `task_text_reply_queue_v2/watchdog/PAUSE`，再停止队列；该文件只暂停看护修复，不停止正在运行的实验。
删除 PAUSE 恢复检查；明确标为 `interrupted` 的人工停止仍受尊重，需要手动启动 transition 才恢复实验。
向看护本身的 PID 发送 SIGTERM 只退出看护，队列保持运行。
