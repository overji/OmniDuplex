# Gemma4 数据标注审查

当前自动顺序为：**现有普通 CE 模型的 AOI-Bench、OmniPro 和四卡 Judge 结束 → 八卡 Gemma4 审查 → 继续其他模型的 AOI-Bench**。全量 AOI-Dataset SFT 已取消自动启动，等待人工阅读审查报告后决定。

审查只生成报告，不删除 QA、不修改时间窗口或参考答案，不替换训练集和评测集。当前快照包含 AOI-Bench 的 2,107 条 QA 和 AOI-Dataset 的 19,210 条 QA，共安排 47,671 个 response 审查任务。Agent 只审查当前评测协议中计分的委托回复；不把工具返回后的完成回复纳入。

## 审查内容和证据

先检查文件、时间窗口、工具可用性和现有 SFT 调度审计，再用本地 `gemma-4-E4B-it` 检查问题清晰度、任务匹配、回复正确性、触发依据、时间因果性、模态、计数一致性，以及 Agent 委托必要性和指令。问题是候选标注的一部分，不能当作可信证据。

每个 response 只提供到其窗口结束时的原始音视频前缀；视频最多 24 帧，兼顾前缀均匀采样和窗口附近，音频覆盖整个前缀，以 30 秒分段输入。不得利用窗口内较晚证据证明较早时刻已成立的结论。稀疏采样无法确认的短暂事件、文字或精确时刻标为 `uncertain`，不能据此断言标注错误。因此，这份模型审查报告需要人工复核，不能替代逐帧核验。

模型输出必须包含九项检查；判为 `issue` 时必须给出前缀内带时间戳的证据。格式失败最多生成两次；上下文溢出、解码失败和运行错误单独记为 `failed`，不算标注错误，也不自动缩短前缀。终态失败记录会保留，重新运行不会反复重试同一项。

## 运行和恢复

基础设置见 `configs/gemma4.json`。本轮配置、输入快照和状态保存在 `outputs/gemma4_aoi_20261103/`，不会提交到 Git。模型权重始终使用仓库外的 `openmodels` 路径。

```bash
# 从仓库根目录执行；当前任务已由等待控制器负责，不要重复启动。
/root/miniconda3/envs/omniduplex/bin/python -m data_quality \
  --config data_quality/outputs/gemma4_aoi_20261103/config.json prepare
/root/miniconda3/envs/omniduplex/bin/python -m data_quality \
  --config data_quality/outputs/gemma4_aoi_20261103/config.json run
```

`prepare` 固定输入哈希并按视频分配八个独立副本；已有 manifest 不允许换数据或配置后继续写入。`run` 等待现有评测和 Judge 到达终态、旧进程退出、八卡释放后运行审查。每分钟更新报告，支持从完整 JSONL 记录恢复。在输出目录创建 `STOP` 可停止这条队列及它启动的子进程，不会停止等待中的现有 benchmark。

需要与后四卡 Judge 并行时，执行 `run --gpus 0 1 2 3`。这只调整执行资源：八份原始分片和已完成 JSONL 均保持原样，每张卡依次处理原有分片，不重新划分或重复评分。Judge 完成后的自动恢复若发现审查控制器已运行，会直接复用，不启动第二组审查。

## 输出和后续队列

- `status.json`：等待、审查、后续评测或失败状态。
- `report.md`、`summary.json`：覆盖率、静态检查、pass / issue / uncertain / failed 汇总。
- `qa_review.json`、`response_review.csv`：供人工复核的问题、参考回复和审查结论。
- `worker_*/judgments.jsonl`：原始模型输出、重试、证据时间、音视频覆盖和耗时；同目录 `worker.log` 为运行日志。

八个副本均正常退出、任务全部有终态记录且有效审查比例至少 90% 后，继续原有四卡（4–7）AOI-Bench 队列：**Baichuan-Omni-1.5 → HumanOmniV2 → OmniVinci**。复用已完成结果和既有分片；保留原来的失败项跳过设置。大量运行失败会停止自动衔接并保留报告；发现标注问题本身不阻止其他模型评测，因为本轮不修改数据。
