# 本地产物归档（2026-10-23）

本次将旧实验产物移动到仓库根目录的 `legacy/20261023T023412Z/`，原有相对路径完整保留。
共 4,039 个文件、4,120,828,197 字节（约 3.84 GiB），不是删除或释放磁盘空间。
移动前后逐文件核对 SHA256、大小与 inode；详细清单位于本地 `legacy/20261023T023412Z/manifest.json`。

## 已归档

- Benchmark：已停止的七字段 Agent 重评、旧续跑/修复计划、早期 56 条及 GroupA100 实验、旧 post-media 结果、MiniCPM 早期产物、judge smoke 和窗口迁移记录。
- 数据生成：Gemma4 路由试验、早期 Qwen caption，以及 `qwen3.8_omni_flash_44` 下除 `all_qa.json` 之外的计划、逐视频输出、worker、attempt 和诊断记录。

上述 873 个原先受 Git 跟踪的生成产物已从索引移除；根 `.gitignore` 忽略整个 `/legacy/`。
这些文件仍保存在本地归档内，仓库中的历史统计报告和图表保持原样。

## 保留在原位置

- 当前 `agent_task_start_v1_20261023` 运行与 `full_2110_dp6_after_task_start_20261023` 续跑队列。
- 队列复用的 `full_2110_dp6_20261021` 模型结果，以及当前全量输入 `full_omni_duplex_plus_ab_gemini37_20261021`。
- gallery 配置引用的 `data_parallel_4567_20261007` 和 `remaining_4567_20261008T022841Z`。
- gallery 仍引用的 `mmduet2_recovery_20261008T062632Z` 历史结果也保留原位置；没有运行 MMDuet2。
- `data_generation/data/qwen3.8_omni_flash_44/all_qa.json`、`data_generation/data/omni-duplex/` 中的现有数据及压缩包。
- 其他合并输入集、完整 judge 结果、Vertex 原始包、对比快照与 Markdown/PDF/图表报告。

历史报告中如果提到已归档的原路径，在路径前加 `legacy/20261023T023412Z/` 即可找到对应文件。
例如旧 `benchmark/results/agent_protocol_v2_20261022/` 现位于
`legacy/20261023T023412Z/benchmark/results/agent_protocol_v2_20261022/`。

## 恢复

先在 manifest 中查找 `source` 和对应文件校验值，再把归档中的条目移回原相对路径。
若原位置已有新结果，先恢复到另一个目录核对，避免覆盖。归档是本地保存，不会随普通 Git clone 下载。
