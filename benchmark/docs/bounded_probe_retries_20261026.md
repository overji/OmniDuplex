# Probe失败项的有限重试

当前队列启用 `run_prepared_suite.py --probe-retry-errors-once`，向各probe worker传递 `--retry-errors-once`。

每个worker先处理尚未尝试的请求，再对失败请求补跑一次。每个模型的worker结束后，仍有失败则记为 `incomplete_final`，继续下一模型。整队列结束但仍有失败时状态为 `complete_with_errors`，不将这些结果伪装成全覆盖；watchdog不因这些已耗尽额度的失败反复重启。

`probe_retry_policy.json`保存启用策略时的历史失败次数。历史失败额外获得一次重试，新请求最多首轮加一次重试；进程重启不重置额度。成功请求保留，包括格式错误或长度截断的正常生成，不进行挑选式重跑。

本次运行保留原prompt、模型、媒体处理、512 token生成预算及成功预测。由于probe配置记录完整源码指纹，修改调度代码后对已有配置进行了显式迁移：逐项核对除implementation外的字段一致，原配置备份在各目录的 `probe_config.before_retry_policy_20261026.json`；迁移记录位于当前运行根目录 `retry_policy_migration_20261026.json`。

此策略不会解决OOM或上下文超限；它确保失败保留记录并不再无限阻塞后续模型。失败原因修复后如需额外补跑，应明确建立新的重试阶段，不删除现有策略文件以重置预算。
