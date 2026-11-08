# 八卡 SFT：GPFS 写日志失败后从 step 350 恢复

原运行在 2026-11-06 15:12 北京时间因 `append_jsonl(metrics, result)` 的 flush 返回 `OSError: [Errno 28] No space left on device` 退出，最后记录第 373 步。未出现非有限 loss；不将此故障解释为训练发散。

2026-11-07 检查时 GPFS 显示约 27 TiB 空闲，inode 未满；原目录及新输出父目录均完成 1 MiB 写入 + fsync + 删除测试。没有可用的 GPFS 配额查询工具，因此未确认历史故障究竟来自共享存储瞬时容量、用户配额还是服务异常。没有删除任何旧实验或模型文件。

## 修改

- 增加显式 `--metrics-dir`，使 SFT 的 `metrics.rank*.jsonl` 可存储于本地磁盘，避免每个 microbatch 都依赖 GPFS 写入；checkpoint、模型和优化器仍放外部存储。
- 外部运行目录下的同名 metrics 文件用软链接指向本地，保留既有报表工具发现路径。独立本地日志目录必须是新目录，不拼接旧运行日志。
- SFT 控制器新增 `resume_checkpoint`，启动前核对八份优化器分片、模型文件、八份 RNG 文件及完全一致的训练配置，向原训练入口传递 `--resume`。运行时仍核对原 identity、恢复优化器、epoch/cursor、RNG。
- `metrics_dir` 是操作参数，不进入模型训练配置；本轮配置逐字节沿用失败运行，数据、split、loss、LR、batch 和 epoch 不变。

## 本轮恢复

- 控制器及状态：`train/outputs/plain_ce_filtered18315_dp8_resume350_20261107/chain.json`、`status.json`、`training.log`。
- 本地数值日志：`train/outputs/plain_ce_filtered18315_dp8_resume350_20261107/metrics/`。
- 恢复来源：外部 `full_sft_plain_ce_dp8_filtered18315_bs16_allocatorfix_20261105/checkpoint-350`；所有模型/优化器分片和 RNG 已用 torch.load 逐份解析，global_steps=350，epoch=0，cursor=700。
- 正式输出：`/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft_plain_ce_dp8_filtered18315_bs16_resume350_20261107`。
- 使用 0–7 八卡，batch 16，累计 2，预计最终仍为 1,852 updates；重做未保存的 23 次更新。
- TensorBoard 新 run：`full_sft_plain_ce_dp8_filtered18315_bs16_resume350_20261107`，从 step 350 之后继续记数；旧 run 保留。
- 原 benchmark 队列仍暂停，新的 SFT 完整成功后按原 continuation 恢复。

15 项恢复/控制器/DP 测试通过；真实恢复状态以外部 `resume_state.json` 和本地新 metrics 为准。GPFS checkpoint 写入仍依赖外部服务可用性，本地遥测调整不能保证共享存储永不再次报错。

## 实际恢复验证

八个 rank 已共同完成恢复后的第 351 次更新，loss=0.20249，无非有限值或写入错误。恢复后的第一个 microbatch（slot 700，loss=0.21647482179105282）与原运行同位置完全一致。已确认外部运行目录的 metrics 软链接指向本地磁盘，和 GPFS 分属不同文件系统。详情见本轮 `resume_verified.json`。
