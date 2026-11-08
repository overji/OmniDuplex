# 八卡全量普通 CE SFT：连续训练内存修复

本轮操作目录：`train/outputs/plain_ce_filtered18315_dp8_allocatorfix_20261105/`。

## 故障和修复

前次八卡 profile 通过，正式训练完成一次更新后，在第三个 microbatch 的 rank 6（QA 15676）触发 CPU 激活预算上限。不是已完成训练：没有保存 checkpoint。

`saved_on_cpu` 原先只用 CUDA 驱动空闲值计算可保留的 GPU 激活。第一条 QA 后 PyTorch 缓存较多显存，驱动认为这部分已占用，实际却可由后续 QA 复用。结果日志中 GPU 激活从 32 GiB 缩至约 5–7 GiB，激活被错误挤到 CPU，最终超过 88 GiB 上限。

修复将可用量计算为 `driver_free + max(0, memory_reserved - memory_allocated)`，再扣除原有 12 GiB 工作区余量、限制在配置的 32 GiB 内。没有将活跃 tensor 内存算作可复用。新增 `activation_gpu_budget_gib` 日志展示每个 microbatch 实际可用的预算；原有 kept 和 saved 计数保留。

CPU 88 GiB/rank、GPU 32 GiB/rank 均保持本轮前次配置。没有增加内存上限、删除长视频或改 loss。

## 多步预检

原 profile 固定一个 optimizer update、累计 1，仅覆盖候选集的一部分，不能验证后续缓存行为。新增可选运行配置：

- `profile_gradient_accumulation_steps`：默认 1；本轮显式设 2，与正式训练一致。
- `profile_optimizer_steps`：默认 1；本轮显式设 2。

选 32 条不同 QA，覆盖先前最长/多回复代表 QA、实际训练首批 QA，以及完整失败批次。包括 QA 15676。八卡每卡 4 个 microbatch，处理完全部 32 条、完成两次更新后才允许正式训练；每个 rank 必须报告两次更新、有限 loss、零排除，且不保存 profile checkpoint。profile 不是最终 SFT，结束后正式训练从原始基座开始。

本轮冻结修复后的 runtime，保留原 prepared、审计与数据文件，输入 SHA256 均核对。`restart_reason.json` 记录原失败证据、修复原因和选样；不覆盖旧尝试日志或输出目录。

## 正式配置及去向

- 0–7 八卡 DP，每卡 1 QA，梯度累计 2，有效 batch 16。
- 全参数普通 CE，完整视频、全部计分 response、沉默监督；LR 1e-6，2 epochs，1,852 次更新。
- 训练 14,812 QA，验证 3,466 QA，划分不变。
- 每 50 步保存，编号 checkpoint 保留一个，TensorBoard 仍开启。
- profile 权重目录（不保存权重）：`/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/plain_ce_filtered18315_dp8_profile_allocatorfix_20261105`。
- 正式输出：`/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft_plain_ce_dp8_filtered18315_bs16_allocatorfix_20261105`。
- `status.json` 明确区分 `sft_profile` 与 `plain_ce_training`。原 benchmark 继续暂停，SFT 正常完整结束才恢复原队列。

```bash
cat train/outputs/plain_ce_filtered18315_dp8_allocatorfix_20261105/status.json
# profile 和正式训练分别写 profile.log / training.log。
```
