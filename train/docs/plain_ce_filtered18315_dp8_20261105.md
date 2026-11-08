# 筛选后 AOI-Dataset：八卡普通 CE SFT

最新重启与连续训练内存修复见 [八卡 allocator 修复](plain_ce_filtered18315_dp8_allocatorfix_20261105.md)。本页保留最初两次尝试记录。

按用户要求，本轮由四卡 profile 切换到 **0–7 八卡 DP**。四卡正式训练尚未开始；原 profile 已停止，八卡 profile 通过后重新从原始基座开始正式 SFT。原 benchmark 队列及结果已保存，SFT 完整结束后恢复原四卡（4–7）队列。

## 配置和数据

实际冻结配置：`train/outputs/plain_ce_filtered18315_dp8_20261105/sft.json`；可复用配置：`train/configs/minicpmo_full_sft_plain_ce_dp8.json`。

- MiniCPM-o-4.5 原始基座，全参数普通 CE，音视频编码器、连接层和 LLM 均训练；TTS 不加载。
- 8 卡 × 每卡 1 QA × 梯度累计 **2** = 有效 batch **16**。四卡版本累计为 4，两版保持相同 batch。
- LR 1e-6，weight decay 0.01，clip 1.0，2 epoch，预计 **1,852 次更新**。
- 完整视频、全部计分 response、问题后空闲秒沉默监督；总 token 上限 65,536，每秒发声 chunk 20 token。Agent 仍为普通文本分隔符加 instruction。
- BF16、FlashAttention 2、ZeRO-2 CPU optimizer offload；激活 CPU 预算 128 GiB/rank、GPU 预算 4 GiB/rank、GPU 保留 12 GiB/rank。
- 每 50 步保存，保留一个编号 checkpoint 和 final；TensorBoard 开启，W&B disabled。

数据审计和划分已重新核验：18,315 个源 QA，排除 35 条目标超视频末尾和 2 条调度冲突，有效 18,278 条。训练 **14,812 QA / 31,539 responses**，验证 **3,466 QA / 7,362 responses**；947 条旧验证 QA、2,519 条新增验证 QA。训练/验证 JSON 与四卡版本逐字节一致，视频划分和采样均未改变。

profile 使用同样 16 条代表 QA，覆盖三种交互、最长时长、多文本和多回复。八卡同时加载和 CPU 激活卸载的主存占用需要重新实测，不能用四卡 profile 代替。主存/显存或完整媒体 token 检查失败时阻止正式训练，不自动修改预算、截断视频或排除选中的 profile QA。

## 路径和监控

- 操作配置、状态、日志：`train/outputs/plain_ce_filtered18315_dp8_20261105/chain.json`、`status.json`、`controller.log`、`profile.log`、`training.log`。
- Prepared：`train/prepared/filtered18315_plain_ce_dp8_20261105`。
- profile 报告目录：`/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/plain_ce_filtered18315_dp8_profile_20261105`，不保存 profile checkpoint。
- 正式权重：`/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft_plain_ce_dp8_filtered18315_bs16_20261105`。
- TensorBoard：现有 localhost:6006，根目录 `train/tensorboard`；新运行正式初始化后出现。
- benchmark 暂停和恢复记录：`benchmark/results/moss_vl_realtime_dp4_20261029/paused_for_sft_dp8_20261105.json`。后续依次继续 Baichuan-Omni-1.5、HumanOmniV2、OmniVinci，保留旧分片和已完成预测。

```bash
# 本轮已由后台控制器启动；此命令只检查配置和数据，不再加载模型。
train/.venv/bin/python train/scripts/sft_then_split_bench.py \
  --config train/outputs/plain_ce_filtered18315_dp8_20261105/chain.json \
  --train-only --check-only
```

`--train-only` 不自动生成新的 AOI/OmniPro 评测任务；本轮 `continuation` 显式恢复的是暂停的其他模型队列。仅在 SFT 正常退出、全部八个 rank 报告及 final checkpoint 校验成功后执行该恢复，使用原队列的独立环境。


## 八卡主存保护停机与重启（2026-11-05）

首次八卡 profile 的 `MemAvailable` 降至 2.90 GiB，控制器以 `Profile host memory reserve exhausted` 停机；未进入正式 SFT，也没有训练 checkpoint。128 GiB/rank 的 CPU 激活预算未给八卡并发留足主存。

修复尝试的配置、状态和日志在 `train/outputs/plain_ce_filtered18315_dp8_memfix_20261105/`。仅本轮运行调整激活预算：CPU 128→88 GiB/rank，GPU 4→32 GiB/rank，GPU reserve 仍为 12 GiB；控制器主存保护 16→64 GiB，每 2 秒检查。源通用配置未修改。

复用原冻结 runtime、审计及逐字节不变的 prepared，`memory_fix.json` 记录差异及失败证据，原输入哈希仍全部校验。重新使用相同 16 条代表 QA 进行完整前后向预检，输出到外部 `plain_ce_filtered18315_dp8_profile_memfix_20261105`。通过后自动从基座开始原正式 SFT；预算是否足够仍以本次 profile 为准。数据、loss、学习率、有效 batch 16、2 epoch 和 1,852 updates 保持不变。
