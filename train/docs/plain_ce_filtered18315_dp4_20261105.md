# 筛选后 AOI-Dataset：普通 CE 四卡 SFT

**状态更新：用户已改为八卡训练，本四卡 profile 已停止，正式训练未开始。** 当前运行见 [八卡配置](plain_ce_filtered18315_dp8_20261105.md)。下列内容保留为四卡配置记录。

本轮从原始 MiniCPM-o-4.5 重新训练，使用 Gemma4 全计分 response 通过的本地 AOI-Dataset 版本。0–3 卡训练，4–7 卡保留当前其他模型的 AOI-Bench；先实卡 profile，通过后自动正式训练。

## 数据和审计

- 正式源：`/remote-home/yushengliao/speech/jimodels/datasets/AOI-Dataset/all_qa.json`，18,315 QA，SHA256 `1da2312fa7c74969a9d346408bad2f0142f1b5d34bd922a2e11e8a009a5e1459`。
- 新输入快照、完整 metadata、质量筛选说明与 CPU 审计：`train/outputs/plain_ce_filtered18315_dp4_20261105/inputs/`、`audit.json`。不复用未筛选 19,210 QA 的 prepared。
- 有效 18,278 QA / 38,901 个计分 response；排除 37 QA / 151 responses：35 个目标超出实际视频末尾，2 个回复调度冲突。数据集本身仍保留 18,315 条，训练单独记录无法执行的 QA，不截断目标或缩短视频。
- 训练 **14,812 QA / 31,539 responses**；验证 **3,466 QA / 7,362 responses**。
- 继承历史 5k manifest 的视频级划分；新视频按原 seed/hash 规则划分。验证集分为 previous 947 QA、added 2,519 QA；所有有效 QA 被完整分配，`sampling_omissions=0`。
- 与筛选后 AOI-Bench 的视频 basename/UUID 交集为零；这是文件名身份检查，不是内容指纹去重结论。
- 最长训练/验证轨迹 415 秒，平均约 225 秒。CPU 审计使用真实 tokenizer，但不代替完整媒体 token 与实卡显存测试。

## 配置

| 设置 | 本轮 |
| --- | --- |
| 可复用配置 | `train/configs/minicpmo_full_sft_plain_ce_dp4.json` |
| 实际冻结配置 | `train/outputs/plain_ce_filtered18315_dp4_20261105/sft.json` |
| 初始化 | 原始外部 MiniCPM-o-4.5；不续用旧 SFT/优化器 |
| 更新参数 | LLM、音视频编码器与连接层；不加载 TTS，不使用 LoRA |
| loss | `plain_ce`，控制/回复/instruction 有效 token 统一均值，再按 QA 平均 |
| GPU / batch | 0–3 DP，每卡 1 QA，梯度累计 4，有效 batch 16 |
| LR / weight decay / clip | 1e-6 / 0.01 / 1.0 |
| epoch / 预计更新 | 2 / 1,852 次；每轮 926 次，尾批补空槽而不重复有效 QA |
| 视频与文本预算 | `full_video`，全部回复，总序列 65,536 token，每秒发声 chunk 20 token |
| 沉默监督 | 问题后空闲秒，包括回复窗口内非参考位置与末次回复后至视频末尾 |
| Agent 输出 | 回复文本 + 普通文本 `<\|task_start\|>` + instruction，不增加单独 token，不使用 JSON 目标 |
| 精度 / attention | BF16 / FlashAttention 2 |
| 分布式 / 优化器 | 独立 QA DP、ZeRO-2，AdamW optimizer 状态卸载 CPU |
| 激活缓存 | CPU 128 GiB/rank、GPU 4 GiB/rank、GPU 保留 12 GiB/rank |
| 保存 | 每 50 次更新；保留一个编号 checkpoint 和 final |
| 监控 | TensorBoard，W&B disabled；保留各项 loss 和 previous/added 验证指标 |

`max_prefix_seconds=300` 和 `post_response_silence_seconds=0` 沿用配置值；`sft_horizon=full_video` 决定实际覆盖完整视频，因此没有在 300 秒或末次回复处截断。

## 哪些调整已经执行，哪些保留为后续选择

已经执行：更换为筛选后的源快照，重新审计和 prepare；四卡累计改为 4，保持与八卡基线相同的有效 batch 16。其他优化目标和学习率沿用普通 CE 基线，便于分析数据扩容/筛选带来的变化。

实卡 profile 使用训练划分中的 16 条代表 QA，覆盖三种交互以及长时长、多文本、多回复等极端情况，完成一次有限更新后退出；不保存 profile 权重。通过且无排除后，正式训练重新从基座加载。profile 失败则阻止自动训练，不自动放宽数据/上下文/显存约束。

后续可评估：

1. 先看第一个 epoch 的验证趋势，再决定是否需要第二轮或更小 LR；当前计划仍为两轮，不引入新的 gate/onset 权重。
2. 当前激活 GPU 缓存预算 4 GiB 比较保守，可根据 profile 的实际峰值另测更高预算，减少 CPU 搬运；不预先假定一定加速。
3. 当前只保留最新编号 checkpoint；若需要回选第一轮或最佳验证模型，应另行确定 epoch/best 保留策略及磁盘预算。
4. teacher-forcing SFT 的 loss 改善不能保证自由生成的重复开口和后续更新问题消失，仍需 Realtime benchmark 验证。

## 操作与路径

控制器：`train/outputs/plain_ce_filtered18315_dp4_20261105/chain.json`、`status.json`、`controller.log`，阶段日志 `profile.log` / `training.log`。

```bash
# 配置和数据验证，不加载模型
train/.venv/bin/python train/scripts/sft_then_split_bench.py \
  --config train/outputs/plain_ce_filtered18315_dp4_20261105/chain.json --train-only --check-only

# 仅在没有同一运行的控制器时启动；本轮已启动
train/.venv/bin/python train/scripts/sft_then_split_bench.py \
  --config train/outputs/plain_ce_filtered18315_dp4_20261105/chain.json --train-only
```

该入口的 `--train-only` 明确只完成 profile 和 SFT，不占用后四卡的后续评测任务。训练结束必须通过各 rank 报告、预期更新数和完整 checkpoint 校验。

- Prepared：`train/prepared/filtered18315_plain_ce_dp4_20261105`。
- profile 权重目录（仅报告，不保存 checkpoint）：`/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/plain_ce_filtered18315_dp4_profile_20261105`。
- 正式权重：`/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft_plain_ce_dp4_filtered18315_bs16_20261105`。
- TensorBoard 根目录：`train/tensorboard`；现有服务监听 localhost:6006，新运行初始化后自动出现。

所有权重实际位于仓库外。仓库中的 outputs/runtime 只保存配置、数据快照、日志和代码，不保存模型权重。
