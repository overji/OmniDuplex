# GRPO advantage 与聚合的四组对照

四组统一使用 `reward_mode=local_turn_judge`，同一个评分器、Judge、时间匹配、原生动作解析和训练数据。三个新增配置从现有四卡完整方案模板复制，仅修改下面两个选择项。

| 实验 | 配置文件（均位于 train/configs） | grpo_advantage_mode | grpo_loss_aggregation |
| --- | --- | --- | --- |
| 普通 GRPO | minicpmo_full_grpo_baseline_dp4.json | group_relative | token_mean |
| 仅局部 advantage | minicpmo_full_grpo_local_advantage_dp4.json | local | token_mean |
| 仅修改聚合 | minicpmo_full_grpo_unit_aggregation_dp4.json | group_relative | unit_weighted |
| 完整方案 | minicpmo_full_grpo_confusion_judge_dp4.json | local | unit_weighted |

选择项省略时解析为 `local + unit_weighted`，其他 reward_mode 不接受这两个选择项。三个新增模板保留相同起始模型、数据、视频路径、seed、Judge、学习率、group_size=4、QA 累计 batch=4、两轮训练、65536 token 上限和 ZeRO-2。运行前将四组 model_path 统一配置为需要比较的同一个 SFT HF 导出。

## Reward 和 advantage

所有组使用 [局部奖励说明](grpo_local_reward.md) 中的同一终局总分与局部归因。评分器与 Judge 不读取实验选择项，改变实验模式不会改变同一轨迹的匹配或分数。

`group_relative` 对候选的终局 `reward.total` 使用现有 GRPO 标准化 `(R-mean(R))/(std(R)+1e-4)`，std 使用总体标准差，std<=1e-8 时跳过该组。这个共享 advantage 用于该轨迹问题输入后的全部实际采样动作。

`local` 保留零基线信号。回复 token 使用回复的合并 advantage，FN 归给对应窗口等待，正确等待使用 +1，成功窗口等待及其他中性动作使用0。整组总分相同仍可更新，仅局部信号全零时跳过。

## 两种聚合

`token_mean` 对问题输入后的所有采样动作求平均。中性位置的局部 advantage 为0，但仍进入分母。重叠的漏答窗口对同一个等待动作的局部 advantage 相加。这个分支不应用 `unit.weight`、窗口数或等待时长的归一化，也不应用等待聚合系数 tn。因此，仅局部 advantage 组的正确等待使用 +1；tn 仍参与四组共同的终局评分。它刻意保留长回复及大量等待对 token 平均的影响。

`unit_weighted` 对回复内部 token 平均，对每个参考窗口的等待决策内部平均，再使用既有事件及等待权重。事件单元权重为1/N；窗口外等待单元的总权重为 `tn * 正确等待秒数 / D`，与完整方案现有公式一致。

为保证“仅修改聚合”采用相同的动作覆盖，聚合结构只依赖回复边界、原生等待状态和参考窗口。每个窗口均建立等待单元，成功窗口的局部 advantage 为0；剩余未归属回复或等待的采样控制动作按秒建立中性单元，权重为1/N。这些中性单元在组相对模式使用共享 advantage。完整方案加入这些零优势单元后的 loss、梯度保持数值一致。

当前 native 回放概率、PPO clip=0.2、采样温度和媒体输入流程沿用原配置。每题内部平均候选，再平均实际 QA batch，候选并行与尾部累计保持同一目标。

## 相同采样预算的运行方法

先配置并显式启动本地 Judge，服务步骤见 [train README](../README.md)。四组共用同一份 prepared 数据；prepare 只做一次。

```bash
train/.venv/bin/python -m train \
  --config train/configs/minicpmo_full_grpo_confusion_judge_dp4.json prepare \
  --output train/prepared/grpo_ablation_shared
```

下面命令通过现有 grpo 入口前台运行。每次选择一个配置和全新的外部权重目录。

```bash
# 普通 GRPO
CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train --config train/configs/minicpmo_full_grpo_baseline_dp4.json grpo \
  --prepared train/prepared/grpo_ablation_shared \
  --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/grpo_baseline

# 仅局部 advantage
CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train --config train/configs/minicpmo_full_grpo_local_advantage_dp4.json grpo \
  --prepared train/prepared/grpo_ablation_shared \
  --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/grpo_local_advantage

# 仅修改聚合
CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train --config train/configs/minicpmo_full_grpo_unit_aggregation_dp4.json grpo \
  --prepared train/prepared/grpo_ablation_shared \
  --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/grpo_unit_aggregation

# 完整方案
CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train --config train/configs/minicpmo_full_grpo_confusion_judge_dp4.json grpo \
  --prepared train/prepared/grpo_ablation_shared \
  --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/grpo_local_complete
```

正式对照使用相同 epoch 和每题候选数，运行至采样预算结束。`--max-steps` 可用于独立 smoke test；固定更新次数提前停止会改变正式对照的采样预算。共同 seed 规则绑定 epoch、QA 顺序位置和候选编号；策略更新后各组生成内容会自然不同。

每组输出保留 run_config、metrics、轨迹及最终报告。采样/Judge 失败按原规则跳过，记录原因。group-relative 组通常会因零方差获得更少的实际更新，需同时报告成本与效果。

## 结果与恢复

- `optimization` 记录每条轨迹实际使用的 advantage_mode、loss_aggregation，组相对模式另存 trajectory_advantage；`reward.local_credit` 保留局部奖励诊断。
- `sampled_qa_groups`、`rollout_attempts`、`completed_rollouts`、`judge_failed_groups`、`scored_qa_groups`、`sampling_wall_seconds`、`updates_this_run` 和 `update_skip_rate` 描述本次进程的采样成本与有效更新。恢复运行时这些运行级计数重新开始，比较完整实验需汇总各次启动的记录。
- TensorBoard 使用 `group_relative_advantage`、`unit_weighted_aggregation` 两个数值标识实验组合。token_mean 模式报告 `local_loss_token_mean`；unit_weighted 模式报告各类实际策略 loss。训练奖励与实际策略 loss 分别记录。
- checkpoint 身份包含解析后的两个选择项、目标版本及相关代码摘要，跨实验恢复优化器状态会被拒绝。

本次交付准备了代码、配置及 CPU 测试。GPU rollout 和实际训练由显式启动命令执行。
