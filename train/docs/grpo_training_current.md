# 当前细粒度 GRPO 训练说明

本文描述 2026-11-16 启动的 MiniCPM-o-4.5 实验：`fp=1.0`、LLM Judge、局部 credit 和 KL 约束。具体运行参数以[本次配置](../outputs/grpo_judge_kl003_fp1_40_20261116/grpo.json)为准；此前无 Judge、`fp=0.5` 的实验不应与此混为一谈。

## 1. 目标和训练对象

从全量数据 SFT 的 MiniCPM-o-4.5 checkpoint 出发，训练模型在流式音视频交互中判断何时等待、回复以及何时委托 Agent。当前实验更新模型参数，使用 DeepSpeed ZeRO-2 和 CPU 优化器卸载；冻结的 SFT 参考模型用于计算 KL 约束。

奖励在完整轨迹生成后依据参考回复窗口计算。模型生成时不能看到参考回复、评分窗口或未来媒体。Agent 输出沿用文本协议：`回复内容<|task_start|>委托指令`；是否包含委托由任务和模型输出决定。

## 2. 数据和采样

- 本次使用准备好的 120 条训练 QA，Agent、PR、RTD 各 40 条；另有 24 条视频不重叠的验证 QA。来源和清单见 [manifest](../prepared/grpo_kl003_120qa_20261115/manifest.json)。
- 按完整视频生成轨迹，使用流式输入：逐秒先接收该秒音视频，再进行该秒的生成决策；问题在其指定时间插入。最大序列预算为 65,536 tokens，超限或失败的轨迹按现有错误处理，不能保证所有样本都成功用于更新。
- 每个 QA 采样 4 条候选轨迹。四张训练卡采用候选并行：同一 QA 的候选分配到不同 rank，**不是**每张卡各处理 4 个独立 QA。
- 梯度累计为 4 个有效 QA 组；完整批次有 4 个 QA × 每组 4 条候选，即 16 条候选轨迹。部分尾批、失败或跳过会影响实际数量。
- 配置最多遍历 2 个 epoch，本次另设 **40 个 optimizer step 上限**。若每批均有效，40 步约处理 160 次 QA 访问，提前达到步数上限时不会遍历完两个 epoch。

## 3. 轨迹匹配和内容评分

生成完成后，将模型的回复起始时刻与参考窗口做一对一匹配，窗口两端均包含：

- **TP**：回复匹配到参考窗口；每个窗口和回复最多匹配一次。
- **FP**：回复没有匹配到窗口，包括多余回复以及无法解析的开口尝试。
- **FN**：没有匹配到回复的参考窗口。
- 正确等待：问题输入后、参考窗口外、存在等待决策且模型选择等待的时刻。

对命中的回复，由 7 号卡运行的 Qwen3.8-27B Judge 进行参考答案评分；PR/RTD 评分回复内容，Agent 分别评分回复和委托指令，合法调用和非空回复检查也会影响内容分。单项评分允许 `0, 0.5, 1, 1.5, 2`，先除以 2 得到 0–1 的质量分；Agent 将两项归一化分数取平均。Judge 使用固定提示、缓存、超时和重试机制；失败的候选组不会被当成零分继续训练。

## 4. 轨迹总 reward

设参考窗口数为 \(N\)，命中回复的归一化质量为 \(q_i\)，不合法调用数为 \(I\)，窗口外可等待时刻数为 \(S\)，其中实际等待数为 \(W\)。当前局部奖励实现记录的总分为：

\[
R=\frac{\sum_{i\in TP}(2q_i-1)}{N}
  -\frac{FN}{N}-\frac{FP}{N}-0.1\frac{I}{N}
  +0.1\frac{W}{S}.
\]

当不存在窗口外可等待时刻时，最后一项为 0。\(2q_i-1\) 将满分、中间分、零分分别映射到 +1、0、-1。FP 和 FN 各按 1.0 惩罚；多次误报分别计算。\(W/S\) 的总贡献最多 0.1，而不是每秒累计 0.1。

示例：2 个窗口，1 个满分 TP、1 个 FN、1 次 FP，正确等待比例 90%，无不合法调用，则总 reward 为 `0.5 - 0.5 - 0.5 + 0.09 = -0.41`。窗口的匹配和等待条件以代码为准，总 reward 是轨迹统计指标，**不直接作为当前所有 token 的共同 advantage**。

之前的 `tp_full_credit` 消融不调用 LLM Judge，对时间命中的回复直接赋内容满分，且当时的 `fp=0.5`；它不衡量真实回复内容质量。

## 5. 局部 credit 与聚合

当前 `grpo_advantage_mode=local`，每条轨迹按动作归属拆成局部单元：

| 单元 | 当前局部 advantage | 权重与说明 |
|---|---:|---|
| 命中窗口的回复 | `2q-1` | 权重 `1/N`；一条回复内部的采样 token 分摊该单元权重 |
| 未匹配的回复 | `-1` | 权重 `1/N`；无效调用还会额外减 0.1 |
| 已命中窗口内的等待 | `0` | 不把命中前的等待额外当 FN 惩罚 |
| 漏答窗口内的等待 | `-1` | 权重 `1/N`，由可归属的等待决策分摊 |
| 被已有回复占满的漏答窗口 | `-1` 分摊到占用该窗口的回复 | 按占用比例分配；无法归属的窗口记录为 unassigned |
| 窗口外正确等待 | `+1` | 单元总权重 `0.1 × W/S`，内部按时刻分摊 |
| 其他结构控制 | `0` | 保留动作覆盖，不增加局部奖励 |

局部 advantage 使用零基准，**没有对四条轨迹的局部分数做组内标准化**。每个单元内的 token/时刻按代码分摊权重，`unit_weighted` 聚合让回复和等待单元不因 token 数量直接线性增权。同一采样 token 若属于多个单元，其系数可叠加。训练从问题后的有效决策范围分配 credit；问题之前的动作不纳入这些单元。

对采样动作，策略比率为 `r = exp(log π当前 - log π采样)`；使用 PPO 风格的 clipped 目标，裁剪参数 `ε=0.2`，分别处理正负 advantage。各单元的目标按权重求和，再对有效 QA 和候选轨迹做批次平均。FP 的惩罚分摊到整条回复的采样动作，**不是专门只惩罚开口 token**，这可能影响抑制误触发的效率，需要单独实验验证。

普通 GRPO 消融可以使用相同轨迹总 reward，但先对同一 QA 的 4 个候选总分做组内标准化，得到每条轨迹一个 advantage，再对其 token 聚合；与当前逐单元 credit 并非同一种分配方式。公平对比时须保持 Judge、奖励权重、数据、采样及 KL 设置一致。

## 6. KL 约束和最终损失

当前 `β=0.03`。采样动作的参考模型和当前模型对数概率之差为 `δ = log πref - log π当前`，单动作使用非负的 K3 形式 `exp(δ) - 1 - δ`。该值先按回复单元和等待/控制单元分别聚合；两类都存在时各占一半。这里是**采样动作上的 KL 代理估计**，不是完整词表精确 KL。

最终损失为：

\[
L=L_{\text{local clipped policy}} + 0.03\,L_{\text{sampled KL}}.
\]

参考模型冻结，每次在训练卡上完成参考轨迹打分后卸载回 CPU；该过程不更新参考权重。KL 在 loss 中单独计算，不包含在上面的轨迹总 reward 中。

## 7. 当前配置和运行

| 参数 | 当前值 |
|---|---|
| 模型起点 / KL 参考 | 全量数据 SFT final checkpoint |
| Judge | Qwen3.8-27B，GPU 7 |
| 训练卡 | GPU 0–3，4 个候选并行 |
| 学习率 / 权重衰减 | `1e-6` / `0.01` |
| 梯度裁剪 / PPO clip | `1.0` / `0.2` |
| 训练上限 / 保存间隔 | 40 步 / 5 步；保留最近的编号 checkpoint |
| 激活卸载预算 | 每个训练进程 160 GiB CPU 预算；不是总内存保证 |
| 温度 / 每秒回复 token 上限 | `1.0` / `20` |

当前实验的权重和 checkpoint 保存在 `~/jidata/OmniDuplex/training_outputs/grpo_judge_kl003_fp1_40_20261116/`，不在 `~/jimodels`。本次运行的[队列状态](../outputs/grpo_judge_kl003_fp1_40_20261116/status.json)、[训练日志](../outputs/grpo_judge_kl003_fp1_40_20261116/training.log)及配置位于仓库运行目录。训练和 Judge 启动需各自检查 GPU 及服务状态；不要在已有任务运行时重复执行启动命令。

## 8. 监控和解释

每个优化步记录 reward、TP/FP/FN/TN 及分项贡献、局部策略损失、KL 代理值、梯度范数、回放误差和耗时；TensorBoard 日志路径记录于训练输出目录的 `tensorboard_run.json`。每个 rank 的 `trajectories.rank*.jsonl` 保存采样轨迹及其 reward、局部 credit；`metrics.rank*.jsonl` 保存优化步指标。

实验入口还启用了 `action_grad_policy_rms_norm`、`action_grad_kl_rms_norm`、`action_grad_kl_policy_ratio` 和 `action_grad_cosine`，比较策略项与加权 KL 项对**采样动作对数概率**的梯度。这不是全模型参数梯度比例，不宜直接用来推断参数空间的影响。

不同 step 的 QA 不同，单步 reward 或 KL 波动不能直接证明模型性能升降。应在固定评测集合上联合检查 TP、FP、FN、Precision、Recall、F1 和内容质量，并比较基座、SFT 与 GRPO。评测时参考窗口用于事后评分，不输入模型。

## 9. 实现入口

- 奖励和归属：[local_reward.py](../src/domain/local_reward.py)、[confusion_reward.py](../src/domain/confusion_reward.py)
- 策略损失：[local_policy.py](../src/models/local_policy.py)、[graph.py](../src/models/graph.py)
- KL 与冻结参考：[reference_kl.py](../src/models/reference_kl.py)
- 训练批次和保存：[grpo_judged.py](../src/training/grpo_judged.py)
- Judge：[judge.py](../src/infrastructure/judge.py)、[评分提示](../../benchmark/configs/prompts/judge.txt)
- 梯度监控：[grpo_gradient_monitor.py](../src/infrastructure/grpo_gradient_monitor.py)
