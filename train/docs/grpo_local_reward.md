# 逐轮局部奖励与策略更新

## 可选冻结参考 KL

`kl_beta` 默认为 0，原训练行为不变。非零时必须提供 `kl_reference_model_path`，指向仓库外的完整 SFT 导出；示例配置 `train/configs/minicpmo_full_grpo_tp_full_credit_kl_dp4.json` 中 0.01 只是待校准的实验值，并非推荐结果。每个 rank 在 CPU 上独立加载一份冻结参考模型，以 SDPA 和原生分段 KV 回放同一条音视频及采样动作；不使用 Judge 代替参考策略。参考策略、tokenizer、动作协议及路径进入身份检查，变更时不得恢复旧优化器。

对采样动作计算 `d = log p_ref(a|h) - log p_current(a|h)`，非负 K3 样本项为 `exp(d) - d - 1`；总损失为原局部策略损失加 `kl_beta * KL`。它是 rollout 策略下的采样代理，不是对完整词表的精确 KL。问题前动作不参与；每条回复先对其动作求平均，每个空闲/控制秒各为一个单元；有回复和等待两类时，分别平均后各占一半，只有一类时该类占全部。数值异常和不匹配会终止训练，不会静默截断或回退。

TensorBoard/metrics 的 `kl_surrogate` 是未乘 β 的聚合值，`kl_contribution` 是实际加到 loss 的数值。参考 CPU 回放每 rank 一份完整模型及 KV，上下文越长内存和耗时越高；当前只做 CPU 单元验证，真实 MiniCPM 四卡和 CPU 参考联动需在空闲资源上进行 profile，不能据此声称性能或显存已验证。参考 CPU SDPA 和策略 GPU FlashAttention 计算精度不同，即使相同权重也可能产生非零数值偏差，需在 profile 中先测同权重基线。

当前 LLM GRPO 模板使用 `reward_mode=local_turn_judge`，在现有完整视频采样、Judge、回放及 PPO 裁剪流程中，采用零基线局部 advantage。默认组合 `local + unit_weighted` 中候选独立计算信用，不做组内中心化。其他组合见 [四组消融指引](grpo_ablations.md)。配置明确保留 `legacy` 和 `confusion_judge` 供历史结果与消融使用。

## 奖励和归因

一次原生新开口到结束形成回复单元，跨秒正文与 Agent instruction 属于同一单元。记录所有采样动作及其回复 ID，强制 token 不参与策略项。同一秒结束并开始新回复时按原生结束 token 分开解析；无正文的尝试仍计 FP。

沿用 benchmark 最早截止窗口的一对一时间匹配，再对命中回复评分。RTD/PR 的 `q=response_score/2`，Agent 的 `q=(response_score+call_score)/4`。无正文的回复质量记零，缺失或非法调用的调用质量记零。

| 行为 | 局部 advantage |
| --- | ---: |
| 时间匹配的回复 | `tp * (2*q - 1)` |
| 未匹配回复或空开口 | `-fp` |
| 尝试调用但格式非法 | 额外 `-invalid_call_penalty` |
| 最终漏答的窗口 | `-fn` |
| 成功窗口内的等待 | 0 |
| 窗口外正确等待 | +1，聚合时乘 `tn` |

模板权重为 `tp=1, fp=0.5, fn=1, tn=0.1, invalid_call_penalty=0.1`。reward 已包含权重，策略项中不重复乘这些权重。Agent 没有调用时调用质量为零；额外格式惩罚只针对尝试过却无效的调用。

FN 优先归给该窗口内实际采样的空闲等待决策，将窗口作为一个单元内部平均。窗口成功匹配时，之前的窗口内等待无需扣分。若窗口没有可归因等待，将 FN 叠加到占用回复的 advantage，多条回复按窗口内占用时刻的比例分摊；同秒多回复均分该秒占用。这是用户选定的整条回复归因，会影响该回复的正确内容和开口。例如正确回复的 +1 与占用下一窗口的 -1 可抵消。

若既没有等待动作，也没有可归因回复，FN 保留计分并记录 `local_unassigned_fn`，该项梯度为零。问题输入前不参与局部奖励。Judge 失败、非法评分、超出视频或 token 限制等情况沿用整组失败处理。

## 聚合

每个回复内部平均其实际采样 token 的 PPO loss。每个等待决策内部先平均采样动作，再在漏答窗口内部平均。设参考窗口数为 N：

```
L_event = (sum(reply_unit_loss) + sum(idle_FN_window_loss)) / N
L_wait  = sum(correct_idle_wait_decision_loss) / D
L_QA    = L_event + tn * L_wait
```

D 为问题输入后、窗口外、进入本秒时处于空闲状态且存在采样动作的决策数，包含这些时刻选择开口的情况；正在续说不进入分母。D 为零时等待项为零。成功窗口等待和剩余采样控制动作也保留结构单元，默认局部优势为0；它们用于聚合消融的共同动作覆盖。占用型 FN 已合入回复 advantage，仅计算一次。回复和 FP 不按生成次数重新归一化，额外 FP 会增加惩罚。

各 QA 内对所有候选平均，再按真实 QA batch 平均。候选并行和梯度累计保持同一目标，padding 仅补齐累计边界。整组总分相同仍可更新；仅局部信号全零才跳过。局部目标不是 F1/scored-F1 的精确分解，实验同时监控时间指标和内容指标。

## 文件与监控

- `trajectories.rankN.jsonl` 保存原始动作、事件和 `reward.local_credit`。每个单元包含回复/窗口 ID、来源、reward/advantage、归一化权重，以及动作数或决策秒列表。
- 回复的 `base_advantage` 与 `blocked_fn_advantage` 分开保存，实际使用其和。`blocked_windows` 记录每个 FN 的分摊比例。
- TensorBoard/metrics 的 `local_loss_reply`、`local_loss_fp`、`local_loss_fn`、`local_loss_wait`、`local_loss_control` 相加为 unit_weighted 策略 loss；token_mean 使用 `local_loss_token_mean`。
- `local_loss_blocked_fn` 是合入占用惩罚后的 clipped loss 相对原回复 loss 的变化，用于诊断，不再次加到总 loss。
- `local_idle_fn`、`local_blocked_fn`、`local_unassigned_fn`、正负单元数、等待机会数和零信号组数用于检查归因。

`minicpmo_full_grpo_judge.json` 的原有 32768 token 限额保留，另两个模板保留65536。完整视频超过相应预算时按已有规则排除，不进行截断后评分。模型、学习率、候选数、梯度累计与采样设置沿用模板值。

GRPO 原生概率回放保持模型为 `eval`，视觉编码器内置的仅训练态重算开关因此不会生效。遇到逐帧视觉激活占满 CPU 预算时，可显式配置 `grpo_vision_checkpointing=true`：仅对无状态的 `get_vision_embedding` 做非重入重算，保留 eval 输出、参数梯度和逐秒输入顺序，不重算有流式状态的音频编码器。该选项默认关闭；启用后仍须通过真实 GPU 的概率回放与反向更新预检。

若 `grpo_replay_checkpoint_segments` 已启用，而重算组内多个 decoder feed 的反向激活导致 CUDA OOM，可额外显式配置 `grpo_replay_checkpoint_feeds=true`。内层逐 feed 重算限制 decoder 激活峰值，外层分组限制需保存的 KV 边界数量；不改变原生 feed 顺序，不 detach 历史 KV。该选项同样默认关闭，需在实际轨迹上验证回放概率与完整反向更新。

原生回放及视觉重算会把嵌套输入展平为独立 Tensor 参数传给 checkpoint，再在函数内部恢复结构。PyTorch 2.6 的 checkpoint 输入保存器不会递归保存容器中的 Tensor；直接传入 KV tuple 会绕过外层 saved-tensor CPU 卸载。展平后只在闭包保留结构信息，KV 边界能进入卸载 hook，原生计算顺序与跨时间梯度不变。可减小 `grpo_replay_checkpoint_segments` 降低重算峰值，但会增加 CPU 边界存储。`activation_cpu_saved_4d_gib_max` 记录卸载的四维张量体积（包含 KV，不能视作纯 KV 精确统计）。

可显式选择 `grpo_replay_mode="full_sequence"`，将实际流式轨迹的媒体 embedding 与全部生成 token 拼成整段 causal 输入，使用 `use_cache=false` 前向。模型保持 eval，并逐 decoder layer 重算；保留跨时间梯度和原有采样动作分布、旧 log-prob、局部目标。此模式不能同时启用 native KV 的分段或逐 feed 重算选项。默认仍为 `native_kv`。

整段与逐 token 计算可能存在 BF16 数值差异，必须保留 `replay_logprob_tolerance` 检查。预检在 `replay_diagnostics.rankN.jsonl` 记录最大、均值、P95 误差及最差动作的位置、token 与新旧 log-prob。超阈值直接停止，不重新覆盖采样时的旧概率。`optimizer_parameter_max_delta` 记录优化器 FP32 主参数探针的实际变化，辅助确认一次更新真实发生。回放模式及实现摘要进入 checkpoint 身份，不能跨模式恢复优化器。

使用 README 中的服务准备、prepare、grpo 命令，选择实际配置及新的外部权重目录。Checkpoint 身份包含局部目标版本、配置和奖励/聚合代码摘要，旧目标的优化器状态无法直接续入。当前修改不自动启动服务、采样或训练。

## 临时 TP 满分模式（无 LLM Judge）

显式设置 `reward_backend="tp_full_credit"`，或使用 `train/configs/minicpmo_full_grpo_tp_full_credit_dp4.json`。默认 LLM 配置不变。此模式不需要 `judge` 字段，不读取 Judge 权重/提示词/缓存，不请求 API，不启动服务或预约 Judge GPU。四卡模板保留局部 advantage 与单元聚合，并使用已验证的 ZeRO-2 原生 KV 分组回放设置。

按既有一对一时间匹配，TP 的合成 response 分数为2；Agent 的合成 call 分数也为2，故 `q=1`、TP 基础 advantage 为 `+tp`。即使正文为空、缺调用或调用 instruction 不正确，也不降低 TP 的合成质量。尝试调用但格式非法时仍另扣 `invalid_call_penalty`；没有形成事件的空开口仍计FP。重复回复、FP、FN、TN、占用型FN、窗口内等待及损失聚合保持原规则。该模式是时间奖励消融，满分不是语义正确性的实测结果。

模板以最新 full-data SFT 导出为起点；运行前核对 model_path 和数据/视频路径。相同视频划分、seed、horizon 和记录格式的 GRPO prepared 可以复用。直接用既有 `python -m train ... profile/grpo` 入口运行，无需执行 `serve_grpo_judge.py`。自动衔接脚本要求 profile 与正式 GRPO 使用相同 backend，在此模式下跳过 Judge 的健康检查与 GPU 预约。

Checkpoint、最终报告记录 `tp_full_credit` 和独立评分器版本，不能与 LLM Judge 模式续接优化器状态。HTTP 请求计数始终为0，逻辑评分计数仅统计匹配的TP。切换回语义奖励时显式设置 `reward_backend="llm"` 并恢复 Judge 配置，使用新的输出目录；旧的冻结 runtime 不会自动获得这个新模式。
