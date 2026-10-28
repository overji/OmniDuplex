# 当前全参SFT参数与验证边界（2026-10-27）

当前支持全参SFT及全参GRPO；LoRA训练和配置已删除。下表记录SFT配置，GRPO差异见后文。完整操作见 [train README](../README.md)，JEV的独立全参分类器/Thinker训练见 [JEV README](../../JEV_train/README.md)。

| 参数 | train当前值 |
|---|---|
| 模型 | MiniCPM-o-4.5，非TTS LLM/媒体/连接层全参训练 |
| 配置 | `train/configs/minicpmo_full_sft.json` |
| 数据 | AOI-Dataset，5000 QA / 10727可评分response |
| 训练目标 | 每条QA的一段完整多回复轨迹，历史回复按真实时间进入上下文 |
| 协议 | `task_start_token_sft_v1`，`qa_trajectory_v1`记录结构 |
| 特殊token | task_start为原子特殊token；原基座151748→151749，新增ID动态取得 |
| 词表训练 | embedding/lm_head扩表，新行均值初始化并参与全参更新；已有训练行不重置 |
| 输入 | 每秒先输入媒体再输出，保留完整历史，不静默截断 |
| 上限 | 首次及后续发声不超过300秒；总序列32768 token；chunk预算20 |
| silence | true，监督问题后、所有窗口外、非说话时刻的listen |
| 优化器 | AdamW，lr=1e-6，weight_decay=.01，max_grad_norm=1 |
| epoch / 累积 | 2 / 1 |
| 精度 | BF16、FlashAttention 2、支持模块的gradient checkpointing |
| 状态分片 | ZeRO-3，CPU optimizer offload；各rank相同QA，不是独立样本DP |
| 保存 | 默认结束保存完整checkpoint，可配置save_every_steps |
| Agent范围 | 只读适配task_start(instruction)；不使用工具返回、完成回复或Agent负样本 |

数据原文件SHA256：`3809db9b78f1f3cadee16cafc13b7fd0133ee17ee84e451428ac87f376014d30`。

## GRPO参数与阶段衔接

`train/configs/minicpmo_full_grpo.json`保留同样的300秒/32768 token、全参、原子Agent协议、lr=1e-6、2轮和ZeRO-3 CPU卸载。额外参数：group_size=4、temperature=1、clip_epsilon=0.2、kl_beta=0、replay_logprob_tolerance=0.15、激活CPU卸载=true、保存张量预算128 GiB。

需单独prepare，要求所有参考窗口完整落在rollout时长内；运行时还检查实际视频长度。参考回复只计算奖励，不作为rollout输入。时间F1、字符二元组回复相似度与Agent调用instruction匹配组成规则奖励，无LLM judge。组内零方差不更新，采样/回放概率不一致时报错。

建议SFT→完整导出→复制GRPO配置并设置model_path→单独prepare→profile→grpo。默认配置仍从原始基座开始，没有自动执行两阶段。四卡仍为同QA的ZeRO分片，不是四路独立样本DP。JEV为独立全参SFT，不改变成GRPO。

## 真实tokenizer CPU审计

[新协议完整审计](atomic_task_start_audit_20261027.json)：4977 QA / 10636 response通过；23 QA / 91 response整条排除，原因20条回复越过视频末尾、3条回复无法在原窗口内排下。新token减少了部分chunk/token开销，本次通过数没有变化。
这是调度、真实tokenizer和metadata时长审计，**没有实际媒体embedding长度或GPU显存验证**。

已生成 `train/prepared/full_sft_task_token_20261027`，时间/素材条件候选3996训练QA、1004验证QA；实际编码仍会剔除不可行轨迹。prepare默认12条训练，完整候选用 `--max-samples 6000` 作为上限，以容纳按视频划分后的全部验证QA。旧prepared和旧优化器checkpoint协议不兼容，不能直接续训。

## 仍需验证的风险

- 完整媒体、system、问题、控制符和全部回复都计token。32768只是上限，不保证编码后全部样本合格，也不保证四卡不OOM。
- listen与speak共用decision权重；标注可能遗漏合理回复，长前缀负秒可能导致过度沉默。
- 数据仍有时间边界问题。整条排除保持明确，不能为提高覆盖率擅自扩大窗口、截断回复或丢掉后续目标。
- teacher-forced历史与推理时模型自己的错误历史不同，需用完整流式推理评价，不能用训练loss证明闭环效果。
- 新词表和输出层都必须随模型保存；旧解码器可能过滤特殊token，新评测使用原始ID识别，不能只看原生text字段。
- ZeRO四卡用于分摊状态，CPU优化器需要约百GiB主存，加上激活、缓存、通信和checkpoint空间；耗时不能直接除以4。
- JEV默认仍是32秒/8192 token、lr=1e-5、累积4，与train配置不同；其完整全参训练还没有真实GPU验证。

## 耗时与实测边界

此前32秒、单response、LoRA或GRPO的预算不代表当前多response全参SFT。当前不存在可用的四卡稳定吞吐实测，不给新的确定ETA。开始前应对三种交互及长短轨迹测20–50个逻辑步骤，再以真实更新数、验证/预处理及保存开销估算。

本次只执行CPU检查、微型模型权重/梯度/导出重载与真实tokenizer审计。ZeRO合并的CPU单测使用state dict替身；没有启动真实大模型GPU训练、ZeRO恢复或完整大模型导出。

## 后续四卡独立QA加速版本

新增 `minicpmo_full_sft_dp.json`（旧默认SFT配置不变），使用DeepSpeed ZeRO-2、单次可微前向、显式长度分桶与GPU保留激活预算。当前部署卡组为0–3，benchmark为6–7；本次不再使用4、5。每卡不同QA，有效batch=4，两轮3996 QA对应1998次更新；lr仍1e-6、前缀300秒、32768 tokens、supervise_silence=true和全部response保持。

实际运行配置在 `train/outputs/full_sft_dp4_wandb_20261027.config.json`。每50步保存并保留最近1个编号checkpoint，最终checkpoint另保留；STOP文件可在步边界安全退出。旧run在42步结束，没有checkpoint，因此新run是从基座和新优化器开始，不能把旧曲线与新曲线视作连续续训。

框架评估和GPU小模型一致性检查见 [框架复用报告](training_framework_review_20261027.md)。真实两卡旧模式QA72测得613.93秒、训练期峰值分配14.60 GiB/rank。新两卡方案已通过真实前向/反向，但更新时还需要16.81 GiB临时FP32分区，因4/5卡额外显存占用而OOM；这不是成功的性能结果。四卡该临时分区约8.4 GiB，正式运行效果以新run日志为准，不提前承诺加速倍数。


## 四卡 batch=16 重启（2026-10-27）

用户要求将有效batch从4提高到16。独立QA配置的 `gradient_accumulation_steps=4`，每卡micro batch仍为1；GPU为0–3，benchmark继续使用6–7。其余训练超参数沿用之前run，包括lr=1e-6、2 epochs、300秒/32768 tokens、全部response、supervise_silence=true、ZeRO-2与CPU优化器/激活卸载。3996训练候选对应每轮250次更新，两轮500次；实际QA排除另见metrics。

旧run通过STOP在优化器边界退出，保留 `full_sft_dp4_wandb_20261027/checkpoint-final` 和原有W&B记录。新run从原始基座重启，不是跨batch的精确断点恢复。新配置：`train/outputs/full_sft_dp4_bs16_wandb_20261027.config.json`。日志：`train/outputs/full_sft_dp4_bs16_wandb_20261027.log`。输出写入 `/remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft_dp4_bs16_wandb_20261027`，仓库outputs下有同名链接；本地盘当时只剩218 GiB，而checkpoint轮换需同时容纳约两份117 GiB状态。

W&B：https://forge.coreweave.com/wandb/overji-shanghai-jiao-tong-university/OmniDuplex-SFT/runs/k1xdi77z 。日志的step是优化器更新次数，同一步内有4个microbatch；loss仍逐microbatch记录，不能当作完整16条QA的平均loss。`global_batch_size=16`是配置上限，末尾padding或运行时排除会降低实际有效数，`valid_qa`记录当前microbatch有效QA数。

实现修复：允许独立QA累积，空padding批次参与零损失反向以推进完整累积边界；本地DeepSpeed CPU卸载累积路径写回中间梯度缓存，不修改site-packages。两卡、累积4的梯度与两次参数更新均通过参考实现比较，最大参数误差1.49e-8；累积1回归最大误差1.64e-8。CPU相关测试16项，其中1项条件跳过。

### 仍需关注的训练风险

- 大batch在相同epoch下减少更新次数（1998→500），lr=1e-6保持保守，不保证2 epochs效果足够；不要自动线性放大学习率。
- 目前固定lr，没有warmup或衰减。新task_start词表行和全部媒体编码器一起训练，需结合验证性能检查是否学习不足或遗忘。
- decision/content/call各组先独立求平均再以1/1/1相加，并非按总token平均；silence与speak仍共用decision组，长静默段可能使开口监督不足。需同时检查负侧正确率、正侧召回和Agent调用内容，不能只看总loss。
- 300秒和32768 tokens仍是硬边界；不可行轨迹整条排除，未静默截断。prepared候选数量不代表实际全部参与训练。
- 训练验证按视频隔离，但目前仅epoch结束评估验证loss；缺少中途when-to-speak及调用语义指标，loss下降不能替代benchmark。
- 累积主要增加每次更新的样本覆盖，通常不显著增加激活显存，也不意味着总训练快4倍；CPU卸载、长序列和跨卡长度不齐仍影响吞吐。


## 后续run：尾部沉默监督

`train/configs/minicpmo_full_sft_dp.json` 现加入 `post_response_silence_seconds=10`。仅影响后续使用该配置的新run；正在运行的batch16 run配置未改、未重启。新增监督起点为所有目标回复结束和所有已标注闭区间窗口结束之后的下一整数秒，最多10秒，按实际视频长度/300秒前缀上限缩短。若token不足，仅撤回新增尾部的完整秒，不舍弃原有response。实际尾部秒数记入 `tail_silence_seconds`。没有把整段长视频尾部都强行标为静默，未修改GRPO。边界假设仍是标注窗口之外没有漏标的合法回复；收益与假负风险尚未经消融验证。


## 训练后自动 Realtime benchmark（2026-10-28）

已为tail10 run安排自动衔接：完整500次更新、4个rank保存完成且训练进程退出后，CPU导出到外部openmodels，再用0–3号卡四实例运行AOI-Bench全量2107条QA的Realtime评测。不运行Probe，不复用基座预测，不调用LLM judge。工作流状态为 `benchmark/results/sft_dp4_bs16_tail10_20261028/after_training_status.json`，结果目录为同目录的 `suite/realtime/minicpmo/`。原有6–7号卡队列不变。所有权重必须实际存放在 `~/jimodels` 以外。


## 后续 GRPO Judge 配置（2026-10-28，尚未启动）

新增 `train/configs/minicpmo_full_grpo_judge.json`：使用SFT导出模型、4候选/QA、4个有效QA组累计（16条轨迹/完整更新）、ZeRO-3、lr=1e-6、2 epochs。仅时间命中的回复/调用使用Qwen3.8-27B文本参考Judge，评分0/.5/1/1.5/2除2；负时间秒新开口行为提供独立沉默分。RTD/PR权重T/C/S=.4/.4/.2，Agent为T/C/A/S=.3/.3/.2/.2，非法调用每次扣.1。

旧规则配置不变，新Judge路径只由rank0请求并广播，失败/零方差不计有效QA，末尾部分batch按真实组数归一化。模型/提示/奖励实现身份参与恢复校验，缓存有完整输入指纹；不截断评分输入、不把API失败当0分。

启动准备脚本为 `train/scripts/serve_grpo_judge.py`，默认只打印计划，显式 `--start` 才使用默认GPU7；卡占用超过2 GiB会拒绝，不停止其他进程。GPU6单卡benchmark迁移须等用户下令再执行。当前SFT、既有benchmark和训练后Realtime衔接均未修改。

离线验证：72项训练测试中71项通过，1项显式CUDA集成测试跳过；静态检查通过。未启动Judge或GRPO，没有进行真实GPU加载/训练验证。
