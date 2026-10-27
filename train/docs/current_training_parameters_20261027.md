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
