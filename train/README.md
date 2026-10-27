# MiniCPM-o-4.5 全参数流式 SFT / GRPO

入口为 `python -m train`。支持全参 SFT 和全参 GRPO：训练 LLM、视觉/音频编码器及连接层，不加载 TTS。LoRA 训练已移除。历史 benchmark adapter 的推理加载保留在隔离的兼容模块，不是训练入口。

## 模型、数据和环境

模型默认位于 `/remote-home/yushengliao/speech/jimodels/openmodels/MiniCPM-o-4.5`。
数据为 `/remote-home/yushengliao/speech/jimodels/datasets/AOI-Dataset/all_qa.json`，`videos_dir` 指向 AOI-Dataset 根目录。数据下载、更新和媒体检查见[仓库README](../README.md)，训练统计见[数据报告](docs/aoi_dataset_statistics.md)。

```bash
python3.11 -m venv train/.venv
train/.venv/bin/pip install -r train/requirements.txt
train/.venv/bin/python -m train --help
```

需要 FFmpeg/ffprobe 和兼容 PyTorch/CUDA 的 FlashAttention 2。SFT配置为 [minicpmo_full_sft.json](configs/minicpmo_full_sft.json)，GRPO配置为 [minicpmo_full_grpo.json](configs/minicpmo_full_grpo.json)，相对路径按仓库根目录解析。两种配置的共同默认值为 BF16、FlashAttention 2、300秒发声前缀、32768 token、每秒chunk预算20、`supervise_silence=true`（监督标签仅SFT使用）、学习率1e-6、2轮、梯度累积1、seed=42。ZeRO-3配置为 [zero3_cpu.json](configs/zero3_cpu.json)，FP32 master/Adam状态卸载CPU。

## 原子 task_start 与连续多回复目标

训练协议为 **`task_start_token_sft_v1`**。`<|task_start|>` 在内存中注册为一个特殊 token，保持原有token ID不变；原始MiniCPM词表由151748扩为151749，新增ID为151748。ID由tokenizer动态取得，不在代码中硬编码。
输入embedding与lm_head一同扩表，新行分别用原权重均值初始化；它们随完整模型参与训练。加载已有新协议模型时只校验，不重复注册或重置。磁盘上的原始模型文件不改写。

```text
<|speak|>我来查找替换配件。<|task_start|>查询视频中配件的可售替换选项。<|turn_eos|>
```

每个Agent回复内分隔符只出现一次，后面是非空instruction，不生成JSON。不执行工具、不输入工具返回和完成回复，也不使用Agent负样本。RTD/PR仅监督自然语言回复。

每条QA是一段连续轨迹，覆盖全部可评分response。先输入 `[t-1,t)` 媒体，再监督时刻t的输出；后续回复能看到之前按时输入的真实回复，未来回复不会提前输入。每个目标都满足问题、证据、trigger及原窗口的时间约束；与前一回复跨chunk冲突时只在原窗口内顺延。输入结束于最后一个目标完成。

`supervise_silence=true`：问题已输入、没有正在说话且处于所有参考窗口之外时监督listen，包含回复之间的等待。窗口内非目标位置、问题前和输入媒体均mask；不添加额外静默尾段。三组loss分别归一化：decision（含原子分隔符）、content、instruction。没有独立的listen/speak类别权重，标注遗漏和类别不平衡仍可能造成过度沉默。

任一目标无法排入原窗口、回复超出视频、超过前缀/token上限时整条QA排除并写原因，不保留残缺轨迹，不静默截断。长序列尚需实际媒体编码和GPU显存验证。

## CPU审计与重新prepare

新协议与旧普通文本、旧Agent token及旧协议checkpoint不自动兼容。旧prepared必须在新目录重新生成；完整轨迹结构仍为 `qa_trajectory_v1`，manifest协议已升级。不要覆盖旧数据或续训旧优化器状态。

```bash
train/.venv/bin/python -m train audit \
  --output train/docs/task_start_token_audit.json

train/.venv/bin/python -m train prepare \
  --output train/prepared/full_sft_task_token --max-samples 6000
```

审计只加载真实tokenizer，使用metadata或ffprobe检查视频时长，不加载模型权重或解码媒体。`text_only_tokens_lower_bound`不含媒体embedding，不能证明完整序列不超32768。
本次[原子token审计](docs/atomic_task_start_audit_20261027.json)：5000 QA / 10727 response 中4977 QA / 10636 response通过，20条QA回复越过视频末尾、3条QA发生无法在窗口内解决的冲突，整条排除23 QA / 91 response。

prepare按UUID视频划分，默认验证视频比例20%。`--max-samples`按QA计数，默认仍为12；验证采样上限为 `round(max_samples * validation_fraction)`。本数据使用6000这个上限可容纳3996条训练、1004条验证候选，不会补足或重复到6000条。正式编码仍会剔除不可行轨迹。
本机已生成 `train/prepared/full_sft_task_token_20261027`，包含全部候选，未启动训练。

## Profile与训练

下面GPU编号仅为示例，按空闲卡显式配置。命令默认前台执行，不自动启动后台任务。

```bash
CUDA_VISIBLE_DEVICES=0 train/.venv/bin/python -m train profile \
  --prepared train/prepared/full_sft_task_token --output train/outputs/full_sft_profile

CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train run --prepared train/prepared/full_sft_task_token --output train/outputs/full_sft
```

先对代表性长短轨迹profile。默认profile保存checkpoint；若仅检查梯度/显存，在复制的配置中显式设置 `save_profile_checkpoint=false`。`--max-steps`限制优化器更新次数，`--sample-index`可选择prepared中存在的QA。

多卡分片模型、梯度和优化器，各rank处理相同QA以保持原生条件媒体路径的ZeRO collective顺序；逻辑batch不随卡数增加，不是独立样本数据并行。CPU状态约百GiB量级，另需激活、媒体缓存及保存空间；旧2k–8k短程显存记录不能作为当前32k上界。日志记录全部目标的实际发声/结束时间、序列长度、监督token及排除原因。

## 全参 GRPO

GRPO沿用相同的原子Agent协议和完整QA，需用GRPO配置单独prepare，保证**每个**可评分窗口的结尾都在rollout上限内。rollout从视频起点到 `min(max_prefix_seconds, floor(video_duration))`；若任一窗口超出实际视频长度，运行时整条排除。它不会在最后一次参考回复后立即结束，因此也会惩罚尾部额外开口。本机已生成 `train/prepared/full_grpo_task_token_20261027`：3996训练QA、1004验证QA，prepare阶段未排除记录；实际媒体长度和token上限仍需运行时检查。`supervise_silence`控制SFT标签；GRPO的沉默由采样动作和时间奖励学习，不另加silence交叉熵。

推荐先做SFT，再复制GRPO配置，将 `model_path` 指向新协议SFT完整导出目录；加载导出时保留训练过的token行。**默认GRPO配置仍指向原始基座**，没有自动串联SFT，也不会自动找到最新checkpoint。不同阶段不续用SFT优化器状态。

```bash
# --config必须位于子命令之前；复制配置后可改model_path为自己的SFT导出。
train/.venv/bin/python -m train --config train/configs/minicpmo_full_grpo.json prepare \
  --output train/prepared/full_grpo_task_token --max-samples 6000

CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train --config train/configs/minicpmo_full_grpo.json profile \
  --prepared train/prepared/full_grpo_task_token --output train/outputs/full_grpo_profile

CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train --config train/configs/minicpmo_full_grpo.json grpo \
  --prepared train/prepared/full_grpo_task_token --output train/outputs/full_grpo
```

每条QA默认采样4条完整轨迹；逐秒先输入音视频，再由原生接口采样listen、speak、回复与可选instruction。不输入参考回复，不执行工具，各rank同步同一采样结果。温度固定1，无top-k/top-p截断、重复惩罚或listen缩放，以保持采样和回放的概率一致。

奖励是**本地规则，不是LLM judge**：RTD/PR为时间F1×0.5＋匹配回复的字符二元组F1×0.5；Agent为时间F1×0.4＋回复相似度×0.4＋task_start instruction匹配×0.2，非法调用默认每次扣0.1。内容和调用只在一对一时间命中的窗口内得分。可通过现有 `reward_weights` 配置覆盖权重。该相似度不能可靠判断语义正确性，尤其易惩罚正确改写；开始大规模RL前应检查奖励是否符合目标。

组内奖励标准化得到优势，采用clip=0.2的策略损失，`kl_beta=0`，不加载冻结参考模型。组内奖励相同时整组跳过、不推进优化步。采样后按原生feed边界做可微KV回放，先后验证无梯度和带梯度log-prob；最大误差超过0.15会报错，避免使用不一致的概率更新。默认启用激活CPU卸载，保存张量预算128 GiB，额外CPU优化器、媒体缓存等不计入该预算。

输出额外包含 `trajectories.rankN.jsonl`（采样动作、事件、奖励），metrics含奖励方差、零方差跳过、回放误差及策略比值。GRPO目前没有独立验证集rollout指标循环；训练奖励不等于held-out benchmark成绩。profile只验证选中QA的一组候选，可能因零方差而没有实际优化更新；看metrics确认。完整模型使用同一个export命令导出。

## 保存、恢复、导出和评测

输出包含 `run_config.json`、`metrics.rankN.jsonl`、`memory_report.rankN.json`、`checkpoint-final/`。配置可设置 `save_every_steps`，默认结束保存。恢复要求同配置、prepared与卡数，且使用新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train run --prepared train/prepared/full_sft_task_token \
  --resume train/outputs/full_sft/checkpoint-final \
  --output train/outputs/full_sft_resumed --max-steps 100

train/.venv/bin/python -m train export \
  --checkpoint train/outputs/full_sft/checkpoint-final --output train/outputs/full_sft_export
```

export在CPU合并ZeRO状态，先按checkpoint tokenizer扩表，再严格加载完整权重。产物包含新增输入/输出行、扩展tokenizer、原生runtime/processor和 `full_training_model.json`。需要足够主存和磁盘，不会下载或修改原模型。

benchmark Realtime支持新协议完整模型；检测到该metadata后，在生成有效配置和提示词之前选择 `task-text` 与 `task-start` schema。从原始generated IDs恢复分隔符，避免原生 `skip_special_tokens=True` 把它删除；普通基座和历史adapter仍沿原路径。评测命令见[benchmark README](../benchmark/README.md)，把 `--model-path` 指向完整导出目录，使用新的结果目录。Probe也会校验新模型并从原始ID保留分隔符；建议使用 `reply-task-text`、`task-specific` 的现有Probe设置。没有自动启动评测。

## 工程层与检查

- `src/application`：CLI、prepare、audit、显存估算。
- `src/domain`：只读Agent适配、因果记录、跨chunk目标、整条轨迹调度、GRPO优势/裁剪损失与规则奖励。
- `src/models`：模型加载、全参设置、原生可微prefill、GRPO采样/回放与损失图。
- `src/training`：全参SFT/GRPO循环、SFT验证、停止与恢复。
- `src/infrastructure`：配置、原子词表、媒体、ZeRO、分布式、checkpoint。

阅读顺序：CLI → prepare/records → targets/trajectory → loading/streaming → graph → training/full → checkpoints。模块和关键函数有职责、输入输出及状态释放注释。`train.src.agent`、`train.src.checkpoints`保留当前公共接口；`train.src.token_rows.load_adapter`仅供历史benchmark推理，使用旧adapter时才需要另装PEFT。

```bash
train/.venv/bin/python -m unittest discover -s train/tests
```

CPU检查覆盖原子注册、旧ID保持、新行梯度、微型模型保存加载、全参媒体梯度、连续回复与沉默mask、旧协议拒绝及checkpoint游标/RNG。GRPO检查还覆盖原生动作概率、裁剪损失、规则奖励、原子分隔符回放梯度与CPU激活卸载。DeepSpeed CUDA集成检查仍需显式启用 `RUN_ZERO_CHECKPOINT_TEST=1`。当前没有执行真实权重的GPU训练、ZeRO恢复或完整大模型导出。
