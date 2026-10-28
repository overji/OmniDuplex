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
  --prepared train/prepared/full_sft_task_token --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft_profile

CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train run --prepared train/prepared/full_sft_task_token --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft
```

先对代表性长短轨迹profile。默认profile保存checkpoint；若仅检查梯度/显存，在复制的配置中显式设置 `save_profile_checkpoint=false`。`--max-steps`限制优化器更新次数，`--sample-index`可选择prepared中存在的QA。

多卡分片模型、梯度和优化器，各rank处理相同QA以保持原生条件媒体路径的ZeRO collective顺序；逻辑batch不随卡数增加，不是独立样本数据并行。CPU状态约百GiB量级，另需激活、媒体缓存及保存空间；旧2k–8k短程显存记录不能作为当前32k上界。日志记录全部目标的实际发声/结束时间、序列长度、监督token及排除原因。

全参SFT也可在复制的配置中显式启用 `sft_activation_offload=true`，通过 `sft_activation_cpu_budget_gib` 设置每个rank保存张量的CPU预算（启用时默认128 GiB）。默认关闭，不改变目标、loss或样本长度；只转移反向传播保存的张量，会增加CPU内存和传输开销。metrics记录卸载量和参数缓存命中。

## 回复结束后的沉默监督

后续独立QA SFT配置 `minicpmo_full_sft_dp.json` 启用 `post_response_silence_seconds=10`：保留全部response及其原窗口，在所有回复生成完毕、且所有闭区间回复窗口结束之后，再逐秒输入最多10秒音视频，监督 `<|listen|>`。例如最后回复结束于100秒、窗口结束于103秒，则104–113秒为新增沉默监督；101–103秒仍按原窗口规则mask。

新增尾部受视频实际长度和 `max_prefix_seconds` 限制；token预算不足时逐秒缩短新增尾部，保留完整原始response，绝不保留半秒输入或为腾出尾部空间而丢弃response。原始必要序列本身超限仍整条排除。`metrics`/W&B中的 `tail_silence_seconds` 为该rank该QA实际加入的尾部沉默秒数；padding为0。

该参数要求 `full_sft` 和 `supervise_silence=true`。其他配置可显式添加同一参数；缺省或设0时保持原行为，不延长时间线。没有修改GRPO的完整rollout规则。尾部监督沿用“标注窗口外不应回复”的数据假设：若尾部有遗漏标注的真实触发，会产生假负标签，需抽查数据。10秒为有限范围的初始选择，并非经过消融验证的最优值。

当前运行的 `full_sft_dp4_bs16_wandb_20261027` 使用其已保存配置，不含该参数，因此本次没有重启或改变现有训练。下一次用上述独立QA配置启动时生效；已有run不能直接改配置做严格断点恢复。

## W&B loss曲线

可选安装：`pip install -r train/requirements-wandb.txt`。在运行环境中设置 `WANDB_API_KEY`，不要写入JSON或Git。在复制的训练配置中加入：

```json
"wandb": {"project": "OmniDuplex-SFT", "name": "minicpmo-full-sft", "mode": "online"}
```

可添加 `entity` 指定团队，省略时沿用账号默认工作区。仅rank0上传 `train/loss`、学习率、epoch、耗时、序列长度、显存和卸载量，以及epoch结束时的 `validation/loss`。横轴为 `train/step`，在W&B中调节平滑程度。不会上传QA、生成文本或模型权重。链接保存在输出目录 `wandb_run.json`，JSONL日志继续保留。默认不启用W&B，支持offline/disabled；初始化失败时所有rank共同退出。

## 全参 GRPO

GRPO沿用相同的原子Agent协议和完整QA，需用GRPO配置单独prepare，保证**每个**可评分窗口的结尾都在rollout上限内。rollout从视频起点到 `min(max_prefix_seconds, floor(video_duration))`；若任一窗口超出实际视频长度，运行时整条排除。它不会在最后一次参考回复后立即结束，因此也会惩罚尾部额外开口。本机已生成 `train/prepared/full_grpo_task_token_20261027`：3996训练QA、1004验证QA，prepare阶段未排除记录；实际媒体长度和token上限仍需运行时检查。`supervise_silence`控制SFT标签；GRPO的沉默由采样动作和时间奖励学习，不另加silence交叉熵。

推荐先做SFT，再复制GRPO配置，将 `model_path` 指向新协议SFT完整导出目录；加载导出时保留训练过的token行。**默认GRPO配置仍指向原始基座**，没有自动串联SFT，也不会自动找到最新checkpoint。不同阶段不续用SFT优化器状态。

```bash
# --config必须位于子命令之前；复制配置后可改model_path为自己的SFT导出。
train/.venv/bin/python -m train --config train/configs/minicpmo_full_grpo.json prepare \
  --output train/prepared/full_grpo_task_token --max-samples 6000

CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train --config train/configs/minicpmo_full_grpo.json profile \
  --prepared train/prepared/full_grpo_task_token --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_grpo_profile

CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train --config train/configs/minicpmo_full_grpo.json grpo \
  --prepared train/prepared/full_grpo_task_token --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_grpo
```

每条QA默认采样4条完整轨迹；逐秒先输入音视频，再由原生接口采样listen、speak、回复与可选instruction。不输入参考回复，不执行工具，各rank同步同一采样结果。温度固定1，无top-k/top-p截断、重复惩罚或listen缩放，以保持采样和回放的概率一致。

奖励是**本地规则，不是LLM judge**：RTD/PR为时间F1×0.5＋匹配回复的字符二元组F1×0.5；Agent为时间F1×0.4＋回复相似度×0.4＋task_start instruction匹配×0.2，非法调用默认每次扣0.1。内容和调用只在一对一时间命中的窗口内得分。可通过现有 `reward_weights` 配置覆盖权重。该相似度不能可靠判断语义正确性，尤其易惩罚正确改写；开始大规模RL前应检查奖励是否符合目标。

组内奖励标准化得到优势，采用clip=0.2的策略损失，`kl_beta=0`，不加载冻结参考模型。组内奖励相同时整组跳过、不推进优化步。采样后按原生feed边界做可微KV回放，先后验证无梯度和带梯度log-prob；最大误差超过0.15会报错，避免使用不一致的概率更新。默认启用激活CPU卸载，保存张量预算128 GiB，额外CPU优化器、媒体缓存等不计入该预算。

输出额外包含 `trajectories.rankN.jsonl`（采样动作、事件、奖励），metrics含奖励方差、零方差跳过、回放误差及策略比值。GRPO目前没有独立验证集rollout指标循环；训练奖励不等于held-out benchmark成绩。profile只验证选中QA的一组候选，可能因零方差而没有实际优化更新；看metrics确认。完整模型使用同一个export命令导出。

## 语义Judge GRPO：独立沉默奖励与QA batch=4

新配置 [minicpmo_full_grpo_judge.json](configs/minicpmo_full_grpo_judge.json) 显式设置 `reward_backend=llm`；旧GRPO配置及其字符相似度奖励不变。新配置指向当前SFT完成后的HF导出，要求存在 `full_training_model.json`，不能误用原始基座或直接传ZeRO checkpoint。当前SFT、训练后Realtime任务和6–7卡benchmark队列不变；本次没有启动Judge或GRPO。

Judge输入为问题、当前参考回复/证据文本/上下文、available工具定义及候选回复/调用，不输入视频和音频，也不传后续工具返回和完成回复。沿用benchmark的0/0.5/1/1.5/2五档语义标准，归一化到0–1；**仅评分一对一时间命中的回复**。未命中窗口贡献0，按全部参考窗口数归一化；Agent缺失/非法调用的调用分为0，不用文本相似度兜底。

独立沉默分 `S=2*quiet_negative_seconds/negative_seconds-1`：从问题输入后的第一个合法整数输出秒，到实际rollout终点，排除所有参考闭区间窗口；负时间秒中没有**新发起回复**即视为正确等待。原生控制token和turn状态用于记录新开口；合法回答的跨秒续说不扣分，无负时间秒时S=0。

| 任务 | 奖励 |
| --- | --- |
| RTD / PR | `0.4*T + 0.4*C + 0.2*S - 0.1*非法调用次数` |
| Agent | `0.3*T + 0.3*C + 0.2*A + 0.2*S - 0.1*非法调用次数` |

T为时间F1，C为归一化语义内容分，A为归一化instruction调用分。完全沉默通常只能获得0.2分，不能获得时间/内容/调用分；同组相同的沉默分会在优势标准化时抵消，不能靠常量加分解决零方差。文本Judge仍依赖参考标注，无法独立核实未提供的音视频事实，尾部漏标可能产生假负奖励。

**batch含义：** 每条QA顺序采样4条候选，收集4个评分成功且奖励非零方差的QA组，再完成一次优化器更新，即QA batch=4、16条候选轨迹。四卡共同处理同一逻辑QA，不是独立QA DP，不再乘4得到QA batch=16。失败/零方差/不可行QA不占有效累积；epoch尾部不足4组用零损失图补齐边界，按实际组数归一化，不重复训练/丢弃有效QA。候选均在同一次权重更新前采样，保证on-policy回放。

学习率1e-6、2 epochs、clip=.2、KL=0、300秒/32768 tokens和每秒20-token预算保持；checkpoint每50次更新保存并保留最新1份，所有权重必须在 `~/jimodels` 之外。新配置的profile不会强制将累积改成1，默认选4条较短QA；可能因零方差/排除导致不足4个有效组，实际数量看日志。

### 服务准备与后续启动

服务配置 [judge_qwen27b_service.json](configs/judge_qwen27b_service.json) 使用现有 `benchmark/.venv-joyai` 环境、本地外部Qwen3.8-27B权重，默认GPU7、BF16、localhost:8917、8192上下文、最多8个并发序列、显存利用率.9。评分请求关闭thinking、温度0、最多512输出tokens；不截断过长输入。服务代码不停止任何进程，显式启动时发现卡上占用超过2 GiB就拒绝启动。单卡实际加载、显存和评分质量尚待后续GPU验证。

```bash
# 默认只打印启动计划，不查询/占用GPU、不启动服务。
benchmark/.venv/bin/python train/scripts/serve_grpo_judge.py

# 以下仅在后续明确启动、且已释放目标卡之后执行；前台运行。
benchmark/.venv/bin/python train/scripts/serve_grpo_judge.py --start
```

之后若要将benchmark缩到GPU6，必须先停止其旧监护/调度进程，创建新的单卡不可变计划，校验模型、数据、prompt和协议身份后按新分片迁入全部已完成预测，再恢复队列；不能直接改旧 `parallel_plan.json` 的GPU数。此切卡操作等待后续指令，本次没有执行。

Judge仅由rank0调用并广播结果，最多8路请求；缓存指纹包含模型元数据/分片stat、prompt、评分输入和解码配置。最多3次尝试（初次+2次重试）；HTTP错误、输入超限、输出截断和非法JSON均不当作模型0分，仍失败则整组跳过。日志不保存provider异常文本或API key。运行时要求已启动的服务 `/v1/models` 中包含指定model ID，不自动启动服务。

```bash
# 先确保SFT完整导出已存在，原有SFT/Realtime任务已按安排完成或由用户显式停止。
train/.venv/bin/python -m train --config train/configs/minicpmo_full_grpo_judge.json prepare \
  --output train/prepared/full_grpo_judge --max-samples 6000

# 首先验证一个完整累计更新；输出目录必须新建。
CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train --config train/configs/minicpmo_full_grpo_judge.json grpo \
  --prepared train/prepared/full_grpo_judge \
  --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/grpo_judge_smoke --max-steps 1
```

正式运行去掉 `--max-steps 1` 并使用新的外部输出目录。支持复制配置加入已有 `wandb` 配置块。metrics/W&B记录时间/内容/调用/沉默分、实际QA/候选数、padding、累计全静默比例、零方差组比例、Judge请求/缓存/失败/重试、梯度范数（引擎可提供时）和回放差异。`step_wall_seconds` 包含收集轨迹、Judge及反向耗时，`backward_wall_seconds`单列更新阶段；只看反向耗时会严重低估GRPO成本。GRPO暂没有独立验证集rollout循环，奖励上升不能替代benchmark。

## 独立QA数据并行与加速

新增显式配置 [minicpmo_full_sft_dp.json](configs/minicpmo_full_sft_dp.json)，复用DeepSpeed原生ZeRO-2；旧配置仍保留同QA的ZeRO-3模式。新模式每卡处理不同QA、每条仅做一次完整可微前向，再做反向和更新；流式音视频顺序、全部response、沉默mask和原窗口保持不变。不缓存训练中会变化的编码器特征。

当前独立QA配置默认梯度累积4：四卡、每卡1条QA时，全局有效batch为16。3996训练QA每轮250次更新，两轮500次（未扣运行时排除）；末尾补空槽，不重复训练样本。不同卡数会改变有效batch，公式为GPU数×每卡1×累积步数；学习率不自动缩放。比较吞吐应使用QA/小时，不能只比较step/s。独立QA模式仅支持SFT、ZeRO-2，不能给旧ZeRO-3模式直接分配不同QA。当前DeepSpeed CPU卸载版本需通过本地兼容层保存每次中间累积梯度；两卡累积1/4均已对照单进程平均梯度和参数更新验证。

`length_bucket_size=120`：每轮先按seed打乱，再在120条QA的局部桶中按最长目标时刻排序，减少长短视频混批等待；不会丢弃、截断或重复有损失的QA。桶大小需被使用的GPU数整除，可设置0关闭。无效/末尾空槽使用正常短QA前向但将loss乘0，并按全局有效QA数归一化；不能用参数求和替代正常图，否则ZeRO-2可能以不同顺序归约梯度。

`sft_activation_gpu_budget_gib=4`允许最多保留4 GiB保存张量在GPU，剩余卸载CPU；还会按当前可用显存和 `sft_activation_gpu_reserve_gib=12` 缩小保留量。两者都是预算，不是总显存保证；最长样本仍需实际profile。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 train/.venv/bin/torchrun --standalone --nproc_per_node=6 \
  -m train --config train/configs/minicpmo_full_sft_dp.json run \
  --prepared train/prepared/full_sft_task_token --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft_dp6
```

profile可在复制配置中用 `profile_sample_indices` 选多条QA，在不同卡上验证长短轨迹。`save_profile_checkpoint=false`可避免仅profile也写大checkpoint；`profile_gradients=false`可关闭额外全梯度诊断。真实两卡梯度等价检查：

```bash
CUDA_VISIBLE_DEVICES=0,1 train/.venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m train.tests.dp_cuda_worker
```

若要安全停止新版本，创建 `输出目录/STOP`；训练会在完整优化步后保存最终checkpoint再退出。`save_total_limit`可限制本run保留的编号checkpoint数（0不清理）；不会删除checkpoint-final。六卡与四卡优化器状态不能直接互相resume；只有同配置、同world size可恢复。

## 保存、恢复、导出和评测

输出包含 `run_config.json`、`metrics.rankN.jsonl`、`memory_report.rankN.json`、`checkpoint-final/`。配置可设置 `save_every_steps`，默认结束保存。恢复要求同配置、prepared与卡数，且使用新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m train run --prepared train/prepared/full_sft_task_token \
  --resume /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft/checkpoint-final \
  --output /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft_resumed --max-steps 100

train/.venv/bin/python -m train export \
  --checkpoint /remote-home/yushengliao/speech/jimodels/training_outputs/OmniDuplex/full_sft/checkpoint-final --output /remote-home/yushengliao/speech/jimodels/openmodels/OmniDuplex-MiniCPM-SFT
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

CPU检查覆盖原子注册、旧ID保持、新行梯度、微型模型保存加载、全参媒体梯度、连续回复与沉默mask、旧协议拒绝及checkpoint游标/RNG。GRPO检查还覆盖原生动作概率、裁剪损失、规则奖励、原子分隔符回放梯度与CPU激活卸载。DeepSpeed CUDA集成检查仍需显式启用 `RUN_ZERO_CHECKPOINT_TEST=1`。新Judge GRPO路径已完成72项训练层测试（71项通过，1项需显式CUDA开关而跳过）；包含本地HTTP假服务和微型模型梯度平均对照。未启动真实Judge/GRPO，单卡加载、ZeRO-3实卡累计与真实SFT导出接入仍待后续GPU验证。既有SFT训练独立运行。


## 当前训练完成后自动评测（仅 Realtime）

当前 `full_sft_dp4_bs16_tail10_wandb_20261027` 已配置训练完成后的自动任务。它等待训练进程全部退出，核对4个rank的最终报告均完成500次更新、最终checkpoint完整且配置匹配，随后在CPU导出完整HF模型，最后用GPU 0–3运行四实例Realtime benchmark。若训练提前停止或保存不完整，任务记录failed，不启动推理。不会运行Probe或付费LLM judge，不复用基座模型的预测。

- 工作流：[benchmark_after_training.py](scripts/benchmark_after_training.py)，通过JSON配置传入进程、路径和最终更新数。`--prepare-only`只校验数据和DP计划，不加载模型；正式运行命令去掉该参数。进程锁防止同一任务重复启动。
- 当前配置：`train/outputs/full_sft_dp4_bs16_tail10_post_benchmark.config.json`。
- 数据：外部AOI-Bench的2107条QA，保存原始JSON快照；通过视频符号链接适配A/B路径，保留QA内容和ID。
- 导出权重：`/remote-home/yushengliao/speech/jimodels/openmodels/OmniDuplex-MiniCPM-o-4.5-SFT-bs16-tail10-20261027`。
- 结果与状态：`benchmark/results/sft_dp4_bs16_tail10_20261028/`，其中 `after_training_status.json` 记录等待、导出和评测状态，最终Realtime结果在 `suite/realtime/minicpmo/`。
- 停止等待任务：在上述结果目录创建 `STOP` 文件。该标记在等待期间及导出前检查；已开始的评测需要停止对应suite进程。不会停止训练。

任何模型权重、训练checkpoint和导出模型都不得实际存放在 `~/jimodels` 或其任意子目录。下载/导出模型使用外部 `openmodels`，训练checkpoint使用外部 `training_outputs`；仓库中可保留指向外部存储的符号链接。该自动工作流会检查checkpoint和导出目标的实际解析路径。
