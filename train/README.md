# MiniCPM-o-4.5 全参数流式 SFT / GRPO

入口为 `python -m train`。支持全参 SFT 和全参 GRPO：训练 LLM、视觉/音频编码器及连接层，不加载 TTS。LoRA 训练已移除。历史 benchmark adapter 的推理加载保留在隔离的兼容模块，不是训练入口。

## 模型、数据和环境

模型默认位于 `/remote-home/yushengliao/speech/jimodels/openmodels/MiniCPM-o-4.5`。
数据为 `/remote-home/yushengliao/speech/jimodels/datasets/AOI-Dataset/all_qa.json`，`videos_dir` 指向 AOI-Dataset 根目录。当前扩充版包含 19,210 条 QA、4,925 个视频；它是旧 5,000 条加新增 14,210 条。数据下载、更新和媒体检查见[仓库README](../README.md)，训练统计见[数据报告](docs/aoi_dataset_statistics.md)。

```bash
python3.11 -m venv train/.venv
train/.venv/bin/pip install -r train/requirements.txt
train/.venv/bin/python -m train --help
```

需要 FFmpeg/ffprobe 和兼容 PyTorch/CUDA 的 FlashAttention 2。当前普通 CE 八卡实验配置为 [minicpmo_full_sft_plain_ce_dp8.json](configs/minicpmo_full_sft_plain_ce_dp8.json)：有效 batch 16（8×1×2）、学习率 1e-6、两轮、65536 token、完整视频与沉默监督、ZeRO-2 CPU optimizer、TensorBoard。CLI 默认仍指向独立的 [minicpmo_full_sft.json](configs/minicpmo_full_sft.json)，使用时须显式选择配置。GRPO配置为 [minicpmo_full_grpo.json](configs/minicpmo_full_grpo.json)，相对路径按仓库根目录解析。各配置的 BF16、FlashAttention 2、每秒 chunk 预算20、seed=42 等参数以选中的 JSON 为准；`full_video` 模式不会按 `max_prefix_seconds=300` 截短视频。ZeRO-3配置为 [zero3_cpu.json](configs/zero3_cpu.json)，FP32 master/Adam状态卸载CPU。

## 文本 task_start 与连续多回复目标

训练协议为 **`task_start_text_sft_v1`**。`<|task_start|>` 是普通文本，用原始基座已有的多个token编码；不注册特殊token、不扩展词表、不初始化新embedding或lm_head行。输出保持 `回复<|task_start|>instruction`。分隔符的全部token参与结构监督，可跨chunk生成。旧原子token checkpoint只保留评测读取能力，不能用于新协议续训。
输入 embedding 与 lm_head 随完整模型训练，但普通文本协议不扩表；磁盘上的原始模型文件不改写。

```text
<|speak|>我来查找替换配件。<|task_start|>查询视频中配件的可售替换选项。<|turn_eos|>
```

每个Agent回复内分隔符只出现一次，后面是非空instruction，不生成JSON。不执行工具、不输入工具返回和完成回复，也不使用Agent负样本。RTD/PR仅监督自然语言回复。

每条QA是一段连续轨迹，覆盖全部可评分response。先输入 `[t-1,t)` 媒体，再监督时刻t的输出；后续回复能看到之前按时输入的真实回复，未来回复不会提前输入。每个目标都满足问题、证据、trigger及原窗口的时间约束；与前一回复跨chunk冲突时只在原窗口内顺延。输入结束于最后一个目标完成。

`supervise_silence=true`：问题输入后所有没有参考回复 chunk 的空闲秒都监督 listen，包括窗口内以及最后一次回复后直到视频结束；问题前与输入媒体位置仍 mask。当前普通 CE 实验使用 `decision_loss_mode=plain_ce`，每条 QA 的有效控制、回复和 instruction token 统一平均，再沿用按 QA 平均的 batch reduction。`balanced_gate`、`event_balanced` 是保留的可选对照配置；历史 `token_mean` 是分组均值相加，不能与普通 CE 混称。

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

prepare按UUID视频划分，默认验证视频比例20%。`--max-samples`按QA计数，默认仍为12；验证采样上限为 `round(max_samples * validation_fraction)`。6000 只适用于旧 5k 数据；当前 19,210 条数据不要沿用此上限。正式全量准备使用足够大的上限，并核对 manifest 的 `sampling_omissions` 全为 0。正式编码仍会剔除不可行轨迹。
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

SFT还记录 `loss_silence`、`loss_speak`、`loss_control`、`loss_response` 和
`loss_agent_instruction`，分别表示沉默、开口、其他控制token、回复文本和Agent指令的
token平均CE，以及各自的 `*_tokens`、`*_qa`。训练指标在 `train/` 下，完整验证集指标
在 `validation/` 下；没有该类token时不填零loss。跨卡统计排除padding和无效QA。
新增 `loss_onset`、`loss_continuation_speak`、`loss_structure` 和gate/structure的实际加权贡献；总loss由decision、response、instruction三项贡献重构。新旧总loss数值不能直接比较。
详见 [分项loss口径](docs/loss_metrics.md)。旧日志无法还原历史分项。断点续训仍使用新的
外部输出目录；通过环境变量 `WANDB_RUN_ID` 和 `WANDB_RESUME=must` 续接原W&B run，
不会在恢复时重新添加step 0点。

## 全参 GRPO

GRPO沿用相同的文本Agent协议和完整QA，需用GRPO配置单独prepare，保证**每个**可评分窗口的结尾都在rollout上限内。rollout从视频起点到 `min(max_prefix_seconds, floor(video_duration))`；若任一窗口超出实际视频长度，运行时整条排除。它不会在最后一次参考回复后立即结束，因此也会惩罚尾部额外开口。本机已生成 `train/prepared/full_grpo_task_token_20261027`：3996训练QA、1004验证QA，prepare阶段未排除记录；实际媒体长度和token上限仍需运行时检查。`supervise_silence`控制SFT标签；GRPO的沉默由采样动作和时间奖励学习，不另加silence交叉熵。

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
- `src/infrastructure`：配置、文本分隔符校验、媒体、ZeRO、分布式、checkpoint。

阅读顺序：CLI → prepare/records → targets/trajectory → loading/streaming → graph → training/full → checkpoints。模块和关键函数有职责、输入输出及状态释放注释。`train.src.agent`、`train.src.checkpoints`保留当前公共接口；`train.src.token_rows.load_adapter`仅供历史benchmark推理，使用旧adapter时才需要另装PEFT。

```bash
train/.venv/bin/python -m unittest discover -s train/tests
```

CPU检查覆盖文本分隔符、词表不变、多token梯度、微型模型保存加载、全参媒体梯度、连续回复与沉默mask、旧协议拒绝及checkpoint游标/RNG。GRPO检查还覆盖原生动作概率、裁剪损失、规则奖励、文本分隔符回放梯度与CPU激活卸载。DeepSpeed CUDA集成检查仍需显式启用 `RUN_ZERO_CHECKPOINT_TEST=1`。新Judge GRPO路径已完成72项训练层测试（71项通过，1项需显式CUDA开关而跳过）；包含本地HTTP假服务和微型模型梯度平均对照。未启动真实Judge/GRPO，单卡加载、ZeRO-3实卡累计与真实SFT导出接入仍待后续GPU验证。既有SFT训练独立运行。


## 当前训练完成后自动评测（仅 Realtime）

当前 `full_sft_dp4_bs16_tail10_wandb_20261027` 已配置训练完成后的自动任务。它等待训练进程全部退出，核对4个rank的最终报告均完成500次更新、最终checkpoint完整且配置匹配，随后在CPU导出完整HF模型，最后用GPU 0–3运行四实例Realtime benchmark。若训练提前停止或保存不完整，任务记录failed，不启动推理。不会运行Probe或付费LLM judge，不复用基座模型的预测。

- 工作流：[benchmark_after_training.py](scripts/benchmark_after_training.py)，通过JSON配置传入进程、路径和最终更新数。`--prepare-only`只校验数据和DP计划，不加载模型；正式运行命令去掉该参数。进程锁防止同一任务重复启动。
- 当前配置：`train/outputs/full_sft_dp4_bs16_tail10_post_benchmark.config.json`。
- 数据：外部AOI-Bench的2107条QA，保存原始JSON快照；通过视频符号链接适配A/B路径，保留QA内容和ID。
- 导出权重：`/remote-home/yushengliao/speech/jimodels/openmodels/OmniDuplex-MiniCPM-o-4.5-SFT-bs16-tail10-20261027`。
- 结果与状态：`benchmark/results/sft_dp4_bs16_tail10_20261028/`，其中 `after_training_status.json` 记录等待、导出和评测状态，最终Realtime结果在 `suite/realtime/minicpmo/`。
- 停止等待任务：在上述结果目录创建 `STOP` 文件。该标记在等待期间及导出前检查；已开始的评测需要停止对应suite进程。不会停止训练。

任何模型权重、训练checkpoint和导出模型都不得实际存放在 `~/jimodels` 或其任意子目录。下载/导出模型使用外部 `openmodels`，训练checkpoint使用外部 `training_outputs`；仓库中可保留指向外部存储的符号链接。该自动工作流会检查checkpoint和导出目标的实际解析路径。


### 普通文本协议与平衡 loss

可选 balanced_gate 配置对 RTD/PR/Agent 使用分组平衡；分隔符的所有普通token均为structure，
不算新开口。完整公式见 [loss说明](docs/loss_metrics.md)。原子token审计及
`full_sft_task_token_20261027` 是历史结果，不可用于当前训练。
新协议须重新prepare到独立目录，例如：

```bash
train/.venv/bin/python -m train --config train/configs/minicpmo_full_sft_dp.json prepare \
  --output train/prepared/full_sft_task_text --max-samples 100000 --validation-fraction 0.2
```

保持视频级train/validation隔离；不能通过删除旧tokenizer注册来转换已有权重。
新实验同时更换平衡loss和分隔符编码，评测时应记录两个变量，不能将收益归因于单一修改。

### 可选：按回复事件平衡 SFT loss

新配置 [`configs/minicpmo_full_sft_event_balanced_dp.json`](configs/minicpmo_full_sft_event_balanced_dp.json)
显式启用 `decision_loss_mode=event_balanced`。在现有训练命令中替换 `--config`，
使用新的输出目录开始实验；不要直接恢复旧目标的 optimizer checkpoint。
原配置、正在运行的任务和自动队列不会切换。

该模式分别平衡首次/后续开口，所有空闲位置合并计算一个沉默均值（不按回复前后等权拆分）；内容与 Agent instruction 按每次
参考回复平均；续说、结束和普通文本 `<|task_start|>` 分隔符分别监督。
默认总目标为 `0.8 × gate + 0.2 × structure + response + instruction`，
缺失类别按层重新归一化。全部空闲位置（包含参考窗口内）继续监督沉默，参考回复
连续输出不受影响。它加强错误开口的监督，但不直接按生成回复次数累计惩罚。
公式、日志和兼容规则见 [loss 指标说明](docs/loss_metrics.md#event-balanced-objective-opt-in)。

### 完整视频 SFT

使用 `configs/minicpmo_full_sft_full_video_dp.json`：`sft_horizon=full_video`、
`max_sequence_tokens=65536`、`supervise_silence=true`，并将固定静默尾段设为0。
该模式从视频起点覆盖到 `floor(video_duration)` 秒，不应用旧的300秒回复起点限制。
每条参考回复正常监督，其余问题后的空闲秒都监督 listen；参考时间窗口不变。
最后不足一秒沿用现有整秒训练约定，不补造音视频。

该模式必须重新 prepare，使用新的输出目录训练。序列超限会排除整条 QA，
不再裁短视频尾部；CPU audit 只验证文本 token 下界，不保证包含媒体后的65536
预算足够。应在空闲 GPU 上用真实媒体 profile 检查最长轨迹后，再决定是否继续扩大。
完整视频监督会增加训练时间和主存/显存需求，现有实验与自动队列不会切换。

### GRPO：显式 TP/FP/FN/TN 奖励

使用 [`configs/minicpmo_full_grpo_confusion_judge.json`](configs/minicpmo_full_grpo_confusion_judge.json)
显式启用 `reward_mode=confusion_judge`，沿用本地 LLM Judge 服务配置和 GRPO 命令。
先将 `model_path` 配置为实际采用普通文本 Agent 协议的 SFT 导出，再重新 prepare，
使用新的外部输出目录训练；旧 judged-GRPO 配置不改变。

TP由一对一窗口匹配确定，再用Judge质量转换为−1到+1；FN每次扣1，FP每次扣0.5，
三项按参考窗口数归一化。真正等待的TN项按负侧秒数归一化，限制在±0.1；
非法调用每次再扣 `0.1/窗口数`。空开口也计FP，续说不计新开口、也不赚TN。
新配置使用完整视频rollout与65536 token预算，不裁短尾段；Judge失败排除整组。
全部公式、分项日志与局限见 [奖励说明](docs/grpo_confusion_reward.md)。

## SFT → Realtime benchmark → GRPO 自动衔接

`train/scripts/sft_bench_grpo.py --config <chain.json>` 按操作配置顺序执行：验证 SFT 正常保存最终 checkpoint，导出到仓库外，运行已有 realtime DP suite，再启动本地 Judge、GRPO profile 和正式 GRPO。`--check-only` 只检查配置和视频级数据划分，不加载模型。配置显式指定 benchmark 配置、GRPO/profile 配置、prepared、外部权重目录、GPU、Judge、待完成的 OmniPro 状态和现有队列 transition 路径。

本机 2026-10-31 的配置在 `train/outputs/sft_bench_grpo_fullvideo_20261031/chain.json`，状态在同目录 `status.json`，日志按阶段保存。0–3 卡先测最新 SFT 的 AOI-Bench realtime，之后仍用 0–3 卡运行 GRPO，7 卡运行 Qwen3.8-27B Judge。GRPO 使用 `confusion_judge`、完整视频、每 QA 4 条候选、累积 4 个 QA/更新、2 epoch、LR 1e-6；这里的有效 QA batch 是 **4**，不等于 SFT 的 16。复用 SFT 的视频隔离划分：3996 训练 QA、1004 验证 QA，验证 QA 不参加策略更新。

借卡前等待整组 OmniPro 推理和 Judge 完成，然后通过 `probe_gpu_reservations.json` 只保留 7 卡给 Judge。4–6 卡持续派发 benchmark 分片，原分给 7 卡的分片在其他卡空闲时运行；分片 manifest、输出目录及已完成结果不变。已启动的 7 卡 worker 正常结束、释放显存后再启动 Judge。GRPO 结束或收到正常停止信号后解除保留，队列重新可用 4–7 卡；不暂停调度器，也不杀掉在途 benchmark worker 来抢卡。SFT 异常结束、benchmark 未完整完成、Judge 失败或 profile 未完成一次有限 loss 的优化更新时，链路报错停止，不会自动跳过检查。

控制器只允许一个实例，保存 GPU 保留归属日志，释放时不会改写其他任务持有的保留。创建链路输出目录下的 `STOP` 文件可请求停止；若在等待 SFT，只停止等待任务，SFT 本身继续。重启控制器前需先排查失败原因；不自动覆盖已有 GRPO 输出或恢复旧优化器。运行时使用冻结的代码快照，避免等待期间仓库修改改变评测或训练语义。

本机自动链路已启用 [四卡候选并行配置](configs/minicpmo_full_grpo_confusion_judge_dp4.json)：每卡生成并回放一个候选，ZeRO-2 同步梯度，Judge 深度为 2 的有界流水线与下一组采样重叠。每次更新前清空流水线，保持同策略采样。正式训练省去额外无梯度回放，但每次带梯度回放仍验证概率一致性；profile 保留双重检查，并覆盖约 50–299 秒视频。配置、归一化、日志和限制见 [GRPO 执行说明](docs/grpo_confusion_reward.md#four-gpu-candidate-parallel-execution)。真实显存及速度以 SFT 完成后的四卡 profile 为准，profile 失败则停止链路，不自动降低视频覆盖或候选数。

## 本地 TensorBoard：SFT / GRPO，无需 W&B

训练环境安装：

```bash
train/.venv/bin/python -m pip install -r train/requirements-tensorboard.txt
```

在训练 JSON 中配置以下字段。现有 SFT 配置与 `minicpmo_full_grpo_confusion_judge_dp4.json` 已启用 TensorBoard；本机后续 GRPO 的 W&B 已关闭。正在运行的 SFT 配置保持原样，通过下文 JSONL 桥接观察，不重启训练。

```json
{
  "wandb": {"mode": "disabled"},
  "tensorboard": {
    "enabled": true,
    "log_dir": "train/tensorboard",
    "flush_seconds": 10
  }
}
```

仅 rank 0 记录白名单数值，不上传文本、QA 或密钥。事件文件在 `<log_dir>/<训练输出目录名>`，独立新实验使用新的输出目录名，避免混合曲线。相对 log_dir 按仓库根目录解析。SFT 的 `train/loss` 按一次参数更新内的梯度累积微批求均值，每个更新一个点；分项原始 CE 在更新内按 token 数加权，缺失分项不计零。GRPO 每次更新记录实际 QA batch、候选数、奖励分项、零方差/排除组数、概率回放误差和 rollout/Judge 等耗时。`validation/*` 是参考轨迹验证 loss，并非 autonomous benchmark F1。

启动本地服务：

```bash
CUDA_VISIBLE_DEVICES= train/.venv/bin/python -m tensorboard.main \
  --logdir train/tensorboard --host 127.0.0.1 --port 6006 \
  --reload_interval 5 --load_fast false
```

本机已后台启动服务及当前 SFT 的 JSONL 跟随器。通过 VS Code 转发训练机 6006 端口，或在自己的电脑执行 `ssh -L 6006:127.0.0.1:6006 <训练机SSH地址>`，然后访问 `http://localhost:6006`。无需 W&B 登录；日志保存在本地，不占 GPU。TensorFlow 未安装的提示不影响 Scalars 曲线。

已有训练、旧实验或当前运行中的训练，可回填并跟随本地 JSONL：

```bash
CUDA_VISIBLE_DEVICES= train/.venv/bin/python train/scripts/tensorboard_from_jsonl.py \
  --run-dir train/outputs/<run> --log-dir train/tensorboard --follow
```

不加 `--follow` 则只回填当前已有记录。跟随器忽略未写完的最后一行，保存字节偏移及未完成微批缓冲，正常重启不会重放已记录的历史；检测到文件替换或截断会拒绝续接。不要给已经原生记录 TensorBoard 的同一训练再启动桥接器。不要更改训练配置来给正在执行的任务追加日志，以免破坏恢复身份；使用桥接方式。

已完成并验证最终 SFT 导出、希望直接开始 GRPO 时，可显式跳过 benchmark 前置任务：

```bash
train/.venv/bin/python train/scripts/sft_bench_grpo.py \
  --config train/outputs/sft_bench_grpo_fullvideo_20261031/chain.json --grpo-only
```

该模式仍校验四卡 SFT 完成报告、最终 checkpoint、导出 metadata 哈希和 GRPO/profile 模型路径一致性，然后启动独立 Judge、profile 和正式 GRPO。它不会把已取消的 benchmark 标成成功。取消旧评测队列是独立的显式操作；先停其监督/自动恢复程序，保留已完成预测。`--grpo-only --check-only` 仅做校验，不加载模型。本机 2026-11-01 已按用户要求取消两路 benchmark 与旧队列自动恢复，并切换到该直启模式。

### 普通交叉熵 SFT 与 GRPO 阶段检查

普通 SFT 对照使用 `train/configs/minicpmo_full_sft_plain_ce_dp.json`，其中
`decision_loss_mode: plain_ce` 将每条 QA 的全部有效输出 token 统一计算 CE 均值，
保留完整视频、全部 response 和沉默监督，之后沿用按 QA 平均的 batch reduction。
旧 `token_mean` 模式是各分项均值相加，并非此普通 CE 模式；已有配置默认行为不变。
loss 分项仍可通过 TensorBoard 查看。

GRPO 可配置 `grpo_evaluation: {sample_indices: [...], seed: 20261101, every_steps: 10}`，
在第 0 步及固定更新间隔对 prepared validation 的固定 QA 自主生成，输出
`timing_evaluation/report.md`。它对比同一批成功 QA 的 FP、precision、recall、F1，
不将失败项当成零 FP，也不代替内容 Judge。详细运行安排和当前路径见
[GRPO pilot 与普通 CE 对照](docs/grpo_pilot_plain_ce_20261101.md)。

当前普通 CE 对照使用八卡 DP、梯度累计 2，保持有效 batch size 16；完成后自动
用 0–3 卡评 AOI-Bench、4–7 卡评 OmniPro。状态、外部 checkpoint 路径和自动评测
安排见[八卡普通 CE 训练说明](docs/plain_ce_dp8_20261101.md)。


### 追加数据：固定旧验证集并使用全量 QA

`--split-from` 指向上一轮 prepared 的 `manifest.json`（其旁边需要有
`validation.json`）。已有视频保持原归属，新视频按 seed 和视频路径的 SHA256
确定验证归属，比例约为 `--validation-fraction`；同一视频的 QA 不跨划分。
不传该参数时仍使用原来的随机视频划分。`--audit-report` 仅接受相同数据哈希和
相同配置的 CPU 审计，完整应用不可调度 QA 的排除记录。

```bash
train/.venv/bin/python -m train \
  --config train/configs/minicpmo_full_sft_plain_ce_dp8.json audit \
  --output train/prepared/expanded_audit.json

train/.venv/bin/python -m train \
  --config train/configs/minicpmo_full_sft_plain_ce_dp8.json prepare \
  --output train/prepared/expanded_plain_ce \
  --max-samples 100000 --validation-fraction 0.2 \
  --split-from train/prepared/full_video_event_65536_20261029/manifest.json \
  --audit-report train/prepared/expanded_audit.json
```

首次准备没有历史 manifest 时省略 `--split-from`。不要重复使用已存在的输出目录。
旧 metadata 缺少新增视频时，audit 会用 ffprobe 补查；当前运行已保存完整时长快照，
没有覆盖原 metadata 的来源信息。manifest 包含候选数、实际数量、采样遗漏及排除原因。
带历史划分时，每轮同时记录 `validation/previous_loss*`（固定旧 QA）和
`validation/added_loss*`，并保留全验证集指标。普通 CE 还会分别记录
`loss_first_onset`、`loss_later_onset`；这些只用于诊断，不改变目标或梯度。

当前全量运行的 1942 步训练、等待 Judge、八卡 profile 及自动分卡评测安排见
[19,210 条数据实验说明](docs/plain_ce_full19210_20261103.md)。
