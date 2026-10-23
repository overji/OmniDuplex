# MiniCPM-o 4.5 双工 SFT 与 GRPO

入口为 `python -m train`。默认配置仍为单 GPU LoRA SFT；另提供语言、视觉、音频及连接层的全参 SFT 和简单流式 GRPO。三者均为音视频输入 / 文本输出，不加载 TTS，支持 RTD、PR 和 Agent 协议。下文前半部分介绍原有 LoRA pilot，末尾介绍全参训练。

## 数据范围与时序

训练只使用 `agent_token_task_start_v1`，不再支持外层 structured JSON 或旧七字段调用的训练分支。生成数据中的 `agent_call` 在训练入口只读转换为 `task_start`，仅保留非空 `instruction`；原始 JSON 不改写。三种训练配置使用同一适配、目标构造与奖励口径。

第一版对每条 QA 只取最早一个可评分 response，从视频 0 秒开始保留完整上下文，到目标回复生成完毕结束。默认目标发声前缀最多 32 秒、总序列最多 8192 token；回复跨秒可能延长输入。超限报错或在准备阶段排除，绝不静默截掉历史或标签。

- 先按 video 文件名划分训练 / 验证，随后在各模式之间轮流采样；同一个 UUID 视频不会跨集合。若存在不同 UUID 的重复素材，需先去重。
- 首次发声采用不早于 question 输入、证据 available_time、trigger confirmed_time 和 trigger_time 的整数秒边界，且必须位于参考窗口内。
- 先输入 `[t-1,t)` 秒媒体，再预测时刻 `t` 的输出；复用 benchmark 的帧时间戳选择、16 kHz 音频和缺失音轨补零规则。
- 通过原生 `streaming_prefill` 录制冻结编码器的 embedding，不运行 LLM 推理。训练时对交错序列执行具有因果 mask 的 LLM forward。
- 使用原生 `<unit>`、`<|listen|>`、`<|speak|>`、`<|turn_eos|>`、`<|chunk_eos|>`、`</unit>`；文本按 chunk 预算分段，回复内容和 Agent JSON 均可跨秒。
- system 只包含任务规则、可用工具和 schema；参考答案、证据、窗口只供构造标签，工具返回和后续完成回复不进入输入。
- 当前 chunk 文本对齐是按 token 预算构造的 teacher-forcing 近似，不是有语音对齐标注的完整双工复现。需以原生 realtime 推理回归确认效果。

`supervise_silence` 默认 `false`：已有标注未承诺穷尽所有合理回复，故此前 listen token 作为上下文、不施加负标签。仍监督目标位置的 speak/turn token。若人工确认标签完整，可设为 `true`，监督问题后、窗口前的 listen；窗口内非目标位置继续不计损失。默认 pilot 不能据此宣称已充分训练“何时保持沉默”。

损失分为听说决策、回复内容、调用字段三组，各自按监督 token 数平均，再按配置权重求和；媒体、输入文本、强制插入的 chunk 结束标记和无监督 listen 的 label 为 -100。只对有标签位置分块计算词表 logits，降低显存。

## 环境

建议建立独立环境，避免改变现有 benchmark：

```bash
python3.11 -m venv train/.venv
train/.venv/bin/pip install -r train/requirements.txt
```

需要系统提供 `ffmpeg`、`ffprobe`。模型默认位于 `/remote-home/yushengliao/speech/jimodels/openmodels/MiniCPM-o-4.5`。配置集中在 `train/configs/minicpmo_lora.json`，相对路径以仓库根目录为基准；使用 `python -m train --config /path/to/config.json ...` 覆盖。

## 准备与运行

在仓库根目录执行：

```bash
python -m train prepare \
  --data benchmark/results/full_omni_duplex_plus_groupa100_20261014T120412Z/inputs/all_qa.json \
  --output train/prepared/task_start_pilot12 --max-samples 12

# 一条样本、一次真实前向 / 反向 / AdamW 更新；默认选择最短前缀
CUDA_VISIBLE_DEVICES=6 python -m train profile \
  --prepared train/prepared/task_start_pilot12 --output train/outputs/profile_short

# 指定原始 sample_index 测量更长序列
CUDA_VISIBLE_DEVICES=6 python -m train profile \
  --prepared train/prepared/task_start_pilot12 --sample-index 118 \
  --output train/outputs/profile_agent

# 小规模训练，配置默认 2 epoch，累积 4 条序列再更新
CUDA_VISIBLE_DEVICES=6 python -m train run \
  --prepared train/prepared/task_start_pilot12 --output train/outputs/pilot12
```

GPU 号只是示例，运行前选择空闲卡。所有命令在前台执行。`--max-steps 1` 可限制优化器更新次数。输出目录必须是新目录；当前 pilot 不支持恢复 optimizer 状态。准备时记录分组和排除原因，run 时不会改变视频划分。不要把用于训练的 QA 再作为独立 benchmark 测试集。

输出目录包含：

- `adapter/`：PEFT LoRA 参数。
- `run_config.json`：有效配置、数据来源及提示词哈希。
- `sequences.jsonl`：每条样本的实际秒数、序列长度和监督 token 数。
- `metrics.jsonl`：训练 loss、梯度范数和每轮验证 loss。
- `memory_report.json`：编码峰值、训练峰值 allocated / reserved 及 GPU 型号。

训练特征暂存 CPU 内存，属于小规模实现；扩大数据后应增加有版本校验的磁盘特征缓存。当前训练器是单 GPU，没有 DDP、自动选最优 checkpoint 或学习率搜索。

## Agent token 输出与 adapter 加载

Agent 正样本目标为原生 speak token、自然语言回复、单个 `<|agent|>`、调用 JSON、原生 turn 结束标记：

```text
<|speak|>我来查找替换配件。<|agent|>{"name":"task_start","arguments":{"instruction":"查询视频中配件的可售替换选项。"}}<|turn_eos|>
```

每次只支持一个调用，唯一参数为非空字符串 `instruction`，不允许额外参数。简短指令应包含任务目标和必要的已观察事实；不再输出 observed_context、suggested_capabilities 或 extracted_parameters 等字段。澄清时自然语言回复，不输出分隔符；RTD、PR 不使用 Agent 分隔符。调用内容可跨多个媒体 chunk，分隔符只出现一次；生成途中仍逐秒输入媒体。当前不执行工具，也不启用 TTS。

tokenizer 新增 `<|agent|>`，保留旧 token ID。本地基座中 token ID 为 151748，扩展词表为 151749；这些值由程序注册并记录，不能手工硬编码迁移。只训练新 token 的输入 / 输出两行 FP32 参数，旧词表行冻结。训练时重新嵌入所有分隔符位置，包括 system prompt 中的分隔符，防止预计算 embedding 阻断梯度。

`adapter/` 中除 PEFT 权重，还包含 `agent_rows.safetensors`、`agent_protocol.json` 和完整 `tokenizer/`。必须整体移动目录；不能仅加载 PEFT 权重，否则新 token 无法工作。

```python
from train.src.token_rows import load_adapter
# model 是原始 MiniCPM；base_path 是其完整基座目录。
load_adapter(model, base_path, 'train/outputs/agent_token_positive/adapter')
model.eval()
duplex = model.as_duplex(generate_audio=False)
```

加载器检查固定新协议、`agent_call_schema: "task-start"`、基座配置 / 分片索引指纹、旧词表 ID、新词表身份及向量形状，扩词表后恢复两行权重并合并 LoRA。旧 prepared 数据必须重新 prepare；旧 adapter、全模型导出和训练 checkpoint 不支持自动迁移，需要重新训练。`<|agent|>` 的字符串及旧词表 ID 保持不变，调用内容的协议单独升级。

训练侧 rollout 和 GRPO 使用新的原生 token 解析器。当前正在运行的 benchmark 结构化 `task_start` 队列保持不变；benchmark CLI 原有 agent-token 入口仍有 legacy schema 限制，本次没有改动其运行协议文件，不能直接用该入口评测新训练导出。

原生 `text` 返回会过滤 special token，因此新协议直接捕获生成 token ID，以 `<|agent|>` 切换状态。解析器跨秒保留状态，完整 turn 结束后才产生标准 `tool_calls`；半截 JSON、重复分隔符或非法调用不生成可执行调用，仍记录 `agent_attempted` 以审计失败尝试。文本回复为空时，以分隔符时间记录事件并单独标为空回复。

预测保留 `raw_generated_ids`、`response_text`、`agent_call_text`、标准 `tool_calls`、解析状态和调用起止时间。用户回复的首次内容时间继续用于现有窗口匹配，调用时间只是附加记录。Judge 直接复用分开的回复和调用，不需要再次让模型推断分隔位置。

## 显存估算

```bash
python -m train estimate --lengths 2048 4096 8192
```

估算通过读取 safetensors header 统计真实参数量，无需加载权重或占用 GPU。默认配置下非 TTS 参数约 90.24 亿，BF16 约 16.81 GiB；LoRA 约 4365 万参数，FP32 参数、梯度、Adam 两个状态共约 0.65 GiB。

对于 batch=1、rank=16、BF16 基座、gradient checkpointing、FlashAttention 2 及仅目标位置分块 logits，初步规划区间：

| 总序列长度 | 估算训练显存 |
| ---: | ---: |
| 2048 token | 22–28 GiB |
| 4096 token | 23–31 GiB |
| 8192 token | 25–36 GiB |

这些是含激活与 workspace 裕量的估算，不是保证上界；实际原生图像处理、CUDA context 和 allocator 会额外影响占用。序列长度包含每秒视觉、音频、控制 token 和 Agent 参数，不能只数文本。优先依据 `profile` 的实测报告选择 batch 和长度；本机为 A100 80GB。profile 使用 FlashAttention 2，避免 SDPA 对本地滑动注意力配置的 Transformers 警告；训练服务器需安装兼容当前 PyTorch / CUDA 的 `flash-attn`。

## 检查

```bash
python -m unittest discover -s train/tests
```

测试包括因果时间点、缺失素材、跨秒标签完整性、视频隔离，以及微型模型上分块 loss 与完整词表 loss 一致和仅 LoRA 接收梯度。缺少 PyTorch / PEFT 的环境会跳过梯度测试。

## 历史 agent_token_v1 实测（A100 80GB）

以下是旧调用协议的历史显存与梯度验证，不代表新 task_start 协议已经执行 GPU 训练；旧权重不用于新协议。

采用 rank=16、batch=1、FlashAttention 2、gradient checkpointing，仅训练 LoRA 和新增 token 两行。以下均完成一次前向、反向及 AdamW 更新；是链路验证，不是完整训练效果。

| 样本 | 总 token | 生成目标 chunk 数 | 峰值 allocated | 峰值 reserved |
| --- | ---: | ---: | ---: | ---: |
| Agent 正样本（原始 index 85） | 3902 | 12 | 19.65 GiB | 20.58 GiB |
| Agent 负样本（原始 index 43） | 2564 | 1 | 18.72 GiB | 19.59 GiB |

产物在 `train/outputs/agent_token_positive/` 与 `train/outputs/agent_token_negative/`；`token_gradients.jsonl` 额外记录两行梯度范数。新增两行共 8192 个参数，adapter 中单独保存约 33 KB；加上梯度和 Adam 状态，额外训练内存约 128 KiB，不会明显改变整体显存需求。

### 本轮验证范围

正、负样本均完成一次真实优化器更新，并已保存新 token 的两行权重与 tokenizer。单步正样本 adapter 成功重新加载到 benchmark，完成 16 秒原生流式输入；该次自然生成保持沉默，没有生成调用，这不能作为学会新协议的效果证明。

另在真实原生 decoder 中做了明确标记的强制 token 传输检查：分隔符位于第二个 chunk 的第一个生成位置，原生 `text` 的确会删除它，而新解析器仍从 ID 流恢复了用户回复和完整的跨 chunk 调用。检查产物为 `train/outputs/agent_token_probe.json`，其中 `forced_token_transport=true`；这是传输/解析验证，不是模型能力得分。

## 全参数 SFT

`training_method: full_sft` 解冻完整 LLM、输入/输出 embedding、视觉编码器 `vpm`、视觉连接层 `resampler`、音频编码器 `apm` 和音频连接层。包括新增 `<|agent|>` 在内的所有词表行一起更新，不使用 LoRA 单行包装。SFT 仍选每条 QA 最早一个 response，保留已有的三组损失和可选 silence 监督。

全参路径缓存的是 CPU 上的图像/音频预处理结果，编码器输出每次重新计算并保留梯度。训练调用原生 prefill 的可微函数体，不修改模型源文件；流式音频 KV 在同一轨迹中保留计算图，每条轨迹结束后释放。LLM/视觉支持梯度检查点；音频编码器使用有状态 KV，不对它启用不安全的 checkpoint 重算。

先在独立训练环境安装本文件前述依赖及 `deepspeed==0.16.9`。默认 `zero3_cpu.json` 使用 ZeRO-3、BF16、CPU 上的 FP32 master weights 和 AdamW 状态，参数留在 GPU。它使用 PyTorch AdamW，不要求编译 CPUAdam。全参配置单步只处理一个逻辑 QA，梯度累积默认为 1；学习率显式设为 `1e-6`。

```bash
train/.venv/bin/python -m train --config train/configs/minicpmo_full_sft.json prepare \
  --data path/to/all_qa.json --output train/prepared/full_sft --max-samples 12

CUDA_VISIBLE_DEVICES=6 train/.venv/bin/python -m train \
  --config train/configs/minicpmo_full_sft.json profile \
  --prepared train/prepared/full_sft --output train/outputs/full_sft_profile

CUDA_VISIBLE_DEVICES=6,7 train/.venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m train --config train/configs/minicpmo_full_sft.json run \
  --prepared train/prepared/full_sft --output train/outputs/full_sft --max-steps 10
```

多卡用于分片模型、梯度和优化器状态。由于原生媒体路径会条件调用模块，各 rank 同步处理相同 QA；GRPO token 由 rank 0 广播，保证 ZeRO collective 顺序一致。增加卡数可降低每卡状态内存；逻辑 batch 不随卡数增加。本版不提供各 rank 独立采样的数据并行。

全参状态估算：

```bash
train/.venv/bin/python -m train --config train/configs/minicpmo_full_sft.json estimate
```

约 90.24 亿参数，对应 BF16 权重约 16.81 GiB、梯度约 16.81 GiB，CPU FP32 master/Adam 状态约 100.85 GiB；还需激活、通信、KV 和保存时的临时内存。单卡卸载可运行，但 CPU 内存和磁盘要求高。一次完整优化器 checkpoint 实测约 101 GiB，导出的 BF16 模型约 17 GiB。`profile` 默认也保存 checkpoint；仅做梯度/显存检查时可在独立配置中显式设 `save_profile_checkpoint: false`。

## 完整流式 GRPO：本地规则奖励

```bash
train/.venv/bin/python -m train --config train/configs/minicpmo_full_grpo.json prepare \
  --data path/to/all_qa.json --output train/prepared/full_grpo --max-samples 12

CUDA_VISIBLE_DEVICES=6 train/.venv/bin/python -m train \
  --config train/configs/minicpmo_full_grpo.json grpo \
  --prepared train/prepared/full_grpo --output train/outputs/full_grpo --max-steps 2
```

GRPO 必须重新 prepare：仅接收全部可评分参考窗口都在观察区间内的 QA，默认从视频 0 秒观察到 32 秒；视频较短时观察到最后完整秒，并检查窗口完整性。准备阶段的排除原因写入 manifest。没有工具返回输入，参考回复不会出现在模型提示词里。

同一 QA 采样 4 条完整 listen/speak 轨迹，输入完该秒媒体后才生成该秒内容。记录原始 token、时间、动作、采样概率与奖励，再用可微媒体前向重放。GRPO 按原生 decoder.feed 的边界逐段前向并保留可微 LLM KV；BF16 整段重算可能产生足以影响策略比率的误差。只对真实采样动作计算策略损失，屏蔽媒体、问题、问题前强制 listen、预算触发的 chunk 结束和 unit 结束标记。

默认 GRPO 配置显式启用 `grpo_activation_offload: true`，通过 PyTorch saved-tensor hooks 将反向所需激活存入 CPU，缓解可微 KV 的显存压力；代价是主存占用和 PCIe 传输。同一参数/view 在一条轨迹内只保留一份 CPU 副本，避免逐 token 重复复制整套权重。`grpo_activation_cpu_budget_gib` 默认 128，达到上限即明确报错，应缩短窗口或根据主存容量调整。短窗口可显式关闭卸载并测量吞吐。32 秒/8192 token 仍需按实际数据 profile，不能将短程验证的显存当作长轨迹保证。

采样显式使用 temperature=1、无 top-k/top-p、无重复惩罚和 listen 概率缩放。保留原生 EOS 两阶段采样分布、禁用 token 列表及未结束轮次的 listen→tts_bos 合并概率。`batch_prompt_prefill: true` 将相同系统 token 一次预填充，避免 ZeRO 下原生逐 token 前缀的开销。更新前检查每条轨迹的原始概率与重放概率，最大差超过 `replay_logprob_tolerance`（默认 0.15 log-prob）即终止，不继续训练。

- 时间：沿用 benchmark 的闭区间一对一匹配，奖励为时间 precision/recall 的 F1，重复输出会降低分数。
- 内容：时间匹配后，以 NFKC/casefold、保留 Unicode 字母数字的字符二元组 F1 比较参考文本；漏答为 0。它衡量参考文本相似度，不能替代语义正确性评测。
- Agent：调用名称相同才比较参数；参数键/类型/列表顺序需对应，字符串用相同文本 F1，其余值精确比较。负样本需有时间匹配的非空回复且整条轨迹没有调用尝试。
- RTD/PR 总奖励为 `0.5×时间 + 0.5×内容`；Agent 为 `0.4×时间 + 0.4×内容 + 0.2×调用`。每个非法、重复或未完成调用扣 0.1。通过 `reward_weights` 覆盖权重。

优势按组内 `(r-mean)/(std+1e-4)` 计算；零方差组跳过并计数。每组只更新一次，clip=0.2；loss 先按轨迹动作平均，再按组平均。`kl_beta=0`，首版不加载参考模型。模型保持 eval 模式进行 GRPO，以关闭 dropout 并保持采样/重放一致；梯度仍会更新所有上述参数。超 token 上限整组排除，不截短轨迹。默认累积为 1；若自行增加累积，排除后留下不完整累积时会显式报错，避免保存无法恢复的半步状态。

结果写入 `trajectories.rankN.jsonl`、`metrics.rankN.jsonl`、`memory_report.rankN.json`。包含分项奖励、零方差/沉默/非法调用、优势、概率误差、ratio、梯度及显存；规则奖励为零时可能连续无更新，日志中的 `optimizer_steps` 才是实际训练步数。

## 全参 checkpoint 恢复、导出和 benchmark

```bash
# 同一配置/数据/卡数恢复到新的输出目录；max-steps 表示累计优化器步数。
CUDA_VISIBLE_DEVICES=6,7 train/.venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m train --config train/configs/minicpmo_full_sft.json run \
  --prepared train/prepared/full_sft --output train/outputs/full_sft_resumed \
  --resume train/outputs/full_sft/checkpoint-final --max-steps 20

# CPU 上合并 ZeRO 权重，保存可用于推理的 HF 模型；需要充足主存和磁盘。
train/.venv/bin/python -m train export \
  --checkpoint train/outputs/full_sft/checkpoint-final --output train/outputs/full_sft_export

CUDA_VISIBLE_DEVICES=6 benchmark/.venv/bin/python -m benchmark run minicpmo \
  --model-path train/outputs/full_sft_export --agent-output-protocol agent-token \
  --data path/to/all_qa.json --output-dir benchmark/results/full_sft_eval
```

checkpoint 保存模型分片、优化器、epoch/样本游标和各 rank 随机状态，并核对配置、数据及提示词指纹。`save_every_steps` 可配置周期保存；默认仅在结束时保存。恢复不能改变卡数或准备数据。导出目录带 `full_training_model.json` 标识，benchmark 会校验 config/index/tokenizer 并将导出身份写入输入协议，避免与基座或其他 checkpoint 混用缓存；该身份不是全部权重字节的校验和。

## 全参短程验证记录（2026-10-16）

以下均使用 A100 80GB、BF16、ZeRO-3、CPU optimizer offload，实际可训练参数为 9,024,099,568。显存为 PyTorch peak allocated，每项只完成一个优化器步，不能据此判断收敛或最终 benchmark 效果。

| 路径 | 输入 / 轨迹 | GPU 峰值 |
| --- | --- | --- |
| 单卡全参 SFT | 1 秒媒体，2564 token，18 个监督 token | 22.50 GiB |
| 双卡全参 SFT | 同一逻辑样本，2 rank 分片 | 16.02 / 16.03 GiB |
| 单卡 GRPO，关闭激活卸载 | 3 秒媒体，4 条轨迹，32 个采样动作 | 69.89 GiB |
| 单卡 GRPO，开启激活卸载 | 3 秒媒体，2 条轨迹，26 个采样动作 | 34.92 GiB |

SFT 和 GRPO 均确认 LLM、视觉、音频及两个连接模块有非零梯度。完整 SFT checkpoint 已保存并导出，对比原始权重确认五个模块都发生参数更新；导出模型通过真实 benchmark loader 在 CPU 加载，新增 token ID 为 151748。微型模型验证 ZeRO checkpoint 恢复后的下一次更新与不中断训练逐值一致。

四轨迹 GRPO 奖励为 `[0, 0, 0, 0.7315]`，最大重放 log-prob 误差约 0.060，平均策略比率约 0.9994，完成了一次实际更新。另一次开启激活卸载的实跑peak allocated 为 34.92 GiB（未卸载的 4 轨迹实跑为 69.89 GiB），CPU 暂存激活峰值约 61.42 GiB；2 条轨迹的一步耗时约 408 秒，说明该方式以主存和传输时间换显存。现在采样后预检和带梯度回放都执行 log-prob 误差阈值检查，超限会在优化器 step 前中止。零方差、时间匹配、负样本调用、可微 KV、CPU 卸载参数去重及 stride 保真另有自动检查。多卡 GRPO、完整 32 秒轨迹和正式长训仍需要独立验证。
