# JEV：先判断开口，再生成内容

MiniCPM-o-4.5 作为独立二分类器，每秒在 `<|silent|>` / `<|response|>` 两个原子 token 间选择。
Qwen2.5-Omni-7B Thinker 作为首个可插拔后端；两个模型分别维护权重、设备和缓存，不加载 TTS。

流程为：分类器观察本秒 → 后端输入同一秒 → silent 时继续积累输入；response 时关闭用户消息、打开 assistant 消息并生成。
Qwen 的输入结束标记是 `<|im_end|>`，后面还必须接 assistant generation prefix；EOS 本身不是生成指令。
生成期间生产者继续把媒体放入队列，只有一个消费者操作模型缓存，生成结束后按时间顺序续接。第一版不支持打断。

## 环境与配置

在仓库根目录执行，建议 Python 3.11 和独立环境：

```bash
python -m venv JEV_train/.venv
JEV_train/.venv/bin/pip install -r JEV_train/requirements.txt
JEV_train/.venv/bin/python -m JEV_train --help
```

需要系统 FFmpeg。固定 Transformers 4.57.3：现有 `benchmark/.venv` 的 4.51.0 没有这里使用的独立 Qwen Thinker 类，不能直接混用该环境。
MiniCPM 原生类已验证可在 4.57.3 导入，但本模块尚未用真实权重执行 GPU 训练或双模型推理。
已有本地权重路径见 `configs/default.json`；模型安装目录仍是 `/remote-home/yushengliao/speech/jimodels/openmodels`。

复制默认配置后按机器修改 `videos_dir`、设备和输出 adapter 路径。所有相对路径以仓库根目录为准；设备号只是配置值，不自动挑卡。
分类器和后端可以配置同一设备，但需自行确保显存充足。当前 benchmark 不会被此模块停止或修改。

两种模式：

- `classifier_only`（默认）：仅训练分类器，后端冻结，用提示词生成内容。
- `separate_sft`：分类器与后端分别 LoRA SFT；没有联合反向传播。后端仅训练文本 Transformer 的 LoRA，媒体编码器冻结。

默认沿用现有 pilot 的 rank=16、alpha=32、学习率 1e-5、两轮、累积四条样本。`max_prefix_seconds=32` 和 `max_sequence_tokens=8192` 是小规模配置，扩大训练时需显式调整；不静默截断。

## 数据准备

默认 `data` 使用 AOI-Dataset 的 `all_qa.json`，`videos_dir` 指向 AOI-Dataset 根目录
（QA 本身含 `videos/`）。下载、更新和媒体校验见[仓库 README](../README.md)。
其他机器修改 `configs/default.json` 的 `data` 与 `videos_dir`；`--data` 可显式覆盖标注来源。


```bash
JEV_train/.venv/bin/python -m JEV_train --config JEV_train/configs/default.json prepare \
  --output JEV_train/prepared/pilot
```

按 UUID 视频划分 train/validation/test。对同一 QA 的全部评分窗口选因果合法的整数触发秒；只要有事件超出 horizon，该 QA 就显式排除，避免把未处理的事件误标 silent。
生成时间的观测规则保持为先输入 `[t-1,t)`，再在 t 决策。问题只能在其时间戳之后的媒体步进入上下文。

每个事件只有一个 response 正例；窗口其余秒忽略。问题输入前忽略，其他窗口外秒按标注完整性假设监督 silent。
训练每个正例至多抽四个负例，优先用一半配额覆盖触发前后 3 秒；验证和测试不抽样。标签之外的媒体仍完整保留。
重叠且同秒触发的事件无法形成唯一目标时记录 `colliding_events` 并排除，不拼出新的参考答案。

Agent 参考只读转换为 `task_start(instruction)`，不改原 JSON，不输入工具返回或总结。准备目录包含三份数据、manifest、原始数据 SHA256、视频划分及 excluded 统计。
数据准备会抽取媒体缓存，所以首次对大视频集 prepare 可能较慢；不加载模型、不占用 GPU。

## 两种训练模式

```bash
# 默认只训练分类器；后端权重不会加载
JEV_train/.venv/bin/python -m JEV_train --config JEV_train/configs/default.json train-classifier \
  --prepared JEV_train/prepared/pilot --output JEV_train/outputs/gate
```

分类器冻结媒体编码器和基座，只训练 LLM LoRA、新增 token 的两行输入 embedding，以及对应两行输出投影。
不生成完整词表 logits；每秒选取媒体结束位置的 hidden state，用二分类交叉熵训练。
历史中的决策 token 行在反向前重新嵌入，避免冻结特征缓存切断其梯度。
每轮验证是参考回复历史下的 teacher-forced 分类指标，不能当作串联系统指标。

要分别训练后端，将配置的 `mode` 改为 `separate_sft`，然后执行：

```bash
JEV_train/.venv/bin/python -m JEV_train --config /path/to/separate_sft.json train-backend \
  --prepared JEV_train/prepared/pilot --output JEV_train/outputs/backend
```

后端完整接收前缀，只在各参考触发点的 assistant 输出上计算 loss；历史回复进入后续上下文，媒体、问题和消息模板均不计目标 loss。
输出格式统一为自然语言，以及可选的：

```text
我来查询替换配件。<|task_start|>查询已观察到的配件的可售替换选项。
```

这里的 `<|task_start|>` 是原词表编码的普通文本分隔串，后面直接为非空 instruction，不输出 JSON；与 train/ 的回复目标一致，不扩展后端词表。无 Agent 时仅输出自然语言；只接受一次分隔符、非空回复和 instruction，不执行工具。

JEV 协议已升级为 `jev_gate_qwen25_task_text_v2`；旧 prepared、分类器和后端 adapter 不自动迁移，需要重新 prepare/训练。gate 仍预测 `<|silent|>` / `<|response|>`，原有两个决策 token 的训练方式不变；只有后端回复格式和相关上下文提示切换。

训练产物包含 adapter、tokenizer/行权重（分类器）、协议和基座身份、配置、数据 manifest 指纹、loss 与验证指标。
输出目录必须是新目录；本实现不恢复 optimizer，不自动运行后续训练。超出上下文限制会记录/报错，不偷偷丢弃媒体。

## 串联评测与流式演示

在配置中将 `classifier_adapter` 设为 `JEV_train/outputs/gate/adapter`；若训练了后端，再填写 `backend_adapter`。
未设置已训练的分类器 adapter 时拒绝评测，避免把随机新增行当作模型结果。

```bash
JEV_train/.venv/bin/python -m JEV_train --config /path/to/evaluation.json evaluate \
  --prepared JEV_train/prepared/pilot --split test --output JEV_train/outputs/evaluation

JEV_train/.venv/bin/python -m JEV_train --config /path/to/evaluation.json stream \
  --prepared JEV_train/prepared/pilot --split test --sample-index 123 \
  --realtime --output JEV_train/outputs/stream123
```

`stream` 使用同一串联流程，`--sample-index` 可选择一条 QA；`--realtime` 按一秒墙钟投递一秒媒体。不加该选项则尽快运行离线队列，queue_depth 不代表实际实时系统负载。
分类阈值默认 0.5，只通过配置显式调整，不用测试集选阈值。

Qwen 后端把每秒帧作为 image、音频作为独立 1 秒 audio 块并附时间戳，使用官方 processor 展开媒体 token；这与原生整段 video/audio 的分块不同。默认 `processor_use_fast: false`，保留 checkpoint 的 slow 图像处理路径。
每包显式计算并平移 MRoPE，持续追加 KV；不逐秒重算完整历史。无音轨时省略音频。
首次真实媒体包后强制比较：相同分包位置编码下，增量缓存与一次性因果前向的最后 logits、KV 长度和 token 数。通过后才允许生成，失败则该 QA 标记 failed。
这个检查验证相同分包语义的一致性，不声称与原始整段视频处理逐位相同。

输出 `predictions.json`、`predictions.jsonl`、`metrics.json`、`report.md`。JSONL 保留原始 sample_index/task/video/question，配置注明 task-start schema，可另外交给现有 benchmark judge；不会自动调用收费 API。
包含逐秒 P/R/F1、AUROC、阶梯积分 AP（auprc）、一对一窗口命中、重复/额外触发、漏窗口、触发延迟及各阶段耗时/积压。失败 QA 保留已完成的部分轨迹，但不混入完整 QA 的主指标。
时间窗口指标使用**逻辑触发时刻 t**；生成墙钟耗时、首 token 延迟另记，不能把它当作实际端到端发声延迟。内容正确性未运行 LLM judge，Agent schema 合法也不等于语义正确。

## 验证范围

```bash
JEV_train/.venv/bin/python -m unittest discover -s JEV_train/tests
```

CPU 测试覆盖标签、数据不变、视频划分、可选调用、队列顺序、媒体恰好输入一次和新增行梯度。
微型原生 Qwen Thinker 测试验证真实注意力/KV/MRoPE、消息切换、冻结参数和文本 LoRA 梯度；媒体编码器输出使用 mock。
尚未验证完整 MiniCPM/Qwen 真实权重的 GPU 训练、模型效果、显存峰值或实时吞吐。实施本模块时没有启动这些实验。
