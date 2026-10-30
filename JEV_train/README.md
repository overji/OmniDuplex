# JEV：全参训练二分类器，再由Omni后端生成

MiniCPM-o-4.5每秒在 `<|silent|>` / `<|response|>` 间二分类。Qwen2.5-Omni-7B Thinker独立维护媒体上下文，收到response决策后关闭用户消息、插入assistant prefix并生成。两个模型分别训练，不进行联合反向、不加载TTS。

**仅支持全参SFT，不再支持LoRA训练或adapter加载。** 默认 `classifier_only` 只训练分类器、后端冻结；`separate_sft` 提供分别训练分类器和完整Thinker的两个入口。

## 环境与配置

```bash
python3.11 -m venv JEV_train/.venv
JEV_train/.venv/bin/pip install -r JEV_train/requirements.txt
JEV_train/.venv/bin/python -m JEV_train --help
```

需要FFmpeg和兼容CUDA环境；依赖包括Transformers 4.57.3、DeepSpeed 0.16.9。现有较旧benchmark环境没有独立Thinker类，不能直接替代此环境。模型仍安装在 `/remote-home/yushengliao/speech/jimodels/openmodels`。

[默认配置](configs/default.json)使用AOI-Dataset，下载/更新方法见[仓库README](../README.md)。相对路径按仓库根目录解析。`classifier_checkpoint`、`backend_checkpoint`填写完整HF导出目录，不能填写旧adapter。
现有JEV参数保持32秒、8192 token、学习率1e-5、2轮、累积4条QA、负秒比例4和阈值0.5；未自动继承 `train/` 的300秒设置或1e-6学习率。全参长序列/显存仍需要profile验证，不保证四卡吞吐线性提升。

## 数据与特殊token

```bash
JEV_train/.venv/bin/python -m JEV_train prepare --output JEV_train/prepared/full_sft
```

协议升级为 `jev_gate_qwen25_full_sft_token_v3`。旧prepared和LoRA权重不自动迁移；重新prepare并使用新的输出目录。按视频划分train/validation/test，使用完整媒体前缀；所有事件必须在horizon内，同秒冲突会排除，不丢掉后续事件。

MiniCPM仅注册silent/response决策token，旧token ID保持不变。`<|task_start|>` 不注册，分类器历史中的Agent回复使用已有普通token编码；其输出仍是二分类，不自行输出任务instruction。后端Qwen保持原词表，可见回复格式为 `回复<|task_start|>instruction`，没有Agent时仅自然语言。工具不执行、不输入返回和完成回复、不加入Agent负样本。

问题前和参考窗口内非目标秒mask，窗口外负秒按现有比例抽样；分类器每秒先观察本秒媒体再决策。参考回复只在该秒决策之后作为历史输入，未来目标不可提前泄漏。

## 两个独立的全参训练入口

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 JEV_train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m JEV_train train-classifier --prepared JEV_train/prepared/full_sft \
  --output JEV_train/outputs/gate_full_sft

# 复制配置，将mode改为separate_sft，再训练后端；不会自动接着执行。
CUDA_VISIBLE_DEVICES=0,1,2,3 JEV_train/.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m JEV_train --config /path/to/separate_sft.json train-backend \
  --prepared JEV_train/prepared/full_sft --output JEV_train/outputs/backend_full_sft
```

设备编号是示例，使用前按空闲卡配置。torchrun按LOCAL_RANK选择设备，单进程沿用配置的设备，不自动挑卡或启动后台任务。所有rank处理相同QA，ZeRO-3分片状态并卸载CPU优化器；不是独立样本DP。尾批按实际QA数量平均，不重复数据补齐。

分类器每次前向重新执行可微媒体编码，训练LLM backbone、视觉/音频编码器、连接层、完整输入embedding和二分类头。TTS及不参与分类的生成lm_head冻结。没有冻结embedding缓存或仅训练新增行的包装。二分类loss取每秒媒体后的hidden state，标签和采样口径保持原设置。
Thinker后端训练完整文本/媒体模型，只对各参考assistant回复及结束符计算loss；媒体、问题、模板与已有历史不计目标loss。两个模型无跨模型梯度。

## 保存、显式导出和加载

训练输出完整ZeRO `checkpoint-final/`、配置、loss、排除信息和每轮验证。验证分类指标使用teacher-forced参考回复历史，不等同于串联系统效果。当前JEV不提供优化器续训入口。

```bash
JEV_train/.venv/bin/python -m JEV_train export \
  --checkpoint JEV_train/outputs/gate_full_sft/checkpoint-final \
  --output JEV_train/outputs/gate_export

JEV_train/.venv/bin/python -m JEV_train export \
  --checkpoint JEV_train/outputs/backend_full_sft/checkpoint-final \
  --output JEV_train/outputs/backend_export
```

导出在CPU完成，需充足主存/磁盘。分类器产物是完整MiniCPM、扩展tokenizer、`jev_gate_head.safetensors`和身份metadata；后端产物是完整Thinker、processor/tokenizer与metadata。新增行和分类头都保存，不依赖原LoRA adapter。

在评测配置中指定 `classifier_checkpoint=JEV_train/outputs/gate_export`；如训练了后端，再指定 `backend_checkpoint=JEV_train/outputs/backend_export`。没有已训练分类器时拒绝评测。

```bash
JEV_train/.venv/bin/python -m JEV_train --config /path/to/evaluation.json evaluate \
  --prepared JEV_train/prepared/full_sft --split test --output JEV_train/outputs/evaluation

JEV_train/.venv/bin/python -m JEV_train --config /path/to/evaluation.json stream \
  --prepared JEV_train/prepared/full_sft --split test --sample-index 123 \
  --realtime --output JEV_train/outputs/stream123
```

生产者继续接收媒体、单消费者维护两个缓存；生成结束后按时间续接。首次后端缓存必须与完整前向验证一致，失败会停止生成。评测不使用reference历史，只反馈实际生成回复；不支持打断或自动阈值搜索。

## 检查边界

```bash
JEV_train/.venv/bin/python -m unittest discover -s JEV_train/tests
```

CPU测试覆盖完整分类器媒体/LLM/头梯度、更新后重新编码、Thinker全参梯度和真实微型KV一致、尾批平均、完整分类器导出和特殊token/头加载。测试中的ZeRO合并使用CPU state dict替身，未执行真实大模型GPU训练、ZeRO合并或双模型GPU推理；不能据此承诺显存和耗时。

本轮19项CPU测试使用本机现有的Transformers 4.52.3环境通过；声明依赖4.57.3的完整安装环境尚未重新验证。
