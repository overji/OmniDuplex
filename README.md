# OmniDuplex

本仓库提供从视频生成 OmniInteraction 问答数据的 pipeline：视频依次经过
caption、任务路由、问答生成与校验，最后合并为 `all_qa.json`。

## 下载视频数据集

公开数据集 [overji/OmniInteractionDataset](https://modelscope.cn/datasets/overji/OmniInteractionDataset)
包含 320 个 MP4 视频和两份 metadata JSON。以下命令在仓库根目录执行；把数据集下载到
仓库旁边，避免将大文件放进 Git 工作区：

```bash
python -m pip install modelscope-hub
ms-hub download overji/OmniInteractionDataset \
  --repo-type dataset \
  --local-dir ../OmniInteractionDataset
```

视频位于 `../OmniInteractionDataset/videos/`，metadata 位于数据集根目录及
`videos/` 目录。运行生成流程只需指定视频目录。

## 配置 Gemini 环境

需要 Python、`ffprobe`（来自 FFmpeg）、可访问的 Gemini API，以及支持视频输入和
`generate_content` 的 Gemini 模型。在仓库根目录创建虚拟环境并安装依赖：

```bash
python -m venv .venv-data-generation
source .venv-data-generation/bin/activate
python -m pip install -r data_generation/requirements.txt
python -m pip install -r data_generation/requirements-gemini.txt
ffprobe -version
```

如果缺少 `ffprobe`，请先安装 FFmpeg。然后从
[Google AI Studio](https://aistudio.google.com/apikey) 获取 API key，设置环境变量：

```bash
export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
export GEMINI_MODEL='YOUR_VIDEO_CAPABLE_GEMINI_MODEL_ID'
```

也可以将这两个变量写入仓库根目录的 `.env`，或用 `--env-file` 指向其他 dotenv
文件；不要提交密钥。`GEMINI_MODEL` 必须填写账号可用的具体模型 ID，程序没有
默认 Gemini 模型。

## 运行生成流程

```bash
python data_generation/generate.py \
  --videos-dir ../OmniInteractionDataset/videos \
  --output-dir data_generation/data/pipeline_runs \
  --num-attempts 100 \
  --provider gemini \
  --seed 42
```

`--num-attempts` 是整个运行的问答任务**尝试次数上限**，跳过和失败的尝试也计入；
它不是保证生成成功的样本数。程序递归读取视频目录，只遍历一次，不会循环视频来凑数。
Gemini 会通过 Files API 上传原视频用于各阶段生成，因此运行会产生 API 调用。

每次新运行会在 `--output-dir` 下创建一个 UTC 时间戳目录，并在终端打印实际路径，
例如：

```text
data_generation/data/pipeline_runs/20261013_120000_000000/
├── all_qa.json           # 本次运行合并后的问答数据
├── coverage.json         # 尝试、成功、跳过、失败及覆盖情况
├── api_usage.json        # caption、router、问答的 API 调用与 token 统计
├── .state/manifest.json  # 运行配置和断点续跑状态
└── <video-name>-<hash>/
    ├── caption.json      # 单视频 caption
    ├── route.json        # 单视频任务路由结果
    ├── qa/               # 各任务生成结果
    └── all_qa.json       # 单视频合并后的问答数据
```

视频仍保留在下载目录，不会复制到结果目录。要继续中断的运行，请把
`--output-dir` 改成终端打印的**具体时间戳目录**，保留原视频目录、模型、
`--num-attempts`、`--seed` 等设置，并加上 `--resume`。

其他 API 提供商、参数和数据格式见
[数据生成详细文档](data_generation/docs/README.md)。
