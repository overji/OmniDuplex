# OmniDuplex

本仓库提供从视频生成 OmniInteraction 问答数据的 pipeline：视频依次经过
caption、任务路由、问答生成与校验，最后合并为 `all_qa.json`。

## 下载视频数据集

公开数据集 [overji/OmniInteractionDataset](https://modelscope.cn/datasets/overji/OmniInteractionDataset)
有 A、B 两组，各含 160 个 MP4 视频及对应 metadata。以下命令在仓库根目录执行；
只下载要生成的 A 组，并把视频放在仓库旁边，避免将大文件放进 Git 工作区：

```bash
python -m pip install modelscope-hub
ms-hub download overji/OmniInteractionDataset \
  --repo-type dataset \
  --include 'A/**' \
  --local-dir ../OmniInteractionDataset
```

生成 B 组时把 `A/**` 换成 `B/**`；需要两组时可分别执行两次下载命令，
指向同一个 `--local-dir`。视频位于 `../OmniInteractionDataset/A/videos/` 或
`../OmniInteractionDataset/B/videos/`，每组的 `video_metadata.json` 位于该组目录。
原数据集根目录的 320 个视频路径仍保留，A/B 生成只读取所选分组。

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
  --dataset-dir ../OmniInteractionDataset --split A \
  --output-dir data_generation/data/pipeline_runs/A \
  --num-attempts 100 \
  --provider gemini \
  --seed 42
```

生成 B 组时使用 `--split B`，同时将输出目录改为
`data_generation/data/pipeline_runs/B`。程序先核对所选组的 metadata 与视频文件，
再检查仓库 `data_generation/data` 和指定输出目录中已有的最终 QA 数据。
视频已有至少一条有效、非空的回复数据时会跳过；空 `all_qa.json`、失败及
`skip` 尝试不会让视频被跳过。`--videos-dir` 和 `--video` 仍可用于其他视频来源。

`--num-attempts` 是整个运行的问答任务**尝试次数上限**，跳过和失败的尝试也计入；
它不是保证生成成功的样本数。程序递归读取视频目录，只遍历一次，不会循环视频来凑数。
Gemini 会通过 Files API 上传原视频用于各阶段生成，因此运行会产生 API 调用。

每次新运行会在 `--output-dir` 下创建一个 UTC 时间戳目录，并在终端打印实际路径，
例如：

```text
data_generation/data/pipeline_runs/A/20261013_120000_000000/
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
`--split`、`--num-attempts`、`--seed` 等设置，并加上 `--resume`。续跑使用原
manifest 中的视频集合，不会因其他人后来提交了新 QA 而跳过本次运行未完成的阶段。

## 从 GitHub 同步生成数据

`data_generation/data` 中的 JSON 数据随 Git 仓库版本化；视频从 ModelScope 下载，
不提交到 GitHub。首次获取仓库用 `git clone`；已有仓库在开始生成前执行：

```bash
git pull --ff-only origin main
```

这会同时拉取最新代码和已提交的生成数据，跳过检查会自动读取这些数据。完成新一轮生成后，
在确认结果与工作区改动后，只提交生成数据及你实际修改的代码/文档：

```bash
git add data_generation/data
git commit -m "Add OmniInteractionDataset A generation results"
git pull --rebase origin main
git push origin main
```

生成 B 组时相应更改提交说明。A/B 使用不同输出目录，便于分别查看和同步运行结果。

其他 API 提供商、参数和数据格式见
[数据生成详细文档](data_generation/docs/README.md)。
