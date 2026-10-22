# OmniDuplex

本仓库提供从视频生成 OmniInteraction 问答数据的 pipeline：视频依次经过
caption、任务路由、问答生成与校验，最后合并为 `all_qa.json`。

本地模型与云端 Realtime 评测见 [benchmark README](benchmark/README.md)，其中包含
Qwen3.5 Omni Plus / Flash Realtime 和 Gemini 2.5 Flash Live 的环境、密钥及运行配置。
同一文档还提供 Qwen3-Omni / Qwen2.5-Omni 的普通推理 Probe 模式，用正负视频前缀评估回复决策。

## 下载视频数据集

公开数据集 [overji/OmniInteractionDataset](https://modelscope.cn/datasets/overji/OmniInteractionDataset)
包含 A、B 两组（各 160 个 MP4 视频及对应 metadata），以及供生成数据使用的
[train 目录](https://modelscope.cn/datasets/overji/OmniInteractionDataset/tree/master/train)。
以下命令在仓库根目录执行；
只下载要生成的 A 组，并把视频放在仓库旁边，避免将大文件放进 Git 工作区：

```bash
python -m pip install modelscope-hub
ms-hub download overji/OmniInteractionDataset \
  --repo-type dataset \
  --revision master \
  --include 'A/**' \
  --local-dir ../OmniInteractionDataset
```

生成 B 组时把 `A/**` 换成 `B/**`；需要两组时可分别执行两次下载命令，
指向同一个 `--local-dir`。视频位于 `../OmniInteractionDataset/A/videos/` 或
`../OmniInteractionDataset/B/videos/`，每组的 `video_metadata.json` 位于该组目录。
原数据集根目录的 320 个视频路径仍保留，A/B 生成只读取所选分组。

数据集维护者更新 ModelScope 上的文件后，可以用相同的仓库、分组和本地目录重新执行
下载命令来同步该分组的最新版本；例如更新本地 A 组：

```bash
ms-hub download overji/OmniInteractionDataset \
  --repo-type dataset \
  --revision master \
  --include 'A/**' \
  --local-dir ../OmniInteractionDataset
```

更新 B 组时改为 `--include 'B/**'`。如果需要两组都更新，分别执行两次。更新后重新运行
pipeline；它会将所选分组的 metadata 与视频文件核对，不一致时会报错，避免用旧 metadata
或错误分组启动生成。若希望从干净目录重新下载，可先将旧的 `../OmniInteractionDataset`
移走，再执行上述下载命令。

### 下载并检查 train 数据

`train` 当前有 12,455 个视频，视频文件总计约 211 GB，来自 LongVALE（5,253 个）和
FineVideo（7,202 个）。它独立于 A/B 分组。下载前请预留视频和生成结果所需的磁盘空间。
只下载 `train`，保留 metadata、验证报告和来源说明：

```bash
ms-hub download overji/OmniInteractionDataset \
  --repo-type dataset \
  --revision master \
  --include 'train/**' \
  --local-dir ../OmniInteractionDataset
```

更新本地 train 时重新执行相同命令。下载后目录为：

```text
../OmniInteractionDataset/train/
├── videos/                     # UUID 命名的视频
├── video_metadata.json         # 视频清单、来源及原始标识
├── validation_report.json      # 数量、时长和素材校验统计
├── validation_exclusions.json  # 素材筛选排除记录
├── FineVideo_TERMS_OF_USE.md
└── README.md
```

请保留来源 metadata，使用或再分发时查阅下载目录中的 `README.md`、
`FineVideo_TERMS_OF_USE.md` 及其中引用的源数据集条款。

train 使用后文的 `--videos-dir` 导入；这个入口不会像 A/B 的 `--dataset-dir --split`
入口那样自动核对 metadata。因此下载或更新完成后，先执行下面的只读检查，确认没有
缺失、多余或重复的视频记录；数量以本次下载的 metadata 为准：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('../OmniInteractionDataset/train')
metadata = json.loads((root / 'video_metadata.json').read_text(encoding='utf-8'))
listed = [row['file'] for row in metadata['videos']]
video_dir = root / 'videos'
actual = {p.relative_to(video_dir).as_posix()
          for p in video_dir.rglob('*')
          if p.is_file() and p.suffix.lower() in {'.mp4', '.mov', '.mkv', '.webm', '.avi'}}
missing = sorted(set(listed) - actual)
extra = sorted(actual - set(listed))
if (metadata['video_count'] != len(listed) or len(set(listed)) != len(listed)
        or missing or extra):
    raise SystemExit(f'素材与 metadata 不一致：missing={missing[:10]}, extra={extra[:10]}；'
                     '同时检查 video_count 和重复记录，请补全或更新后再生成。')
print(f'train 校验通过：{len(actual)} 个视频')
PY
```

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
  --num-attempts 500 \
  --provider gemini \
  --seed 42
```

生成 B 组时使用 `--split B`，同时将输出目录改为
`data_generation/data/pipeline_runs/B`。例如，要分别为 A/B 各生成最多 500 次 QA 尝试：

```bash
python data_generation/generate.py \
  --dataset-dir ../OmniInteractionDataset --split B \
  --output-dir data_generation/data/pipeline_runs/B \
  --num-attempts 500 --provider gemini --seed 42
```

每次运行会在相应输出目录下建立带时间戳的子目录，终端会打印该路径。`--num-attempts 500`
表示最多进行 500 次 QA 任务尝试，失败、跳过也计入上限；不是保证产出 500 条成功 QA。
若要让 A 或 B 的单次运行最多尝试 500 次，分别执行对应的 `--split A` 或 `--split B`
命令即可。程序先核对所选组的 metadata 与视频文件，
再检查仓库 `data_generation/data` 和指定输出目录中已有的最终 QA 数据。
视频已有至少一条有效、非空的回复数据时会跳过；空 `all_qa.json`、失败及
`skip` 尝试不会让视频被跳过。`--videos-dir` 和 `--video` 仍可用于其他视频来源。

每个视频抽取 1–8 个不同任务类别，候选充足时的期望为 **2.5 类**：1–4 类分别占
15%、40%、30%、13%，5–8 类各占 0.5%。这不是 QA 条数；每个类别仍独立生成
RTD/PR，并以原有概率追加 Agent。路由候选不足、跳过、失败或预算限制会降低实际平均值。
新的分布用于新运行；更改采样配置或实现后请新建运行目录，不覆盖已有 manifest 的选择。

Agent 数据只生成正样本（`sample_type=positive`），不再生成 `no_context` 或
`missing_tool` 负样本。任务必须能用现有可用工具完成，否则返回 `skip`。
兼容参数 `--agent-positive-probability` 仅接受 `1`；合并数据时也会排除历史 Agent 负样本。

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
├── dataset_report.pdf    # pipeline 完成后生成的数据统计图表报告
├── .state/manifest.json  # 运行配置和断点续跑状态
└── <video-name>-<hash>/
    ├── caption.json      # 单视频 caption
    ├── route.json        # 单视频任务路由结果
    ├── qa/               # 各任务生成结果
    └── all_qa.json       # 单视频合并后的问答数据
```

pipeline 结束时会根据最终的 `all_qa.json` 自动生成 `dataset_report.pdf`，包含任务及交互
分布、触发模态、视频长度和提问至首次回复延时等统计。

视频仍保留在下载目录，不会复制到结果目录。要继续中断的运行，请把
`--output-dir` 改成终端打印的**具体时间戳目录**，保留原视频目录、模型、
`--split`、`--num-attempts`、`--seed` 等设置，并加上 `--resume`。续跑使用原
manifest 中的视频集合，不会因其他人后来提交了新 QA 而跳过本次运行未完成的阶段。

### 使用 train 生成数据

先按上面的步骤安装 Gemini 依赖，设置 `GEMINI_API_KEY` 和 `GEMINI_MODEL`，并完成
train 素材检查。然后在仓库根目录运行：

```bash
python data_generation/generate.py \
  --videos-dir ../OmniInteractionDataset/train/videos \
  --output-dir data_generation/data/pipeline_runs/train \
  --num-attempts 500 \
  --provider gemini \
  --seed 42
```

当前 `--dataset-dir --split` 仅支持 A/B，train 请使用 `--videos-dir`，不要添加
`--split train`。程序递归读取视频目录，只遍历一次；`500` 是 QA 任务尝试上限，
失败和 `skip` 也计入，不保证生成 500 条成功 QA。已有有效、非空 QA 的视频会跳过，
判断依据与 A/B 相同。

结果写入 `data_generation/data/pipeline_runs/train/<UTC时间戳>/`。该目录包含最终
`all_qa.json`、`coverage.json`、`api_usage.json` 和完成后生成的 `dataset_report.pdf`；
各视频子目录包含 `caption.json`、`route.json`、`qa/` 及单视频 `all_qa.json`。
原视频仍保存在下载目录中。

继续中断的 train 运行时，将下面的 `<UTC时间戳>` 替换为终端打印的实际目录，
保持原视频路径、模型、尝试上限和 seed 不变：

```bash
python data_generation/generate.py \
  --videos-dir ../OmniInteractionDataset/train/videos \
  --output-dir data_generation/data/pipeline_runs/train/<UTC时间戳> \
  --num-attempts 500 \
  --provider gemini \
  --seed 42 \
  --resume
```

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

## 为 A/B 生成 QA benchmark 并评测

上面的 pipeline 每次运行都会生成可直接作为 benchmark 输入的 `all_qa.json`。使用
终端打印的实际时间戳目录，分别将 A、B 的 QA 文件和对应分组的视频目录传给 benchmark；
不要把 A 的 JSON 配上 B 的视频目录。例如，下面对 A 组结果运行 Qwen Realtime：

```bash
python -m benchmark run qwen_realtime \
  --model qwen3.5-omni-plus-realtime \
  --data data_generation/data/pipeline_runs/A/<A运行时间戳>/all_qa.json \
  --videos-dir ../OmniInteractionDataset/A/videos \
  --output-dir benchmark/results/omni_interaction_A/qwen35_plus
```

B 组使用其运行目录和 `../OmniInteractionDataset/B/videos`，并使用独立的结果目录：

```bash
python -m benchmark run qwen_realtime \
  --model qwen3.5-omni-plus-realtime \
  --data data_generation/data/pipeline_runs/B/<B运行时间戳>/all_qa.json \
  --videos-dir ../OmniInteractionDataset/B/videos \
  --output-dir benchmark/results/omni_interaction_B/qwen35_plus
```

评测 Gemini 2.5 Flash Live 时，将命令中的 `qwen_realtime` 换成 `gemini_live`，并将模型
改为 `gemini-2.5-flash-native-audio-preview-12-2025`；评测 Qwen Flash Realtime 时将模型改为
`qwen3.5-omni-flash-realtime`。先按 [benchmark README](benchmark/README.md) 配置相应 API
依赖、密钥和 Realtime endpoint。`benchmark/results/` 中保存预测、指标、运行配置及逐样本
日志；A、B 应使用不同结果目录，便于单独比较。

其他 API 提供商、参数和数据格式见
[数据生成详细文档](data_generation/docs/README.md)。
