# Omni-InteractScope：AOI 协议泛化实验

本实验比较 MiniCPM-o-4.5 基座与当前全量普通 CE SFT 的 `checkpoint-final`，仅测 realtime。
准备步骤不启动推理、模型导出、Judge、GPU 服务或现有队列。**不测 Omni-DuplexEval，也不测需要 Judge 看视频的指标。**

## 数据下载与准备

固定来源：

- [官方代码](https://github.com/YinsongYan/Omni-InteractScope)，revision `b571e6e42aebaf4dd0c57171dcc72f251c20fd4f`。
- [官方数据](https://huggingface.co/datasets/Darden7/Omni-InteractScope)，revision `b3939d4e9bcb2653301ea5757a64e7d5653912f5`。
- [官方数据获取指引](https://github.com/YinsongYan/Omni-InteractScope/blob/main/docs/data.md#video-access)。

配置集中于 `benchmark/configs/interactscope_aoi.json`。仓库外下载目录为
`/remote-home/yushengliao/speech/jimodels/datasets/Omni-InteractScope`，实际数据根目录为其
`data/Omni-InteractScope` 子目录。配置记录数据版本、代码版本、Python、基座、当前训练身份、
SFT 导出目标、解码参数和本地 Qwen27B Judge 设置，不保存任何凭证。

在仓库根目录执行：

```bash
python -m benchmark external prepare \
  --config benchmark/configs/interactscope_aoi.json \
  --output-dir benchmark/results/interactscope_aoi_prepared \
  --download
```

下载会清除本次调用的代理环境，默认使用 HF 镜像，保留缓存以便续传。数据准备需要
`huggingface_hub`、`PyYAML`、`ffprobe`；推理沿用现有 MiniCPM 环境，不需要安装官方整个模型环境。
已下载时去掉 `--download` 即可；正式下载失败应续传，不能把不完整文件视为成功。

官方有 1,479 条交互、1,790 个目标，其中两条不进入官方评分。托管包包含 207 个自录视频，
**另外 905 个平台视频不在 HF 仓库中**。完整测试需补齐这些媒体；托管视频主要覆盖 T8，
不能将该子集称为全量 T1–T8 实验。

准备时逐项检查现有文件的官方大小、SHA256 和视频解码，生成：

- `inputs/all_qa.json`：可转换记录，保留完整 `source_sample`；无效标注原件仍在源数据中。
- `inputs/runnable_qa.json`：实际支持且通过媒体校验的固定运行集合。
- `inputs/source_mapping.json`：原始索引/ID 到运行索引及每条无法运行的原因。
- `inputs/findings.json`、`inputs/manifest.json`、`inputs/preparation.md`：覆盖、异常、窗口冲突和 AOI 决策网格不可达窗口。
- `inputs/public_video_sources.tsv`：去重后的 905 个平台视频来源、裁剪边界和目标文件名。
- `inputs/missing_video_sources.json`：当前缺失或未通过校验的视频。
- `inputs/disjoint_qa.json`：可运行集合的无窗口冲突诊断子集；默认主结果保留官方窗口冲突。
- `inputs/no_training_overlap_qa.json`：排除通过源 ID/URL 或可用 SHA256 检出的训练视频重合；该检查不是视觉去重。
- `experiment.json`、`run_commands.json`、`comparison.md/json`：固定实验配置、待运行命令及尚未测量的报告。

### 补齐平台视频

来源列表不是下载器。可以按链接获取原视频，依照 `source_start_ms/source_end_ms` 裁剪并保留音轨，
保存到官方指定的 `video` 文件名。**新下载或重编码后即使内容一致，也不保证匹配官方参考 SHA256**；
这些文件不会直接混入严格校验的主集合，不能绕过校验冒充官方媒体。

已有匹配官方清单的完整视频目录时，通过官方导入器校验后导入：

```bash
python -m benchmark external prepare \
  --config benchmark/configs/interactscope_aoi.json \
  --output-dir benchmark/results/interactscope_aoi_full_prepared \
  --public-videos /path/to/reference-video-archive
```

导入器不会覆盖现有文件；同文件系统优先硬链接，否则复制。准备后的集合不可原地换版本，
补齐视频或修改协议后用新输出目录。原始标注、窗口和视频 UUID 不改写。

## AOI 适配规则

- T1–T7：保留 `[trigger−2s, trigger+2s]`；T8-A：原始 valid interval；T8-B/C：
  `[max(query_time, trigger−2s), trigger+2s]`。毫秒只换算一次，不能再次减 source crop 起点。
- 改用 AOI 一对一 onset 匹配；不用官方首答决定成败、两秒输出分组、T8 完成时间戳或 T5 历史滚动时钟。
  T5 也按本实验的 AOI 全片流式规则处理。**这里的分数不能冒充原论文/排行榜分数。**
- 每秒先输入场景媒体再生成，保留完整视频；中途问题在其注释时间插入文本，非整数问题时间会拆开输入单元，
  问题后的下一段媒体仍先于输出。问题时间缺失不能默认设为 0，进入异常清单。
- T8-A 无问题任务从流开始给出官方用户需求，不伪造 user request。T8 其他任务保留用户背景与原请求。
- 采用 AOI 文本问题条件：有公开 `query_text` 的语音问题使用文本，**不额外播放 query_audio**。
  场景音轨正常输入。当前独立 query_image 尚不支持，有附件的记录明确标为 unsupported。
- 所有模型共用任务提示词与配置中的解码参数，不注入答案、trigger_reason、参考时间或未来信息，
  不采用冷却、去重或额外生成限频。
- T1 披露官方全部工具名称和说明，输出明确 `<call:NAME>` 标记；原生显式调用名也可识别。
  工具名称集合必须完全一致；额外/错误名称同样保留并导致失败。普通文字提到工具名不算调用。
  不把工具任务转成 `task_start(instruction)`，不补造参数，不执行工具。

## 显式运行（准备完成后另行执行）

先只检查命令和最终 checkpoint 身份，不占 GPU：

```bash
python -m benchmark external run \
  --config benchmark/results/interactscope_aoi_prepared/experiment.json \
  --model all --dry-run
```

正式运行必须明确 GPU，例如下方 `<GPU_ID_1> <GPU_ID_2>` 替换为当时空闲的卡：

```bash
python -m benchmark external run \
  --config benchmark/results/interactscope_aoi_prepared/experiment.json \
  --model base --gpus <GPU_ID_1> <GPU_ID_2>

python -m benchmark external run \
  --config benchmark/results/interactscope_aoi_prepared/experiment.json \
  --model sft --gpus <GPU_ID_1> <GPU_ID_2>
```

运行前检查卡占用，不停止其他作业。命令在前台监督，周期更新 `comparison.md`；中断时清理自己启动的推理进程。
断点续跑沿用固定配置、数据和协议摘要，不能复用其他版本输出。

SFT 固定对应本配置的当前训练。最终 checkpoint 须通过所有 rank 报告、step 数、训练身份与 checkpoint 完整性校验；
未就绪时拒绝启动，不替换为 pilot/中间 checkpoint。正式 `run --model sft` 才会在需要时调用现有 CPU 导出入口，
并写入 export 完成标记。权重物理目录始终在仓库外。准备及 dry-run 不导出模型。
本配置绑定当前正在训练的 18,315-QA 来源版本（已准备的训练划分 14,812 条、验证划分 3,466 条），
不是窗口冲突清理后的 18,021-QA 新版本；本实验不会重启或改变该训练。

## 文本 Judge 与报告

先按现有 `train/scripts/serve_grpo_judge.py` 指引，显式为 Qwen3.8-27B 分配空闲 GPU 和端口启动服务。
配置参考 `train/configs/judge_qwen27b_service.json`，不要直接沿用其中 GPU 编号而覆盖正在使用的卡。
该服务由用户单独管理，本实验客户端不启动或停止它。

```bash
python -m benchmark external run \
  --config benchmark/results/interactscope_aoi_prepared/experiment.json \
  --model all --stage judge --judge-base-url http://127.0.0.1:<PORT>/v1

python -m benchmark external report \
  --config benchmark/results/interactscope_aoi_prepared/experiment.json
```

文本任务只对时间匹配的当前目标比较参考答案和实际回复，沿用 AOI 0/0.5/1/1.5/2 正确性刻度；
未匹配窗口为 0，Judge 失败保持未定义并允许显式续跑。Judge 不看媒体，也不能独立核验视频事实。
T1 无需请求 LLM，按精确工具名称集合给 0/1 内容信用。

报告将文本与工具任务分成独立面板，分别给时间 R/P/F1、文本 scored-F1 或工具名称正确性加权 F1，
并按原始任务和视频长度细分。重复开口计入输出分母。两模型共同比较只使用都完整完成且提示词/观察时长一致的 QA；
模型单独覆盖单列，不能混用分母。训练重合排除后的共同结果另保存在 JSON。
Judge 支持在 `--judge-base-url` 后传入多个本地服务 URL，在这些副本之间轮询分配请求；
各服务的 GPU 仍需显式独立配置。文本请求与工具计分不会调用视频 Judge。
尚未推理/评分的结果显示缺失，不能填造 0 或宣称清洗/训练收益。

## 从平台重新获取视频，不检查视频 SHA256

用户明确不要求官方视频字节一致时，可使用前台下载入口：

```bash
python -m benchmark external acquire --config benchmark/configs/interactscope_aoi.json
# 提供本地 Netscape cookies.txt 时：
python -m benchmark external acquire --config benchmark/configs/interactscope_aoi.json \
  --sources douyin --cookies /path/to/cookies.txt
```

该入口使用无代理 yt-dlp 获取可访问的视频；小红书尝试读取公开页面明确提供的视频地址。
原始来源和区间保持记录，视频按给定起止毫秒精确裁剪，保留音轨及画面尺寸，使用配置中的
CPU 编码参数重新封装。不会覆盖已存在的视频。日志保存在仓库外下载根目录的
`public_acquisition/current.json` 和 `attempts.jsonl`；不会输出或保存 cookie 内容到报告。

平台需要 cookies、来源返回不可用或裁剪失败时会明确记录。抖音的公共 cookie 前置条件失败时，
其余同平台记录标为 blocked，区别于已经逐条尝试下载的 failed；不反复请求相同错误。
`status=incomplete` 表示仍有未获取素材，不代表下载任务还在后台运行。

下载后准备新的非 SHA 校验实验版本：

```bash
python -m benchmark external prepare \
  --config benchmark/configs/interactscope_aoi.json \
  --output-dir benchmark/results/interactscope_aoi_reacquired \
  --skip-media-sha256
```

这个显式选项仅取消视频的原始大小/SHA256 匹配要求，改做视频解码及下载时的时长、音轨检查。
标注和配置仍保留版本指纹。新集合标明 `media_verification=decode`，不冒充官方参考视频的字节副本。
`media_manifest.json` 固定实际素材的大小与修改时间；正式运行前检查素材是否变更，防止混用轨迹。
旧的严格校验配置和实验快照不会被原地修改。

### 正常浏览器生成抖音请求

若 Cookie 文件可读但 yt-dlp 的旧接口仍报 fresh cookies，可使用浏览器复制的 GET detail 请求作为
会话输入。命令只读取 User-Agent 和 Cookie，不执行 cURL shell，也不把某个视频的旧签名套到其他 ID。
隔离的 Playwright 浏览器按正常页面流程为每条视频生成新请求；不自动解决验证码。

本机隔离工具位于配置中的 `public_acquisition.browser_python` 和 `browser_cache`，未修改训练环境。
换机器时需在独立虚拟环境安装 Playwright/Chromium，并更新这两个路径。
请求文件含会话凭证，应保存在忽略的本地目录，不能提交 Git。

```bash
# 先验证少量素材；命令保持前台运行。
python -m benchmark external acquire --config benchmark/configs/interactscope_aoi.json \
  --sources douyin --browser-request /path/to/copied_request.txt --max-items 3

# 去掉数量限制，继续所有尚未获取的抖音素材。
python -m benchmark external acquire --config benchmark/configs/interactscope_aoi.json \
  --sources douyin --browser-request /path/to/copied_request.txt
```

仅当标注范围与整个源文件一致（起点为 0，时长差不超过配置的 0.02 秒）时直接保留原文件，
否则按原有配置精确裁剪并转码。新文件不检查 SHA256；原始文件不会被覆盖。
`browser_status.json` 记录本轮 PID、选中数量及完成数量；`current.json` 区分已获取、未尝试和失败。
后台运行仍需用户明确授权，准备或少量验证不会自动启动后台整批任务。
