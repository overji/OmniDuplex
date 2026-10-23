# Realtime timing benchmark

云端 Qwen3.5 Omni Plus / Flash Realtime 和 Gemini 2.5 Flash Live 的安装、API key、
endpoint 与运行命令见[云端 Realtime 配置](#云端-realtimeqwen35-omni-与-gemini-25-flash-live)。
普通 Qwen3-Omni / Qwen2.5-Omni 的前缀二分类评测见
[Probe 模式](#probe-模式普通-omni-llm-的回复决策评测)。

## 全量数据集与闭源 Realtime 快速开始

公开评测数据集：[overji/AOI-Bench](https://modelscope.cn/datasets/overji/AOI-Bench)。
包含 **2,107 条 QA、852 个视频**（RTD 706 / PR 678 / Agent 723），视频约 17.67 GiB。
标注与原 2,110 条合并集的 2,107 条一致；仅去掉缺失素材 `video_103.mp4` 的 3 条 QA，
没有调整参考答案或评分窗口。训练另用 `overji/AOI-Dataset`，不要混用两份数据。

### 1. 下载数据并配置视频路径

从当前 OmniDuplex 仓库根目录执行。完整的下载、`.env` 认证、直连和更新流程见
[仓库 README](../README.md) 的 AOI 数据章节。无需切换到数据下载目录：

```bash
export TRAIN_ROOT=/remote-home/yushengliao/speech/jimodels/datasets/AOI-Dataset
export BENCH_ROOT=/remote-home/yushengliao/speech/jimodels/datasets/AOI-Bench
# 在已安装 modelscope-hub 的环境中执行；需要认证时设置 MODELSCOPE_API_TOKEN。
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  ms-hub download overji/AOI-Bench --repo-type dataset --revision master \
    --local-dir "$BENCH_ROOT"

# 同时下载 AOI-Dataset 后检查两份媒体，并建立兼容来源路径的本地别名。
python benchmark/scripts/prepare_aoi_data.py \
  --dataset-dir "$TRAIN_ROOT" --bench-dir "$BENCH_ROOT"
export BENCH_DATA="$BENCH_ROOT/all_qa.json"
export BENCH_VIDEOS="$PWD/benchmark/cache/aoi_bench_videos"
```

更新时重复相同下载命令与准备脚本。AOI-Bench 视频平铺在 `videos/`，标注仍保留部分
`A/videos/`、`B/videos/` 路径，所以不要直接把平铺目录传给 `--videos-dir`。
别名目录通过 `video_manifest.json` 解析这些引用，不修改原始 QA 或视频。
`benchmark` CLI 默认数据为上述 AOI-Bench 标注，默认媒体为别名目录；支持命令行覆盖。
`benchmark/configs/suites/full_2110.json` 也已切换到该发布集（文件名为历史命名，实际为 2,107 条）。
准备 suite 时会在其输出目录建立独立别名。迁移数据目录后需重新准备并使用新结果目录。

### 2. 安装 CPU 运行环境并配置密钥

Python 3.10+；系统需安装 FFmpeg（含 `ffprobe`）和 libsndfile。

```bash
python3 -m venv benchmark/.venv-realtime
source benchmark/.venv-realtime/bin/activate
python -m pip install -r benchmark/requirements-realtime.txt
ffmpeg -version
ffprobe -version

export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'
export DASHSCOPE_REALTIME_URL='wss://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime'
export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
```

Qwen 的 Workspace ID、密钥与 endpoint 必须属于相同地区。新加坡 endpoint 为
`wss://YOUR_WORKSPACE_ID.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime`。
Realtime adapter **只读取进程环境，不自动加载 `.env`**。若自行创建仅含可信 shell
赋值语句的 `.env`，可用 `set -a; source .env; set +a` 导出变量；不要提交真实密钥。
ModelScope token 用于仓库上传认证，不是模型推理 API key，也不需要写入下载包。

### 3. 运行三个闭源 Realtime 模型

从包含 `benchmark/` 的目录执行，三个模型各使用独立结果目录。以下为当前代码配置的模型 ID；
云端 adapter 尚未完成真实 API 联调，账号需具备相应模型权限。接口不可用时直接报错，
不会自动替换模型；可以通过 `--model` 显式指定账号实际支持的 ID。

```bash
python -m benchmark run qwen_realtime \
  --model qwen3.5-omni-plus-realtime \
  --data "$BENCH_DATA" --videos-dir "$BENCH_VIDEOS" \
  --output-dir benchmark/results/qwen35_omni_plus_realtime_full

python -m benchmark run qwen_realtime \
  --model qwen3.5-omni-flash-realtime \
  --data "$BENCH_DATA" --videos-dir "$BENCH_VIDEOS" \
  --output-dir benchmark/results/qwen35_omni_flash_realtime_full

python -m benchmark run gemini_live \
  --model gemini-2.5-flash-native-audio-preview-12-2025 \
  --data "$BENCH_DATA" --videos-dir "$BENCH_VIDEOS" \
  --output-dir benchmark/results/gemini25_flash_live_full

python -m benchmark compare \
  benchmark/results/qwen35_omni_plus_realtime_full \
  benchmark/results/qwen35_omni_flash_realtime_full \
  benchmark/results/gemini25_flash_live_full \
  --output benchmark/results/realtime_full_comparison
```

命令以前台方式运行；全量运行不要设置 `--max-samples` 或 `--max-seconds`。
重复相同命令会跳过已保存预测，未完成且未保存预测的 QA 从头开始；
改变数据、配置或模型需另选结果目录。输出包括 `predictions.jsonl`、`metrics.json`、
实际提示词及原始 realtime 事件。截短运行会标记 partial，不能作为全量结果。
运行协议为逐秒手动分轮，输入不快于 1 FPS；该设置不是原生 VAD 连续发声评测。
详细输入、Agent 调用和时间评分口径见下方云端 Realtime 章节。

### 4. 可选：DeepSeek 内容评分

推理完成后再运行独立 judge；该步骤会向 DeepSeek 发送问题、参考标注及模型回复，不上传视频。
使用进程环境中的 DeepSeek key；不要复用其他平台的 `OPENAI_API_KEY`。

```bash
export OPENAI_API_KEY='YOUR_DEEPSEEK_API_KEY'
python -m benchmark judge \
  benchmark/results/qwen35_omni_plus_realtime_full \
  benchmark/results/qwen35_omni_flash_realtime_full \
  benchmark/results/gemini25_flash_live_full \
  --data "$BENCH_DATA" --output-dir benchmark/results/realtime_full_judge \
  --provider openai --model deepseek-flash --base-url https://api.deepseek.com
```

内容原始分数为 0–2，除以 2 可归一化为 0–1；全窗口内容均分将未命中记零。
API 失败不记零，汇总显示 incomplete；重复原命令仅补跑失败记录。

### 5. 三个模型运行完成后，需要交回什么

请把以下产物打包为 **`realtime_full_submission.tar.gz`** 交回，用于复算时间指标、
运行内容 judge 和排查异常。三个模型分别保留完整结果目录，不要只交最终分数截图。

| 交付内容 | 文件 / 目录 | 用途 |
| --- | --- | --- |
| 三个模型的完整结果目录（必需） | `benchmark/results/qwen35_omni_plus_realtime_full/`、`qwen35_omni_flash_realtime_full/`、`gemini25_flash_live_full/` | 保留下表所列逐样本预测、配置、提示词和原始事件 |
| 时间指标比较表（必需） | `benchmark/results/realtime_full_comparison.csv`、`.md` | 由第 3 步 `compare` 命令生成 |
| 实际使用的标注和发布清单（必需） | 下方命令复制为 `realtime_full_all_qa.json`、`realtime_full_dataset_manifest.json` | 确认数据版本与样本索引，后续按同一份标注复算 |
| 内容评分目录（做了第 4 步时交回） | `benchmark/results/realtime_full_judge/` | 包含逐条 judgments、judge 配置、指标、报告、API 用量和模型比较表 |
| 异常信息（有异常时交回） | 手动保存的报错文本或终端日志 | 说明模型名、未完成进度、报错；不要包含真实 API key |

每个模型的结果目录内，至少要保留：

| 文件 | 内容 |
| --- | --- |
| `predictions.jsonl` | 每条 QA 的 `sample_index`、完成状态、回复文本、回复时间等；内容 judge 和复算必须使用 |
| `metrics.json` | 当前总体、交互类型、能力分类与逐样本时间指标 |
| `run_config.json` | 实际模型 ID、endpoint、数据/提示词哈希及运行参数；只记录密钥变量名 |
| `sample_system_prompts.jsonl` | 实际输入的系统提示词；续跑时可能有同一 QA 的多次记录，保留即可 |
| `sample_*.realtime.jsonl` | 各 QA 的逐秒输入时间、输出文本/转写、协议事件与 API 返回用量（若有）；原样保留 |

**完整结果判据：** 每个模型的 `predictions.jsonl` 应包含 **2,107 条**预测，
`sample_index` 为不重复的 `0–2106`，所有 `status` 都为 `complete`；
`metrics.json` 的 `evaluated_samples` 和 `total_benchmark_samples` 均为 **2107**。
`run_config.json` 的 `data_sha256` 应与实际标注的 SHA-256 一致，
全量命令的 `max_samples`、`max_seconds` 均为 0。
没有回复（`events: []`）也是有效预测，不能删掉或用参考答案补写。
仅看分数或 JSONL 行数不足以确认跑完：截短样本可能也已写入预测，因此必须检查完成状态。

如果中途失败，先重复原命令续跑；若暂时无法完成，也可以交回现有目录并标明未完成，
不要把 partial 或缺失样本的结果作为完整评分。内容 judge 是可选步骤，未运行时交回三个
模型的预测及配套文件即可，后续可以据此补做内容评分。

从包含 `benchmark/` 的目录执行以下打包命令；运行前确认第 3 步比较表已生成：

```bash
cp "$BENCH_DATA" benchmark/results/realtime_full_all_qa.json
cp "$BENCH_ROOT/manifest.json" benchmark/results/realtime_full_dataset_manifest.json

set -- \
  benchmark/results/qwen35_omni_plus_realtime_full \
  benchmark/results/qwen35_omni_flash_realtime_full \
  benchmark/results/gemini25_flash_live_full \
  benchmark/results/realtime_full_comparison.csv \
  benchmark/results/realtime_full_comparison.md \
  benchmark/results/realtime_full_all_qa.json \
  benchmark/results/realtime_full_dataset_manifest.json
if [ -d benchmark/results/realtime_full_judge ]; then
  set -- "$@" benchmark/results/realtime_full_judge
fi
tar -czf benchmark/results/realtime_full_submission.tar.gz "$@"
```

最终交回 **`benchmark/results/realtime_full_submission.tar.gz`**。
若有额外报错日志，请一并提供并标明对应模型。无需重复打包原始视频、媒体缓存或虚拟环境；
不要打包 `.env`、API key 或 ModelScope token。

## Code structure and commands

Run commands from the repository root with the appropriate model's Python environment.
The former standalone `run_*.py`, collection, comparison and download scripts have
been replaced by a single entrypoint:

```bash
python -m benchmark --help
python -m benchmark run minicpmo --help
python -m benchmark compare benchmark/results/data_parallel_4567_20261007/minicpmo benchmark/results/remaining_4567_20261008T022841Z/joyai
python -m benchmark download --repo openbmb/MiniCPM-o-4_5 --revision REVISION \
  --model-dir /remote-home/yushengliao/speech/jimodels/openmodels/MiniCPM-o-4.5
```

| Layer | Responsibility |
| --- | --- |
| `src/config/` | Stable paths, shared CLI defaults, pinned model settings and service defaults |
| `configs/` | Model/data manifests and the shared system prompt |
| `tests/` | CPU-only regression tests and simulated worker tests |
| `src/cli/` | Command dispatch and all command/model argument definitions |
| `src/runtime/` | Sample selection, resume, model lifecycle, prompt/raw/prediction artifacts and per-sample metric updates |
| `src/adapters/` | Native model loading, streaming protocols, preprocessing and response extraction |
| `src/data/` | JSON helpers, media cache, prompts, timestamps, result types and weight download |
| `src/evaluation/` | Window matching, grouped metrics, shard collection and Markdown/CSV reports |

Adapters expose `load(args)`, `run_config(args, base_prompt)` and
`run_sample(args, state, context) -> SampleResult`. The execution layer supplies
the model-facing sample without reference annotations. Adapters use
`context.build_prompt(native_protocol)` to construct and log the actual system
prompt and `context.open_raw(suffix)` to save their native raw output. The
execution layer assembles the existing prediction format and updates metrics.
To add a model, add its adapter and CLI options, then register its name in the CLI.
Model and download dependencies are imported only by the selected command;
help, collection and comparison work without GPU libraries.

The package entrypoints remain at the benchmark root; implementation modules live
under `src/`, with existing `benchmark.cli`, `benchmark.runtime`, etc. imports
preserved. No package installation is required in the model environments.

`src/config/defaults.py` owns CLI defaults, `models.py` owns pinned metadata and
native inference settings, and `services.py` owns JoyAI service defaults. Explicit
CLI arguments and service configuration overrides retain their existing precedence.
Inference settings and scoring behavior are unchanged. Operational data, caches and
result artifacts retain their existing locations. The default prompt and manifests
have moved into `configs/`; use a new output directory for experiments after this
migration. Pre-migration runs are retained for analysis, without resume compatibility.
Generated run, queue and service configuration snapshots remain under `results/`.
Historical plain `.log` files were removed; predictions, raw outputs, prompt records,
metrics and diagnostics were retained. Future runs still produce normal service and
worker logs.

MiniCPM applies its sample limit to the selected shard before skipping existing predictions; the
other models apply it to remaining samples. MiniCPM rejects changed run
configuration, while the other adapters retain their existing overwrite policy.
Recorded partial predictions continue to count as completed for resume, and
their metrics still use the sample's full reference windows. Interrupted samples
without a prediction restart and may append another prompt-log entry.

The working checkout currently defaults to 33 positive-only QA; the following counts describe the earlier 44-QA snapshot, not the full release.

Historical small-sample data (no longer the CLI default): `all_qa.json`, derived from
`data_generation/data/qwen3.8_omni_flash_44/all_qa.json` with an explicit Agent-window
migration. `configs/manifests/data_manifest.json` records the source hash, current
hash, backup and migration ledger. The original snapshot contained 44 samples, 189 reference responses,
and 173 responses with timing windows. Its historical scoring used 157 of those windows:
Agent positives score only assistant responses calling `agent_call`; tool results
and completion replies are excluded, even when they have windows. Their full
traces remain in the data, but neither tools nor reference assistant text are
provided as model inputs.
Agent negative refusal responses (`missing_tool` / `no_context`) remain scored.
All predicted onsets still count toward timing precision and extra predictions.
`excluded_agent_non_trigger_responses` reports excluded Agent tool and completion references; this
count can overlap `excluded_reference_responses_without_windows`.

Future Agent generation gives completion replies natural PR-style windows.
Refusals use `[question_time, question_time + 3]`. The existing 44-sample benchmark
was migrated separately: 16 completion references use `[tool time + 1, tool time + 3]`
and their recommended timestamp is the window start; 11 refusals use the new
three-second window; one zero-width call window was extended to three seconds.
Other reference windows, question text and recorded tool results are unchanged.
Completion windows remain available in the dataset but are not evaluated by the
current benchmark protocol.

The migration is explicit and idempotent:

```bash
python -m benchmark migrate-agent-windows --dry-run
python -m benchmark migrate-agent-windows
```

It backs up the dataset and manifest under `results/` before replacement, records
every changed reference, and rolls back file replacements on write failure.
Original generation files, experiment input snapshots, predictions and metrics
are not rewritten. Existing model results were not rescored for this migration;
historical reports retain their original 157-window evaluation.

Only the user request, available capability/schema context and observed media
are supplied to the model. Recorded Agent results are not extracted or injected,
and no tool-result input logs are produced. MiniCPM, MMDuet2 and ROMA use their existing
end-of-interval clocks. For JoyAI and LiveStar,
output clocks use the frame interval end (`frame time + 1`, capped at video
duration) for every task; their original frame clocks remain in raw records.
User requests are text prefills, never standalone generation
triggers: generation follows the current second's media input. MiniCPM and ROMA
consume audiovisual input; MMDuet2, LiveStar and JoyAI consume their native visual
inputs and do not receive audio. Native speak/listen policies remain active.

The input protocol is recorded as `agent_input_protocol` in run and parallel
configuration as `post_media_generation_without_agent_results_v3`. Use new output
directories: runs that supplied Agent results cannot be resumed as this protocol.
Saved historical results are not rewritten or rescored automatically.

The model is installed at
`/remote-home/yushengliao/speech/jimodels/openmodels/MiniCPM-o-4.5`.
`configs/manifests/model_manifest.json` records the exact Hugging Face revision and verified file hashes.
The loader creates a cache symlink named `MiniCPM_o_4_5` because Transformers 4.51
cannot import remote code from a local folder containing a dot.

## Run

### Data parallel

One independently loaded instance runs on each selected GPU. Run the supervisor
with any Python that can import the benchmark; `--worker-python` selects the
model's own environment:

```bash
python -m benchmark parallel minicpmo --gpus 4 5 6 7 \
  --worker-python benchmark/.venv/bin/python \
  --model-path /remote-home/yushengliao/speech/jimodels/openmodels/MiniCPM-o-4.5 \
  --output-dir benchmark/results/minicpmo_parallel
```

The global candidate list is fixed before dividing it round-robin. `--max-samples`
limits the entire parallel run, not each worker. MiniCPM's `--sample-start` and
`--sample-step` select the candidate list first. Each worker sees only its GPU as
`cuda:0`, while keeping the original sample index and seed. Do not point parallel
runs at existing single-instance result directories.

`parallel_plan.json` stores immutable arguments, GPUs, dataset/prompt hashes and
sample assignments. Repeating the same command resumes those assignments;
configuration changes require another output directory. `worker_N/` stores each
instance's predictions, prompts, raw decisions and `worker.log`. The supervisor
refreshes root predictions, metrics, collection status and report every 10 seconds,
using complete JSONL records and atomic replacements. `parallel_status.json`
contains PIDs, exit codes, missing/partial samples and failure details. Failed
workers do not stop other shards. Interrupted records without a terminating
newline are discarded only when their worker resumes. Shared video extraction
uses a per-video process lock.

JoyAI additionally requires `--joyai-services CONFIG.json`. The configuration
specifies `python`, `code`, `main_model_path`, `summary_model_path`, and optional
`port_base`, `startup_timeout_seconds`, `main_memory_utilization`,
`summary_memory_utilization` and `environment`. Four independent official
main/summary/adapter stacks are launched, with three unique ports per worker.
Main and summary memory budgets default to 0.65 and 0.25; all native generation,
context and memory settings remain unchanged. Health endpoints and real inference
requests must pass before workers start. The supervisor owns and cleans up only
its own process groups.

The GPU 4–7 experiment queue in `results/data_parallel_4567_20261007/` uses
the pinned official vLLM `0.22.0+cu129` wheel with Torch `2.11.0+cu129`. Its
summary-memory budget is explicitly 0.27: the preflight at 0.25 provided only
8.55 GiB KV cache, below the 9 GiB required by the unchanged 65536-token summary
context. Main-model budget remains 0.65. The environment lock and service configuration snapshots remain alongside that
queue's configuration; historical preflight `.log` files were removed during cleanup.

Sequential experiment queues use `python -m benchmark queue --config CONFIG.json`.
The JSON contains `output_dir`, `gpus`, and ordered `jobs`, each with `model`,
`python` (worker interpreter), and `arguments` (a list of CLI tokens). The queue
runs one four-GPU model at a time, records each exit, continues after failures,
and creates a final comparison for models with predictions. `queue_status.json`
records the queue PID and model supervisor PIDs. Commands run in the foreground
unless background execution is explicitly requested. Signal the queue PID with
SIGTERM to stop its active supervisor, which then cleans up workers and services.

Use an isolated environment with `requirements.txt`. This checkout has
`benchmark/.venv` (Torch 2.6.0 / Transformers 4.51.0).

```bash
CUDA_VISIBLE_DEVICES=2 benchmark/.venv/bin/python -m benchmark run minicpmo \
  --model-path /remote-home/yushengliao/speech/jimodels/openmodels/MiniCPM-o-4.5 \
  --sliding-window-mode off
```

Commands run in the foreground. `--max-samples 1 --max-seconds 10 --output-dir
benchmark/results/smoke_off` is a short interface check, not a full benchmark score.
Completed samples resume automatically when the command is repeated with the same
configuration. Interrupted samples restart; completed outputs are preserved.
Use another output directory when changing configuration.

## Protocol

Uses the official native full-duplex API:
`as_duplex`, `prepare`, `streaming_prefill`, `streaming_generate`.
The model receives one audiovisual chunk per second: full-length mono audio at
16 kHz and the latest available approximately 1 Hz video frame. Frame timestamps
come from ffmpeg; no frame at or after the chunk endpoint is supplied. The last
short audio chunk is zero-padded. Video is processed chronologically, without
wall-clock sleeps. No 64-frame video truncation is applied.

The question is injected at its actual `question_time`, as a separate text unit.
Earlier chunks are forced to listen. The system prompt is `configs/prompts/system_prompt.txt`.
Reference replies, evidence, triggers and scoring windows are never passed to the
model. Full context is retained (`sliding-window-mode off`, the official default)
so the user request is not evicted. Sampling keeps the official duplex defaults
(temperature 0.7, top-k 100, top-p 0.8, 20 speak tokens per chunk); seed is
42 + sample index. Configuration and prompt hash are saved per run.

Unused speech synthesis weights are kept on CPU (`--unused-tts-device cpu`) to
leave GPU memory for full video context. This does not change text decoding.
Speech synthesis output is disabled (`generate_audio=False`); the native
listen/speak decision and streamed response text are used. This is an offline
streaming evaluation, not a measurement of audible speech latency. Tools are not
executed. Agent samples receive available tool descriptions and application
context and can announce grounded delegation or explain limitations.

For every Agent sample, the actual system message also contains the available
tool descriptions and the supplied `agent_call` schema. Unavailable tools are
omitted. `missing_tool` means the frontend exposes tools but not one the request
requires; `no_context` means the frontend exposes no tools or function schema.
The question itself is still injected at `question_time`.

## Model runs

Each model has its own adapter and result directory. The model adapters preserve
their native silence/trigger behavior and write `run_config.json`, the exact
per-sample system prompts, raw decisions, `predictions.jsonl`, and `metrics.json`.
Pin records for downloaded model weights are stored as `configs/manifests/model_manifest_*.json`.
All model weights belong under `/remote-home/yushengliao/speech/jimodels/openmodels`.

```bash
CUDA_VISIBLE_DEVICES=1 /tmp/mmduet-benchmark-venv/bin/python -m benchmark run mmduet2 \
  --model-path /remote-home/yushengliao/speech/jimodels/openmodels/MMDuet2 \
  --official-code /tmp/OmniDuplex-MMDuet2/proactive_eval

CUDA_VISIBLE_DEVICES=3 /tmp/livestar-benchmark-venv/bin/python -m benchmark run livestar \
  --model-path /remote-home/yushengliao/speech/jimodels/openmodels/LiveStar-8B \
  --official-code /tmp/OmniDuplex-LiveStar

CUDA_VISIBLE_DEVICES=0 /tmp/roma-benchmark-venv/bin/python -m benchmark run roma \
  --model-path /remote-home/yushengliao/speech/jimodels/openmodels/ROMA \
  --roma-code /tmp/OmniDuplex-ROMA
```

ROMA uses its pinned Transformers fork. JoyAI uses the official `services/webinfer`
adapter with local vLLM main and summary endpoints, then
`python -m benchmark run joyai --model-path ...` streams timestamped cached frames to
the adapter. Its adapter keeps the native response/silence format and memory.

## Scoring and outputs

A response onset is the video time when its first nonempty spoken text is emitted.
Continuous text chunks are merged until listen or end-of-turn. A prediction hits
if `window_start <= onset <= window_end`. Windows and onsets are matched one to
one, earliest deadline first. This maximizes timing matches; it does not check
semantic answer correctness. A plausible timestamp with incorrect text can
therefore receive a timing hit. Content needs separate human evaluation.

- Window hit rate: matched windows / reference windows.
- Timing precision: matched responses / all predicted responses.
- Timing F1: harmonic mean of the two.

`results/data_parallel_4567_20261007/minicpmo/predictions.jsonl` contains generated response text
and onset times; worker `sample_NNN.chunks.jsonl` files preserve every native decision and
chunk timing. `metrics.json` is updated after each sample and includes matches
and mode/ability breakdowns. A partial smoke run's metrics cover evaluated
samples' full reference windows, so they are not a complete benchmark result.

```bash
python -m unittest discover -s benchmark/tests -p 'test_*.py'
```

Official model and realtime examples:
https://huggingface.co/openbmb/MiniCPM-o-4_5
https://github.com/OpenBMB/MiniCPM-o

## Retained experiment results

The latest audited comparison is
[`results/experiment_summary_20261008/report.md`](results/experiment_summary_20261008/report.md).
MiniCPM-o-4.5, LiveStar and ROMA completed 44 samples under
`results/data_parallel_4567_20261007/`; JoyAI completed 44 under
`results/remaining_4567_20261008T022841Z/joyai/`. Their worker directories retain
raw outputs, prompts and configuration snapshots. ROMA's gate-score AUC diagnostic
remains in `results/data_parallel_4567_20261007/roma/auc_diagnostic/`.

MMDuet2 was stopped at the user's request with 32 of 44 samples complete. Its
combined predictions and report remain under
`results/mmduet2_recovery_20261008T062632Z/mmduet2_combined/`; source workers remain
in the two original experiment directories. Its partial results must not be
treated as a complete 44-sample score. Failed recovery records retain the CUDA
error evidence and memory-optimization configuration.

Old smoke runs, prompt-debug runs, early standalone shards and failed JoyAI startup
experiments were removed. `results/cleanup_20261008.json` records the removed files.
The combined `report.md` files list generated replies, onsets, reference windows
and matches; `collection_config.json` files record source runs and missing indices.

## Historical baseline result

All 44 samples completed, with 112 generated response events. The 157 scored
windows received 36 hits: **22.93% window hit rate**, **32.14% timing precision**,
and **26.77% timing F1**. The 32 references without windows were excluded.
Window hit rates by mode: RTD 18.00%, PR 36.67%, agent 25.93%.

The run uses model revision `503e754207c94da6bb26850b4469f367c9ea3582`.
Detailed text and matches are in `results/minicpmo45_realtime/report.md`.
This older prompt condition is retained separately from the current comparison:
combined predictions, scores, verification and source configuration snapshots
remain, while its early shards and raw chunk records were removed during cleanup.
`verification.json` records completeness checks. All 44 videos reached their
actual endpoints, sample indices are unique and complete, and the original timing checks passed at collection time.

Observed outputs include correct small-cat and reed-cover alerts, but also late
or premature responses, incomplete text, repeated alerts, and claims of tool
completion despite no tool execution. These are qualitative examples, not an
additional content-correctness score.

## Running selected unfinished samples

`parallel --sample-indices FILE.json` accepts a JSON array of original dataset
indices. The selected samples are balanced across GPUs without renumbering or
modifying the dataset; the exact selection is saved in the immutable parallel plan.
Use a new output directory, then `collect` the prior predictions and new run to
produce a complete report without duplicate indices.

MMDuet2 also supports `--mlp-chunk-tokens N` (default `0`, disabled). During
inference this computes decoder MLPs in token chunks, bounding their intermediate
activations while retaining the complete sequence and attention context. The
chunk size is recorded in `run_config.json`. Different matrix kernel shapes can
introduce floating-point rounding differences. This option does not support training.

MMDuet2 supports `--logits-mode last` as an explicit memory optimization for the
native generation path, which consumes only the last-position logits. It leaves
transformer inputs, token dropping, KV context, FPS and generation settings intact.
The default remains `full`. Projection kernels may have small floating-point
rounding differences; the selected mode is recorded in `run_config.json`.

## Response 内容 LLM Judge

内容评分作为推理完成后的独立步骤运行，直接复用 `predictions.jsonl`，不需要 GPU。
RTD / PR 评回复正确性；Agent 正样本分别评回复内容及实际 `agent_call` 参数；
Agent 负样本评拒绝 / 澄清内容，调用分为 N/A，并单列整个 QA 的误调用率。
工具返回和工具返回后的完成回复仍不评测。

裁判依据为问题、当前参考回复及标注证据、可用工具和 schema，不上传视频，也不提供后续工具结果。
这是参考标注一致性评测，不代表重新验证了原始音视频。
五档分数为 `0 / 0.5 / 1 / 1.5 / 2`，分别表示核心错误或缺失、少量正确、部分正确、基本正确、完全正确。
提示词位于 `configs/prompts/judge.txt`，可通过 `--system-prompt` 指定替代文件。

```bash
# 密钥通过环境变量配置；不要提交密钥到仓库。
export GEMINI_API_KEY='your-key'
python -m benchmark judge \
  /path/to/results/minicpmo /path/to/results/livestar \
  --data /path/to/inputs/all_qa.json \
  --output-dir /path/to/content_judge \
  --provider gemini --model YOUR_JUDGE_MODEL \
  --concurrency 4 --timeout 120 --retries 2
```

Gemini 使用项目现有 `google-genai` 依赖（`pip install -r data_generation/requirements-gemini.txt`）。
OpenAI 兼容接口使用 Python 标准库：指定 `--provider openai --model ...`，通过
`OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 配置。Gemini 支持 `GEMINI_BASE_URL`。
两者均支持 `--base-url`、`--api-key` 显式覆盖；可选 `--env-file` 通过现有 `python-dotenv`
加载配置，优先级为命令行 > 进程环境 > dotenv。裁判模型必须显式指定。

每个模型目录会产生：

- `judgments.jsonl`：实时追加逐 response 分数、理由、事件索引、解析状态和每次请求用量。
- `judge_metrics.json` / `judge_report.md`：整体、交互类型、任务类别和 Agent 子类型统计。
- `judge_config.json`：数据、预测及提示词哈希、裁判配置，不含密钥。
- `api_usage.json`：历史累计请求数、失败次数与 API 返回的 token 用量；未提供的用量不估算。

根目录另有 `judge_comparison.json` / `.csv`。使用以下命令并列查看时间与内容分数：

```bash
python -m benchmark compare /path/to/results/minicpmo /path/to/results/livestar \
  --judge-dir /path/to/content_judge --output /path/to/comparison
```

内容评分沿用原先的时间窗口一对一匹配，不按答案好坏重新选回复。
`response_mean_matched` 只统计时间命中的回复；`response_mean_all_windows` 将未命中记零。
Agent 调用分仅汇总正样本，同时报告命中与全窗口两种均分。
未完成推理、缺失视频不算零分，而是显示覆盖率。API 失败也不算零分：
受影响的汇总分显示 N/A，评分任务标记 incomplete 并以非零状态退出。
重复运行同一命令复用成功记录、重试失败记录；改变输入或裁判配置需要新的输出目录。
目录支持单写入进程锁，不要让两个进程同时写同一评测目录。

### Agent 结构化输出（可选）

默认继续使用已有的文本输出，不执行工具、不注入工具结果。
旧提示词只要求描述委托任务；旧预测中缺少实际调用参数时，正样本调用分为零，
不能据此单独推断模型的工具参数生成能力。解析只接受完整 JSON 或 JSON 代码块中的明确调用，
不从普通自然语言或参考答案补出调用参数。

后续推理可在现有 `run` / `parallel` 参数中加入：

```text
--agent-output-protocol structured
```

仅 Agent 样本会追加 `configs/prompts/agent_structured.txt`，要求每次回复为：

```json
{"response": "面向用户的回复", "tool_calls": [{"type": "function", "function": {"name": "agent_call", "arguments": {"instruction": "委托内容及 schema 要求的其他参数"}}}]}
```

拒绝或澄清使用空 `tool_calls`。仍由原生模型机制决定何时发声，且不改生成长度等超参数。
原文保留在 `events[].text`，另存 `response_text`、`tool_calls`、`agent_parse_status`。
协议及其提示词哈希参与运行身份校验，必须使用新输出目录，不能续接旧协议结果。
本模式仅要求输出格式；短输出预算导致的截断或模型不遵循格式会被标记，不能假定调用完整。

### MiniCPM `<|agent|>` 协议

训练后 adapter 可通过 `run minicpmo` / `parallel minicpmo` 的
`--lora-adapter /path/to/adapter --agent-output-protocol agent-token` 加载。
完整训练、词表扩展及加载说明见 [train README](../train/README.md)。

输出形式为 `<|speak|>自然语言回复<|agent|>{"name":"agent_call","arguments":{...}}<|turn_eos|>`。
调用参数可跨秒生成。解析直接使用原生生成 token ID，保留分隔符，完整轮次结束后才生成 `tool_calls`。
截断、重复分隔符和非法 JSON 都保留原文与错误状态；负样本输出分隔符即计为调用尝试。
Judge 沿用已有字段和时间匹配规则。默认 benchmark 输出协议仍为 text；旧 structured JSON 协议继续可用。
协议与 adapter / tokenizer 指纹加入并行计划和续跑校验，必须使用新结果目录。仅 MiniCPM 支持新协议。

## 全参数训练模型

`train` 同时提供全参 SFT 和完整流式 GRPO，参见 [训练说明](../train/README.md)。将 ZeRO checkpoint 用 `python -m train export` 导出后，可将导出目录作为 MiniCPM 的 `--model-path`，并指定 `--agent-output-protocol agent-token`；不需要再传 LoRA adapter。导出身份、config/index 和 tokenizer 会加入评测协议校验。

## 云端 Realtime：Qwen3.5 Omni 与 Gemini 2.5 Flash Live

新增两个 API adapter，覆盖以下三个模型，不需要下载权重或使用 GPU：

| CLI adapter | `--model` | 返回内容 |
| --- | --- | --- |
| `qwen_realtime` | `qwen3.5-omni-plus-realtime`（默认） | 文本 |
| `qwen_realtime` | `qwen3.5-omni-flash-realtime` | 文本 |
| `gemini_live` | `gemini-2.5-flash-native-audio-preview-12-2025`（默认） | 音频的输出转写 |

这部分按官方协议实现，尚未进行真实 API 联调。模型 ID 可通过 `--model` 覆盖；
账号需要开通相应地区、模型和 Live API 权限。服务端模型下线或权限错误会直接报错，
不会自动替换模型。参考 [Qwen WebSocket 协议](https://www.alibabacloud.com/help/en/model-studio/omni-realtime-interaction-process)、
[Qwen 事件参数](https://help.aliyun.com/en/model-studio/client-events)、
[Gemini 2.5 Flash Live 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-native-audio-preview-12-2025)
和 [Gemini Live 能力说明](https://ai.google.dev/gemini-api/docs/live-api/capabilities)。

### CPU 环境

在仓库根目录创建单独环境；无需安装本地模型使用的 Torch / CUDA 依赖：

```bash
python3 -m venv benchmark/.venv-realtime
source benchmark/.venv-realtime/bin/activate
python -m pip install -r benchmark/requirements-realtime.txt
ffmpeg -version
ffprobe -version
```

需要 Python 3.10+、FFmpeg 和 libsndfile（供 `soundfile` 使用）。媒体缓存沿用
`--cache-dir`，先提取带原始时间戳的约 1 Hz 视频帧和 16 kHz 单声道音频。
图片最长边默认 640、JPEG quality=75；超过接口大小限制时进一步缩小。
容器中的音轨短于视频时，对缺失尾段补静音以维持输入时间对齐，并记录补齐的采样点数。

### Qwen Plus / Flash 的连接和密钥

在百炼控制台获取已开通目标模型的 API key，并复制该地区的 Workspace ID。
北京和新加坡的 Workspace、API key、endpoint 应对应同一地区。以下使用北京：

```bash
export DASHSCOPE_API_KEY='YOUR_API_KEY'
export DASHSCOPE_REALTIME_URL='wss://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime'

python -m benchmark run qwen_realtime \
  --model qwen3.5-omni-plus-realtime \
  --data "$BENCH_DATA" --videos-dir "$BENCH_VIDEOS" \
  --output-dir benchmark/results/qwen35_omni_plus_realtime

python -m benchmark run qwen_realtime \
  --model qwen3.5-omni-flash-realtime \
  --data "$BENCH_DATA" --videos-dir "$BENCH_VIDEOS" \
  --output-dir benchmark/results/qwen35_omni_flash_realtime
```

新加坡 endpoint 为
`wss://YOUR_WORKSPACE_ID.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime`。
也可以用 `--base-url` 覆盖 endpoint；代码会为 URL 设置 `model` 查询参数。
密钥通过 Authorization header 发送。不要把密钥写入 endpoint、命令参数或结果文件。

### Gemini 2.5 Flash Live 的连接和密钥

使用支持目标 Live 模型的 Gemini Developer API key；这里使用 `google-genai` 的
`client.aio.live`，不是普通 `generateContent`，也不是 Vertex AI adapter。

```bash
export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'

python -m benchmark run gemini_live \
  --model gemini-2.5-flash-native-audio-preview-12-2025 \
  --data "$BENCH_DATA" --videos-dir "$BENCH_VIDEOS" \
  --output-dir benchmark/results/gemini25_flash_live
```

默认 API 版本为 `v1beta`，可用 `--api-version` 覆盖。需要兼容的 Gemini 网关时，
设置 `GEMINI_BASE_URL` 或传 `--base-url https://YOUR_GATEWAY`，网关必须支持
Live 双向 WebSocket；仅支持普通 HTTP 生成接口的代理不能使用此 adapter。
Realtime adapter 从进程环境读取密钥，不自动加载 `.env`；可以通过
`--api-key-env YOUR_KEY_VARIABLE` 指定其他环境变量名。结果只保存变量名，不保存密钥。

### 流式输入与评分时间

每条 QA 使用一个独立有状态会话，从视频 0 秒开始。到 `question_time` 才注入用户问题；
只传用户问题、允许的工具及 `agent_call` schema，不传参考答案、窗口、证据或 Agent 返回。
默认保持文本输出协议；可选 `--agent-output-protocol structured`，继续复用现有 Agent parser。
Gemini 音频转写不保证逐字保留 JSON，应结合 `agent_parse_status` 检查调用参数。

每次先输入 `[t,t+1)` 的音频和落在该区间内的视频帧，再允许输出；`t+1` 的帧属于下一步。
输入不快于每秒一帧，收到本轮完整响应后才推进下一段。Qwen 使用关闭 VAD 的
`append → commit → response.create`；Gemini 关闭自动 VAD，使用
`activity_start → 实时音视频/文本 → activity_end`。这是逐秒手动分轮的可控评测，
与模型连续接收音视频、依赖原生 VAD 自行发声的模式不同。

云端 API 没有本地模型的 listen token，因此附加
`configs/prompts/cloud_realtime.txt`：无需回复时生成单词 `SILENCE`，实际回复则直接输出内容。
Gemini 以音频方式输出，评分采用 `output_audio_transcription`；不播放或保存输出音频。
`SILENCE` 是决策标记，不代表输出音轨真正静音，独立标记会从预测事件中过滤。
没有原始音轨时，Gemini 不发送音频；Qwen 的图片缓冲协议要求先追加音频，
因此补同长度静音，最后不足 100 ms 的音频补静音到提交下限；这些操作记录在原始日志中。

事件的 `onset_seconds` 记为本轮输入结束时间。例如输入 00:43–00:44 后生成的回复记为
44 秒。网络等待不会累加到视频评分时间；首个输出到达的墙钟时间和整步耗时另行保存。
生成策略、API endpoint、模型、输入策略及提示词哈希均进入续跑校验；不同模型使用不同结果目录。
输出仍为 `predictions.jsonl`、`metrics.json`、`run_config.json` 和逐样本
`sample_XXX.realtime.jsonl`，可继续使用 `collect`、`compare` 和 `judge`。
原始日志包含文本、usage 和输入区间，不记录 API key、Base64 媒体或生成音频。

`--api-timeout` 默认 180 秒，覆盖会话初始化或单次完整输入/输出步骤；
`--max-output-tokens` 默认不设置，沿用服务端默认生成长度。
`--max-samples`、`--sample-start`、`--sample-step` 和 `--max-seconds` 可用于后续小规模验证。
API 报错、缺少输出转写、会话到期或网络中断会中止当前样本，不写入一条“已完成”的空预测；
重跑相同配置会复用已完成 QA，并从该失败 QA 的视频起点重新开始。
本版不自动续接超长服务端会话；视频较长或逐秒分轮数超过服务端限制时需要处理会话续接，
不会通过清空上下文或截断视频冒充全程评测。云端 adapter 仅注册在 `run`，不使用 GPU `parallel`。

## Probe 模式：普通 Omni LLM 的回复决策评测

Probe 测试模型在看到一个完整视频前缀后，能否判断“是否已出现新的应回复机会”。
它使用普通非双工推理，与在线模型自主决定何时发声的 timing benchmark 分开保存、评分。
第一版提供 `qwen3_omni` 和 `qwen25_omni` 两个本地 adapter；实现尚未执行模型或测试，
后续运行需要先配置独立环境和本地权重。

### 正负样本与已处理次数

对每个可评分 response，固定构造两个独立请求：

| 侧别 | 视频前缀 | 期望判定 |
| --- | --- | --- |
| pre，负样本 | `[0, window_start − 1s)` | `NO` |
| post，正样本 | `[0, window_end)` | `YES` |

例如回复窗口为 10–12 秒，负前缀截至 9 秒，正前缀截至 12 秒；不额外加入 12–13 秒内容。
问题在 `question_time` 发出，之前的视频可以作为上下文，但只统计问题发出后（含该时刻）的回复机会。
每条 response 独立配对；Agent 仅包含触发时的首次委托回复，不包含工具返回及完成回复。
历史 Agent 能力负样本也不纳入。

参考 [OmniPro evaluator](https://github.com/RuixiangZhao/OmniPro/blob/main/evaluators/probe_evaluator.py)
及其 [prompt](https://github.com/RuixiangZhao/OmniPro/blob/main/utils/prompts.py)，
对重复事件告知“此前已有 N 次回复机会被处理”，判断是否出现第 N+1 次或更多新机会。
同一配对的两侧使用相同 N；N 基于全部可评分窗口的时间顺序，而不是保留的配对数量。
这是带 GT 既往计数辅助的诊断设置，不代表完全自主的在线判断能力。
本项目统一评 YES/NO，未照搬 OmniPro 部分任务的计数或选择题指标；截断点也固定为上表规则。

模型不接收目标窗口、标签、参考回复、标注证据或目标事件定义。Probe prompt 位于
`configs/prompts/probe.txt`，结合原问题、可用工具/schema、前缀时长和已处理次数生成。
它判断的是“截至此处是否已应回复”，而非“最后一帧是否还处在回复窗口内”。
输出要求为 `NO` 或 `YES: 简短说明`；裸 `YES` 也可解析，YES 说明仅保留供分析，不做内容评分。

负截断点不晚于问题时间、前缀为空、窗口越界或媒体缺失时排除该对；
既往机会尚未在负前缀中出现时，也不会向模型错误宣称这些机会已处理。
缺少窗口会记录排除；窗口自身时间非法时排除整个 QA，以避免错误的历史计数。
所有排除原因保存，不通过裁到 0 秒或视频结尾来凑齐正负样本。

### 准备清单

以下命令仅在后续需要实际评测时执行。`prepare` 使用 Python 标准库和 `ffprobe`，不加载模型：

```bash
python -m benchmark probe prepare \
  --data "$BENCH_DATA" --videos-dir "$BENCH_VIDEOS" \
  --output-dir benchmark/probes/default
```

输出 `probe_manifest.jsonl`、`probe_manifest_config.json`、`probe_exclusions.jsonl`
及 `probe_prepare_report.json`。清单记录稳定的 pair/probe ID、源文件指纹、截断时间、
期望标签和已处理次数。必须使用空的准备目录，不覆盖已有清单。
这里只写清单，不批量生成前缀视频；运行时按需缓存到 `benchmark/cache/probe_prefix/`。

### 独立模型环境与运行

Python 3.10+，系统需要 `ffmpeg`、`ffprobe`；独立环境使用 Transformers 4.57.3，
不要升级现有 MiniCPM 等模型环境里的 Transformers 4.51。
Qwen 推理使用官方 [Qwen3-Omni API](https://huggingface.co/docs/transformers/v4.57.3/model_doc/qwen3_omni_moe)
及 [Qwen2.5-Omni API](https://huggingface.co/docs/transformers/v4.57.3/model_doc/qwen2_5_omni)。

```bash
python3 -m venv benchmark/.venv-probe
source benchmark/.venv-probe/bin/activate
python -m pip install -r benchmark/requirements-probe.txt

python -m benchmark probe run qwen25_omni \
  --manifest benchmark/probes/default/probe_manifest.jsonl \
  --model-path /remote-home/yushengliao/speech/jimodels/openmodels/Qwen2.5-Omni-7B \
  --output-dir benchmark/results/probe/qwen25_omni

python -m benchmark probe run qwen3_omni \
  --manifest benchmark/probes/default/probe_manifest.jsonl \
  --model-path /remote-home/yushengliao/speech/jimodels/openmodels/Qwen3-Omni-30B-A3B-Instruct \
  --output-dir benchmark/results/probe/qwen3_omni
```

`--model-path` 必须指向已准备好的本地 checkpoint，不自动下载权重。
默认 BF16、`device_map=auto`、SDPA；可通过 `CUDA_VISIBLE_DEVICES` 限制模型分片可见的卡。
这只是单进程模型分片，不是 GPU data parallel。可显式设置 `--dtype`、`--device-map`
或 `--attn-implementation flash_attention_2`；后者需另行安装兼容的 FlashAttention。

默认 1 FPS、每帧最大像素 `360×420`、greedy 解码、最大新 token 数 256，
分别由 `--fps`、`--max-pixels`、`--max-new-tokens` 覆盖。默认值集中在 `src/config/defaults.py`。
两模型关闭 Talker，仅输出文本；每个 probe 建立全新的对话，不把另一侧的答案作为历史。
有音轨时输入音视频，无音轨时只输入视频。完整前缀超过模型上下文或显存时记录失败，不截短输入。
`--max-pairs` 可限制固定配对子集；限制先于续跑跳过已完成请求执行。

### 保存、续跑与评分

前缀经 FFmpeg 分别 trim 音视频后重编码，保留流之间的时间偏移，避免关键帧复制包含未来内容。
缓存键包含源文件指纹、截断时间及处理命令；源视频变化后需重新 prepare。

结果目录包含 `probe_predictions.jsonl`、`probe_errors.jsonl`、`probe_prompts.jsonl`、
`probe_config.json`、`probe_metrics.json` 和 `probe_report.md`。
每完成一侧就立即保存并更新统计；同一命令重跑会复用已完成请求，仅重试失败侧。
格式错误是已完成但答错的结果，不自动重试到答对。
修改数据、prompt、模型、实现或推理参数后必须使用新结果目录。

无需加载模型即可重新统计：

```bash
python -m benchmark probe score \
  --manifest benchmark/probes/default/probe_manifest.jsonl \
  --predictions benchmark/results/probe/qwen25_omni/probe_predictions.jsonl \
  --output-dir benchmark/results/probe/qwen25_omni
```

主指标为正/负准确率、整体准确率、balanced accuracy 和 paired accuracy；
paired accuracy 要求一对两侧均正确，只在完整配对上计算，同时显示完整配对覆盖率。
无法解析为 YES/NO 的文本记为 invalid，在上述准确率中计错，不默认为 NO。
推理失败属于未完成覆盖，不伪装成模型的 NO。
Precision/recall/F1 仅基于可解析的已完成结果，并同时报告 parse coverage、invalid rate、
失败数；结果按 `mode` 和 `ability` 汇总。该结果不交给在线 timing scorer 或内容 judge 混合计分。

## 完整 2110 条数据：本轮模型准备（尚未启动推理）

本轮配置集中在 `configs/suites/full_2110.json`。Realtime 包含 Gander、
JoyAI-VL-Interaction、MiniCPM-o-4.5、LiveStar；probe 包含
Qwen3-Omni-30B-A3B-Instruct、Qwen2.5-Omni-7B、MiniCPM-o-4.5、
Ming-Lite-Omni-1.5、Qwen3-VL-8B-Instruct、Baichuan-Omni-1.5、HumanOmniV2、OmniVinci。
没有安排 MMDuet2。

```bash
# 下载 Gander / Ming 官方源码（不走代理），然后创建隔离环境、安装依赖。
# 不加载模型、不启动服务。
bash benchmark/scripts/fetch_native_sources.sh
bash benchmark/scripts/prepare_environments.sh

# CPU 文件/模型类导入检查、视频路径解析、probe 配对清单；不会推理。
python -m benchmark prepare --suite benchmark/configs/suites/full_2110.json
```

准备产物位于 `benchmark/probes/full_2110/`：`preparation_report.md/json`、
`run_commands.json`、`videos/` 别名以及 `probe/probe_manifest.jsonl`。
`run_commands.json` 只保存后续命令，不会自动执行；启动时必须另行指定 GPU 分配。
JoyAI 命令需要在正式启动阶段通过已有 `parallel joyai --joyai-services` 流程管理
原生 main/summary/adapter 服务。准备阶段不启动这些服务。

Probe 沿用已有规则：负前缀截至 `window_start−1s`，正前缀截至 `window_end`。
Qwen3-VL 只输入视频；其余模型按实际音轨输入音视频。每个 pre/post 请求独立，
使用原问题与既往已处理次数，输出 YES/NO；没有参考回复、目标窗口或标签输入模型。
超出上下文或处理器降低完整前缀的采样覆盖时记录失败，不截去视频开头来完成评测。
新增原生 adapter 按模型自己的空间处理器编码视频；Qwen 和 MiniCPM/Baichuan
使用配置中的像素预算，OmniVinci 保留原生视觉处理器，不能把各模型的视觉 token 数视为相同。

准备时保留源 JSON，并为 `A/videos/...`、`B/videos/...` 和可由 metadata 识别的旧名称
创建只读素材别名。缺失素材及被排除的 probe 配对写入报告，不补回外部视频。

Gander 使用官方 `mcpmft.infer.online.OnlineRunner`，Thinker 每秒 8 个词汇 token、
11 个含控制标记的解码预算。时间评测收集 Thinker 文本，不运行 Talker 语音合成。
Ming 固定使用官方 `v1.5` 源码；`prepare_ming_text.py` 将语音输出模块改为按需导入，
probe 关闭 Talker/image generation，但保留音频输入编码器。其源码变更记录在
缓存中的 `text_probe_patch.json`。OmniVinci 环境按官方要求应用其 Transformers 补丁，
只修改隔离环境。Ming、Baichuan、OmniVinci 默认使用原生支持的 eager attention，
其余 probe 配置沿用 SDPA。这些配置的 GPU 推理尚未验证，待明确启动后再检查。

Gander 权重安装在 `/remote-home/yushengliao/speech/jimodels/openmodels/Gander`。
下载时清除所有代理变量，允许使用镜像站：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
    hf download Gander-Omni/Gander \
      --revision 24fc4cc8543f95daf99be53b6199403a7732688f \
      --local-dir /remote-home/yushengliao/speech/jimodels/openmodels/Gander
```

Gander 与 Ming 原生源码位于忽略的 `benchmark/cache/official/`；隔离环境位于
`benchmark/.venv-*`，均不加入 Git。安装脚本在源码缺失时明确报错，不下载错误版本。

### 2026-10-22 修复后补跑

前缀编码使用 FFmpeg 4.4 支持的 `-vsync 0`，保持原采样和截断边界。
JSON 产物显式按 UTF-8 保存。Gander 复用原生 OnlineRunner，在每条视频开始时
通过原生 `prepare()` 重置会话；没有额外启用滑窗或丢弃历史。
Ming 从官方源码目录加载音频处理资源；Qwen3-VL 显式传递原视频 FPS 和采样帧索引。
Baichuan 使用 checkpoint 自带的 chat template，本轮显式设置
`--attention-query-chunk-size 512` 分块计算原生 eager attention，保留完整键值和因果掩码。
该选项默认仍为 0，并记录在 probe 配置中。

JoyAI 服务使用 `benchmark/.venv-joyai`，固定 vLLM 0.19.1 / CUDA 12.8，
避免 CUDA 13 超出当前驱动支持范围；不修改系统驱动。
安装该环境需要 `uv`，可运行 `bash benchmark/scripts/prepare_environments.sh joyai`。

补跑命令（前台监督；需要后台运行时由调用者显式安排）：

```bash
benchmark/.venv/bin/python benchmark/scripts/run_prepared_suite.py \
  --commands benchmark/probes/full_2110/run_commands.json \
  --gpus 0 1 2 3 6 7 \
  --output-dir benchmark/results/full_2110_dp6_repair_20261022 \
  --reuse-root benchmark/results/full_2110_dp6_20261021 \
  --joyai-services benchmark/probes/full_2110/joyai_services.json
```

`--reuse-root` 复用已完成的 realtime 运行，并按原计划续跑已有分片。
Probe 使用新结果目录，避免混用修复前后的实现。新目录内的 `suite_status.json`
和 `suite.log` 记录队列状态；被复用模型的具体预测仍保存在原目录中。

### Agent 工具协议重评

2026-10-22 的协议审查、模型差异、验证限制和 Agent-only 重评命令见
[工具调用协议审查](docs/agent_protocol_audit_20261022.md)。
新 Agent 结果与之前 timing-only 结果分开保存，不把“按时说会调用”计为有效工具调用。

### 简化 Agent 调用（2026-10-23）

Benchmark 默认使用 `--agent-call-schema task-start`：仅暴露
`task_start({"instruction":"一句简短、明确的后台任务"})`。这是评测自定义 schema，
不同于 Gander 官方 `task_start(name)`。保留 available 后端能力列表及权限限制，
模型无需再输出七字段参数。Gander 仍走原生 `<tool_call>`，其他模型沿用所选文本/structured 通道。

原始 QA、生成器和训练协议不变；运行时将旧参考调用的 instruction 转成评测视图。
`judge` 从结果配置识别 schema，旧目录默认 legacy，不要求新调用输出旧字段。
需要原协议时显式传 `--agent-call-schema legacy`；训练权重的 `--agent-output-protocol agent-token`
也必须搭配 legacy。新旧 Agent 结果使用不同目录，不能互相断点续用。

```bash
benchmark/.venv/bin/python benchmark/scripts/run_agent_recheck.py \
  --commands benchmark/probes/full_2110/run_commands.json \
  --output-dir benchmark/results/agent_task_start_v1_20261023 \
  --agent-call-schema task-start --max-tool-tokens 1024 \
  --gpus 0 1 2 3 6 7
```

添加 `--dry-run` 仅写入计划，不启动推理。当前安排是 Agent 重评后自动补齐全量队列，
状态目录分别为 `results/agent_task_start_v1_20261023` 和
`results/full_2110_dp6_after_task_start_20261023`。
调用统计区分语法可解析与 schema 合法；简化协议不再统计模型输出的 suggested_capabilities
或 extracted_parameters。语义正确性仍需另外运行内容 judge，工具不会实际执行。
