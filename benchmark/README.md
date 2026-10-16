# Realtime timing benchmark

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

Current data: `all_qa.json`, derived from
`data_generation/data/qwen3.8_omni_flash_44/all_qa.json` with an explicit Agent-window
migration. `configs/manifests/data_manifest.json` records the source hash, current
hash, backup and migration ledger. There are 44 samples, 189 reference responses,
and 173 responses with timing windows. Benchmark scoring uses 157 of those windows:
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
