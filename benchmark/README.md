# Realtime timing benchmark

## Code structure and commands

Run commands from the repository root with the appropriate model's Python environment.
The former standalone `run_*.py`, collection, comparison and download scripts have
been replaced by a single entrypoint:

```bash
python -m benchmark --help
python -m benchmark run minicpmo --help
python -m benchmark compare benchmark/results/minicpmo45_realtime benchmark/results/mmduet2
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

Current data: `all_qa.json`, copied byte-for-byte from
`data_generation/data/qwen3.8_omni_flash_44/all_qa.json`.
`configs/manifests/data_manifest.json` records the source and SHA-256. There are 44 samples,
189 reference responses, and 157 responses with timing windows. The other
32 references are not scored because they have no window.

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

`results/minicpmo45_realtime/predictions.jsonl` contains generated response text
and onset times; `sample_NNN.chunks.jsonl` preserves every native decision and
chunk timing. `metrics.json` is updated after each sample and includes matches
and mode/ability breakdowns. A partial smoke run's metrics cover evaluated
samples' full reference windows, so they are not a complete benchmark result.

```bash
python -m unittest discover -s benchmark/tests -p 'test_*.py'
```

Official model and realtime examples:
https://huggingface.co/openbmb/MiniCPM-o-4_5
https://github.com/OpenBMB/MiniCPM-o

The current full evaluation uses `initial_sample` (sample 0) plus three foreground
shards (`shard_1`, `shard_2`, `shard_3`, start indices 1/2/3 and step 3).
To refresh their combined metrics and readable output report:

```bash
benchmark/.venv/bin/python -m benchmark collect \
  benchmark/results/initial_sample benchmark/results/shard_1 \
  benchmark/results/shard_2 benchmark/results/shard_3
```

The combined `report.md` lists every generated reply, its onset, reference windows,
and timing matches. `collection_config.json` lists missing sample indices.

## Completed baseline result

All 44 samples completed, with 112 generated response events. The 157 scored
windows received 36 hits: **22.93% window hit rate**, **32.14% timing precision**,
and **26.77% timing F1**. The 32 references without windows were excluded.
Window hit rates by mode: RTD 18.00%, PR 36.67%, agent 25.93%.

The run uses model revision `503e754207c94da6bb26850b4469f367c9ea3582`.
Detailed text and matches are in `results/minicpmo45_realtime/report.md`;
raw predictions and full chunk decisions remain in their result directories.
`verification.json` records completeness checks. All 44 videos reached their
actual endpoints, sample indices are unique and complete, and the original timing checks passed at collection time.

Observed outputs include correct small-cat and reed-cover alerts, but also late
or premature responses, incomplete text, repeated alerts, and claims of tool
completion despite no tool execution. These are qualitative examples, not an
additional content-correctness score.
