# 36-task audiovisual interaction generation

The pipeline is **original video → dense caption → 12-ability routing → task generation
and validation → QA merge and coverage report**. It replaces the former twelve
tasks; old task names have no aliases. Source videos, task definitions and existing
JSON files are retained, but old task samples are rejected by the new merger.

## Task matrix

Every ability below supports three task suffixes: `_rtd` (real-time description),
`_pr` (proactive reminder), and `_agent` (triggered delegation).

| Ability ID | Focus |
| --- | --- |
| `attribute_perception` | Attributes and changes |
| `grounding` | Event moments and completed time spans |
| `action_perception` | Actions and phase boundaries |
| `object_perception` | Objects, identities and categories |
| `emotion_recognition` | Emotion inference from voice, expression and context |
| `event_understanding` | Events, participants and relations |
| `text_rich_understanding` | Visible text and its meaning |
| `object_counting` | Current objects, with identity deduplication |
| `action_counting` | Cumulative independent action/speech/sound occurrences |
| `spatial_understanding` | Relative positions and spatial relations |
| `prospective_reasoning` | Uncertain predictions from current evidence |
| `causal_reasoning` | Candidate causes and their supporting evidence |

`src/domain/catalog.py` is the shared directory for registration, routing and prompt
composition. `route` and `caption` are auxiliary stages, not dataset tasks.
Definitions follow [the v2 task document](<Agentic Omni Interaction 主动交互任务定义 v2 (1).md>).
Prompts combine common rules, an ability specification and a mode template. They
reuse the local paper `tmp/arXiv-2605.18577v1.zip`, appendix `data_details.tex`, and
the previous pipeline's video-first, dense-caption and modality-labeling design.
Examples do not restrict Agent work to particular applications or operations.

## Source architecture

The four layers under `src/` separate command handling, orchestration, business
rules and external I/O:

| Directory | Responsibility |
| --- | --- |
| `cli/` | Argument and environment configuration, terminal output and exit codes |
| `application/` | Request preparation, single generation, batch/resume, QA merge and prompt export |
| `domain/` | Task catalog, sampling configuration, contexts, pure prompt rendering and validation |
| `infrastructure/` | Model HTTP/SSE clients, Qwen uploads, ffprobe, JSON adapters, artifact hashes and usage records |

CLI modules call application services; applications compose domain rules and
infrastructure adapters. Domain modules receive template text and loaded inputs;
they do not read files or call external services. Resource locations are defined
in `infrastructure/paths.py`, so prompts and the default repository `.env` remain
independent of the working directory. Sampling settings live in `domain/config.py`.

Both generation commands share `application/generation.py`. Its input is a
`GenerationOptions` dataclass and its return value contains `sample` and
`usage_record`; it does not mutate CLI arguments. Batch execution accepts
`BatchOptions` and an optional progress callback. CLI output, parameters, dataset
schemas and generation defaults remain unchanged. The old Python module import
paths have been removed; Python callers should use the new layer paths.

The source reorganization changes the implementation fingerprint. Existing batch
artifacts will be regenerated on the first resume after this change; subsequent
unchanged runs resume normally. Use a new output directory to retain an old run.

## Single-task CLI

Set `OPENAI_MODEL`, `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`, or pass
`--model`, `--api-key`, `--base-url`. The model endpoint must accept the existing
`video_url` payload, including audio. Existing temperature (0.1), timeout (120 s),
retry count (2), video preprocessing and caption segmentation remain unchanged.
`ffprobe` is needed to determine video duration.

```bash
python data_generation/generate.py --task caption \
  --video data_generation/videos/test.mp4 \
  --output /tmp/omni-v2/test/caption.json

python data_generation/generate.py --task route \
  --video data_generation/videos/test.mp4 /tmp/omni-v2/test/caption.json \
  --output /tmp/omni-v2/test/route.json

python data_generation/generate.py --task object_perception_agent \
  --video data_generation/videos/test.mp4 /tmp/omni-v2/test/caption.json \
  --output /tmp/omni-v2/test/object_perception_agent.json
```

Any task accepts `--print-prompt` to render its sampled request without an API
call. `generate_video_prompts.py` still renders the standalone caption prompt.
Captions retain the array format with `start`, `end`, `caption`, `visual`, `audio`,
`speech`, covering the rounded video duration without gaps or overlaps. The 5–30 s
segment guidance and 45 s maximum remain unchanged. Caption boundaries are not
precise event timestamps; every task also receives the original video.

### Qwen3.8-Omni-Flash

Install the dotenv reader with `python -m pip install -r data_generation/requirements.txt`.
Use `--provider qwen` in either the single-task or batch CLI. It reads `QWEN_API_KEY`
and `QWEN_URL` from the repository-root `.env` (override its path with `--env-file`).
An optional `QWEN_MODEL` overrides `qwen3.8-omni-flash`. Explicit CLI arguments take
precedence over process environment variables, which take precedence over `.env`.
Qwen credentials never fall back to `OPENAI_API_KEY`. The default provider remains
`openai`, with its existing environment configuration.

```dotenv
QWEN_API_KEY=<your-key>
QWEN_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

```bash
python data_generation/generate.py --provider qwen --task route \
  --video videos/video_161.mp4 tmp/rendered_prompts/video_161/caption.json \
  --output /tmp/omni-qwen/video_161/route.json

python data_generation/generate_batch.py --provider qwen \
  --videos-dir /path/to/videos --output-dir /tmp/omni-qwen-batch
```

The Qwen adapter uses Chat Completions streaming with text output, combining only
final-answer deltas and excluding reasoning. It preserves server thinking defaults
and omits `response_format`; the prompts and existing local validators still
require the expected JSON. Temperature, timeout, retries, and video preprocessing
are unchanged. See the [model API example](https://www.qianwenai.com/models/qwen3.8-omni-flash)
and [Qwen-Omni input limits](https://help.aliyun.com/zh/model-studio/qwen-omni).

With `--provider qwen`, both CLIs automatically upload the original local video
to DashScope temporary storage, then send the resulting `oss://` URL in `video_url`
with `X-DashScope-OssResourceResolve: enable`. Upload uses a streamed multipart
form and does not Base64-encode, compress, crop, or resample the video. The policy
API determines the size limit for the selected model. Temporary files expire after
48 hours. See the [official upload procedure](https://help.aliyun.com/zh/model-studio/get-temporary-file-url/).

The upload policy endpoint is inferred for standard DashScope regional URLs.
For another endpoint, set `QWEN_UPLOAD_URL` in `.env` or pass `--upload-url` with
the appropriate region's policy endpoint. The upload and model call use the same
API key and model. Uploaded URLs are cached only within the current process,
scoped by file identity, credential, model and endpoint, and expire after 48 hours.
Batch stages reuse the same upload; a later process uploads again when needed.

In the single-task CLI, `--video-url` accepts an existing accessible HTTP(S) URL
or Qwen temporary `oss://` URL to skip automatic upload. `--video` remains required
for local duration checks. Other providers require an explicit remote video URL;
local `file://` and Base64 URLs are not sent to the API.

## Batch generation and resume

```bash
python data_generation/generate_batch.py \
  --videos-dir data_generation/videos --output-dir /tmp/omni-v2-batch \
  --samples-per-task 1 --seed 42

# Use the printed run directory and identical configuration to resume.
python data_generation/generate_batch.py \
  --videos-dir data_generation/videos --output-dir "/tmp/omni-v2-batch/<UTC-run-timestamp>" \
  --samples-per-task 1 --seed 42 --resume

# Restrict generation; routing still evaluates all 12 ability categories.
python data_generation/generate_batch.py \
  --videos-dir data_generation/videos --output-dir /tmp/omni-v2-selected \
  --tasks grounding_pr prospective_reasoning_agent --seed 42
```

Execution is sequential and in the foreground. Videos are discovered recursively
(`mp4`, `mov`, `mkv`, `webm`, `avi`). Each gets a collision-resistant folder, a caption,
a complete route and uniquely numbered task files. Routing partitions the 12 ability IDs into `ranked_tasks` and `unsuitable_tasks`
using the existing `task` field, without mode suffixes. A category is suitable
when at least one mode can generate a valid task. By default all three modes of
each suitable category are attempted once; `--tasks` limits those attempts and
individual generators may return `skip` for unsupported modes.
`--samples-per-task` counts attempts, not guaranteed accepted samples. There is no
unbounded retry-to-fill loop. Individual failures are recorded and other work
continues; upstream failures mark dependent attempts blocked. Exit status is 1 if
any stage failed, while `skip` and unsuitable routes are normal outcomes.

Every new invocation creates an independent UTC-named run under `--output-dir`
and prints its actual path. The timestamp format is `YYYYMMDD_HHMMSS_ffffff`.
With `--resume`, `--output-dir` must point to that specific run, not its parent.

```text
output-dir/
└── YYYYMMDD_HHMMSS_ffffff/
    ├── all_qa.json
    ├── coverage.json
    ├── .state/manifest.json
    └── video-name-path-hash/
        ├── caption.json
        ├── qa/task_name.0000.json
        ├── all_qa.json
        └── .state/route.json
```

Each video has its own accepted QA bundle. The run's `all_qa.json` combines only
accepted, revalidated samples from the current invocation, including valid resumed
artifacts. Both bundles and `coverage.json` are updated after each video, including
videos with failures or no accepted samples. Coverage includes all 36 tasks,
accepted/skip/failed/unsuitable/blocked counts, Agent branches, errors and API usage.
The manifest retains fingerprints, content hashes and per-stage usage for resume.
Original videos remain in `--videos-dir`; they are not copied into the run.

After a video's bundles and statistics are saved, successful stages' independent
`.usage.json` files are removed: their usage is already retained in the manifest
and coverage report. Failed-stage diagnostic files, route, caption, task results,
source videos and unrelated files are retained. Atomic JSON writes replace their
own `.pending` files on success. Summary-write failures prevent cleanup; cleanup
failures are recorded and produce a nonzero exit status.

Resume checks source video content, source path, model/API settings, seed/task
selection, implementation and prompt fingerprints, upstream artifacts, output
hashes, sampled context and schema validation. API keys are never stored. Changing
configuration invalidates reuse; regenerating an artifact replaces its output
within the selected run. New invocations automatically retain prior runs. Explicitly resumed legacy manifests
keep their original flat video layout; no existing run is migrated. Stable per-stage seeds
make sampling reproducible across resume, but do not make remote model output
deterministic. Batch samples store video paths relative to `--videos-dir`.

## Temporal and capability contracts

New task samples use `schema: "omni-interaction-v2"`, `task`, `ability`, `mode`,
`video`, `generation_context`, `question`, `question_time` and `responses`.
`trigger_type` retains its audiovisual meaning (`visual`, `sound`, `speech`, or
combinations); `trigger.kind` separately denotes `event`, `condition`, `count`, or
`target`. Time strings remain `MM:SS`.

- Every response has `window_start <= trigger_time <= window_end`: earliest
  eligible time, recommended trigger time, and latest still-useful time. These
  describe valid response timing, not an event's duration. All evidence needed
  for the response must already be available at `window_start`; `confirmed_time`
  equals that earliest boundary. Boundaries come from video evidence and the
  request, not fixed tolerances. Instantaneous windows may have all three equal.
- RTD requires at least two meaningful updates with increasing recommended times.
  Valid windows for distinct updates may overlap.
- PR responds to every independent matching episode after the instruction.
  Persistent conditions require a reset before retriggering. Count predicates
  must transition from false to true; all compound conditions must hold.
- Agent questions cover every independent matching episode through video end.
  Each episode has an explicit short `response`, equal to its acknowledgement,
  and three messages: acknowledgement plus `agent_call`, simulated backend result,
  final reply with an `outcome` matching the result. Overlapping delegations remain
  separate episodes; merged messages are sorted chronologically.

The former 10–180 s initial silent interval is removed for these new tasks.
`initial_response_delay_seconds` is persisted as zero. Existing start/non-start
question sampling and audio/vision preference remain 50/50; a preference never
overrides an ability's required visual evidence. Backend delay remains sampled
between `max(5 s, ceil(10% duration))` and `floor(50% duration)`; positive samples
skip if any matching trigger has no valid window allowing its backend completion within the
video; never silently omit late matches.

Every response contains a brief, precise user-facing `response`; `event_description`
is no longer requested or required. Coverage runs from the sampled `question_time`
through video end, without a fixed maximum number of matches.

Each observation has timestamped `evidence` with `time <= available_time <=
window_start <= trigger_time <= window_end`, plus ability-specific `details`. Complete grounding spans cannot
end after the response. Object counts equal unique current target IDs; action
counts preserve a cumulative occurrence history and counting origin. Predictions,
emotions and causal explanations are labeled as inference rather than observed
facts. Semantic accuracy still requires checking the original video.

### Agent tools and negative samples

Agent questions should express a believable everyday need in brief conversational
language. Choose the need before the tool; do not invent specialist inventories,
logging projects or elaborate personas just to justify calls. Keep schema and
annotation instructions out of the user's wording. Skip when no natural request
fits the scene and capabilities.

Every Agent question must require actual backend tool work. Apply a removal test:
if the frontend can fully satisfy the request from observed video and ordinary
reasoning without a listed capability, redesign the task. Do not add irrelevant
searches, notes or reports to manufacture dependency. Re-reading already available
frames to identify an object is not necessary tool use. External persistence or
transactions must be an explicit, useful requested outcome. This semantic
necessity check is a generation instruction; structural validation enforces the
nonempty tool list, permitted work types and execution evidence, not semantic truth.

`src/agent_tools.py` defines the shared application tools, required named string
parameters, read/write scope and limitations. It includes todo, calendar, notes,
notifications, shopping, budget, routes, weather, public/knowledge/history search,
observed-video segment retrieval, statistics and reports. OCR, ASR, perception and
reasoning are not tool names. The frontend can only call `agent_call`; the backend
can perform `tools` or `tools_and_reasoning`, with nonempty `required_tools`. All execution is simulated.

The program samples capabilities before generation using existing probabilities:
50% `positive`, 25% `no_context`, 25% `missing_tool`. Positive exposes the shared
catalog; `no_context` exposes no Agent function; `missing_tool` exposes Agent with
one required tool removed. Pure reasoning is not an eligible Agent task. Accepted proportions may differ because of skips.

Capability negatives immediately refuse at `question_time` (all three timing
fields equal that time), based only on the
instruction and actual `frontend_context`. They contain no calls or backend
results. A separate root `annotations` array records all real matching triggers, including
brief observational `response` text and ability-specific evidence. These records
are for dataset validation only;
**never expose it, the complete caption, or the generator's full catalog as the
streaming frontend's input**. Without any suitable real trigger, all branches
return `skip`; the pipeline does not create all-video no-trigger silence examples.

Tool results keep `simulated: true`, separate observed facts/tool information/
reasoning, and record read evidence in `sources`, writes in `actions`, with typed
operation scopes and arguments. Claimed success requires evidence for every
required tool. Video retrieval cannot access footage after delegation. Failure
and confirmation-needed results must be reflected in the final `outcome`.

## Merge and verification

Batch mode merges automatically. To merge an explicitly selected directory of
single-task results (one subdirectory per video), or a specific timestamped run
containing per-video `qa/` directories:

```bash
python data_generation/merge_qa.py --data-dir /tmp/omni-v2 \
  --videos-dir data_generation/videos --output /tmp/omni-v2/all_qa.json

PYTHONPATH=.:data_generation python -m unittest discover -s data_generation/tests -v
ruff check data_generation/src data_generation/tests data_generation/scripts
```

Merge revalidates samples, preserves capabilities and annotations, and flattens
Agent messages into timestamped `responses`. It rejects old sample schemas rather
than migrating or deleting them. Caption, route, usage, state and existing QA bundles are excluded from merge
inputs to avoid duplication.
`build_evidence_samples.py` excludes capability refusals, Agent traces and
prediction/inference samples before creating video clips.

Tests use mocked API responses. They establish routing, schema, temporal/capability
contracts, merge and resume behavior, not real-model perception quality. Structural
validation cannot prove free-text truth, completeness of every occurrence, genuine
tool necessity, or absence of future information hidden in prose. These require
review against video prefixes. Multi-model difficulty filtering and a full judging
system are outside this pipeline.
