# 36-task audiovisual interaction generation

The pipeline is **original video → dense caption → 36-task routing → task generation
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

`src/catalog.py` is the shared directory for registration, routing and prompt
composition. `route` and `caption` are auxiliary stages, not dataset tasks.
Definitions follow [the v2 task document](<Agentic Omni Interaction 主动交互任务定义 v2 (1).md>).
Prompts combine common rules, an ability specification and a mode template. They
reuse the local paper `tmp/arXiv-2605.18577v1.zip`, appendix `data_details.tex`, and
the previous pipeline's video-first, dense-caption and modality-labeling design.
Examples do not restrict Agent work to particular applications or operations.

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

## Batch generation and resume

```bash
python data_generation/generate_batch.py \
  --videos-dir data_generation/videos --output-dir /tmp/omni-v2-batch \
  --samples-per-task 1 --seed 42

# Use identical configuration to reuse validated artifacts.
python data_generation/generate_batch.py \
  --videos-dir data_generation/videos --output-dir /tmp/omni-v2-batch \
  --samples-per-task 1 --seed 42 --resume

# Restrict generation; routing still evaluates all 36 combinations.
python data_generation/generate_batch.py \
  --videos-dir data_generation/videos --output-dir /tmp/omni-v2-selected \
  --tasks grounding_pr prospective_reasoning_agent --seed 42
```

Execution is sequential and in the foreground. Videos are discovered recursively
(`mp4`, `mov`, `mkv`, `webm`, `avi`). Each gets a collision-resistant folder, a caption,
a complete route and uniquely numbered task files. By default every suitable task
is attempted once; unsupported combinations are not forced onto a video.
`--samples-per-task` counts attempts, not guaranteed accepted samples. There is no
unbounded retry-to-fill loop. Individual failures are recorded and other work
continues; upstream failures mark dependent attempts blocked. Exit status is 1 if
any stage failed, while `skip` and unsuitable routes are normal outcomes.

Outputs include:

- `all_qa.json`: only accepted samples from the current invocation, revalidated and merged.
- `coverage.json`: all 36 tasks, accepted/skip/failed/unsuitable/blocked counts,
  Agent branch counts, errors and artifact reuse information.
- `manifest.json`: fingerprints and content hashes for resumable stage artifacts.

Resume checks source video content, source path, model/API settings, seed/task
selection, implementation and prompt fingerprints, upstream artifacts, output
hashes, sampled context and schema validation. API keys are never stored. Changing
configuration invalidates reuse; regenerating an artifact replaces that batch
output. Use a new output directory to keep multiple runs. Stable per-stage seeds
make sampling reproducible across resume, but do not make remote model output
deterministic. Batch samples store video paths relative to `--videos-dir`.

## Temporal and capability contracts

New task samples use `schema: "omni-interaction-v2"`, `task`, `ability`, `mode`,
`video`, `generation_context`, `question`, `question_time` and `responses`.
`trigger_type` retains its audiovisual meaning (`visual`, `sound`, `speech`, or
combinations); `trigger.kind` separately denotes `event`, `condition`, `count`, or
`target`. Time strings remain `MM:SS`.

- RTD requires at least two meaningful updates. Each has a non-overlapping
  `window_start`/`window_end`, with the end equal to response time.
- PR responds to every independent matching episode after the instruction.
  Persistent conditions require a reset before retriggering. Count predicates
  must transition from false to true; all compound conditions must hold.
- Agent questions scope work to the next matching episode and produce exactly
  three messages: acknowledgement plus `agent_call`, simulated backend result,
  final reply with an `outcome` matching the result.

The former 10–180 s initial silent interval is removed for these new tasks.
`initial_response_delay_seconds` is persisted as zero. Existing start/non-start
question sampling and audio/vision preference remain 50/50; a preference never
overrides an ability's required visual evidence. Backend delay remains sampled
between `max(5 s, ceil(10% duration))` and `floor(50% duration)`; positive samples
skip when no legal trigger and backend completion fit within the video.

Each observation has timestamped `evidence` with `time <= available_time <=
trigger_time`, plus ability-specific `details`. Complete grounding spans cannot
end after the response. Object counts equal unique current target IDs; action
counts preserve a cumulative occurrence history and counting origin. Predictions,
emotions and causal explanations are labeled as inference rather than observed
facts. Semantic accuracy still requires checking the original video.

### Agent tools and negative samples

`src/agent_tools.py` defines the shared application tools, required named string
parameters, read/write scope and limitations. It includes todo, calendar, notes,
notifications, shopping, budget, routes, weather, public/knowledge/history search,
observed-video segment retrieval, statistics and reports. OCR, ASR, perception and
reasoning are not tool names. The frontend can only call `agent_call`; the backend
can perform `tools`, `reasoning`, or `tools_and_reasoning`. All execution is simulated.

The program samples capabilities before generation using existing probabilities:
50% `positive`, 25% `no_context`, 25% `missing_tool`. Positive exposes the shared
catalog; `no_context` exposes no Agent function; `missing_tool` exposes Agent with
one required tool removed. Pure reasoning is valid only in the positive branch
and has empty tool lists. Accepted proportions may differ because of skips.

Capability negatives immediately refuse at `question_time`, based only on the
instruction and actual `frontend_context`. They contain no calls or backend
results. A separate root `annotation` records a real suitable trigger, including
its ability-specific evidence. This annotation is for dataset validation only;
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
single-task results (one subdirectory per video):

```bash
python data_generation/merge_qa.py --data-dir /tmp/omni-v2 \
  --videos-dir data_generation/videos --output /tmp/omni-v2/all_qa.json

PYTHONPATH=.:data_generation python -m unittest discover -s data_generation/tests -v
ruff check data_generation/src data_generation/tests data_generation/scripts
```

Merge revalidates samples, preserves capabilities and annotations, and flattens
Agent messages into timestamped `responses`. It rejects old sample schemas rather
than migrating or deleting them. Caption and route files are not QA entries.
`build_evidence_samples.py` excludes capability refusals, Agent traces and
prediction/inference samples before creating video clips.

Tests use mocked API responses. They establish routing, schema, temporal/capability
contracts, merge and resume behavior, not real-model perception quality. Structural
validation cannot prove free-text truth, completeness of every occurrence, genuine
tool necessity, or absence of future information hidden in prose. These require
review against video prefixes. Multi-model difficulty filtering and a full judging
system are outside this pipeline.
