# Dataset generation

The pipeline is **one dense audiovisual caption → routing/task generation → QA
merge**. Every task receives the original video (including its audio) and the
complete caption. Captions are fallible references, never ground truth; generated
responses must use only evidence available at their own timestamps.

## Caption

The prompt follows the dense-caption template in the local paper
`tmp/arXiv-2605.18577v1.zip`, appendix `data_details.tex`. One model request produces
a JSON array with `start`, `end`, `caption`, `visual`, `audio`, and `speech` for each
segment. Segments usually span 5–30 seconds, never exceed 45 seconds, and cover the
whole video without gaps or overlaps. Suggested segment count is `ceil(duration/15)`;
it is guidance, not an enforced count. Timestamps are `MM:SS` (minutes may exceed
59); coverage ends at the video duration rounded to the nearest second, with halves
rounded up. Videos rounding to zero seconds cannot be captioned with this format.

```bash
python data_generation/generate_video_prompts.py data_generation/videos/test.mp4 \
  --output-dir /tmp/caption-prompt

python data_generation/generate.py --task caption \
  --video data_generation/videos/test.mp4 \
  --output data_generation/data/test_video/caption.json
```

The first command only renders `caption.prompt.md` and video attachment instructions.
The second generates and validates `caption.json`. Set `OPENAI_MODEL`,
`OPENAI_API_KEY`, and optionally `OPENAI_BASE_URL`, or supply the corresponding
`--model`, `--api-key`, and `--base-url` arguments. Existing temperature (0.1),
timeout (120 seconds), and retry count (2) are unchanged. Caption requests allow a
JSON array; task requests continue to request a JSON object.

## Twelve tasks

| Trigger family | Non-agent tasks | Agent task |
| --- | --- | --- |
| Event | `event_alert`, `event_narration` | `event_agent` |
| Condition | `semantic_condition_alert` | `condition_agent` |
| Count | `cumulative_counting`, `snapshot_counting`, `deduplicated_counting` | `count_agent` |
| Target | `target_grounding`, `state_monitoring` | `target_agent` |

Task definitions are in [DEFINITION.md](DEFINITION.md). `route` ranks these twelve
tasks; it is an auxiliary stage, not a thirteenth dataset task.

```bash
python data_generation/generate.py --task route \
  --video data_generation/videos/test.mp4 \
  data_generation/data/test_video/caption.json \
  --output data_generation/data/test_video/route.json

python data_generation/generate.py --task event_agent \
  --video data_generation/videos/test.mp4 \
  data_generation/data/test_video/caption.json \
  --output data_generation/data/test_video/event_agent.json
```

Use `--print-prompt` to inspect a sampled request without calling the API. The
caption input is mandatory for every task and route, including state monitoring.
Task outputs include `schema: "dense-caption-v1"`, `task`, `video`, and
`generation_context`, allowing merge-time validation of the sampled constraints.
Use distinct output filenames to retain multiple samples of a task.

## Agent capabilities and negatives

The frontend's only callable function is `agent_call`. It delegates daily tool
work, complex reasoning, or both to the backend. Per-family application tools and
operation scopes are defined in `src/agent_tools.py`; perception and reasoning are
not tool names. Each sample stores:

- `frontend_context`: the actual frontend system prompt and callable function definitions;
- `available_tools`: daily backend tools exposed by that context;
- `required_tools`: tools genuinely required by the requested work;
- `work_type`: `tools`, `reasoning`, or `tools_and_reasoning`;
- `sample_type`: `positive`, `no_context`, or `missing_tool`.

The program selects the branch and available capabilities before generation. A
model cannot add missing tools to the context. The generator sees the task-family
catalog as annotation guidance, but that catalog is not part of the evaluated
frontend's capability context. Use `frontend_context`, not the generator's prompt,
as the capability context when consuming the dataset. Captions remain annotation
references; do not expose future caption segments to a streaming evaluation model.

Defaults in `src/config.py` select 50% positive samples, 25% negatives with no Agent
capability context, and 25% with a partial catalog missing a necessary tool. The
second negative split is conditional on selecting a negative. These are sampling
probabilities, not guaranteed proportions of accepted samples; unsuitable positive
videos may return `skip`. Pure complex-reasoning tasks occur only in the positive
branch and delegate through `agent_call` with empty required/suggested tool lists.

Positive samples contain exactly one trigger episode and three messages:
acknowledgement/delegation, simulated backend result, and final reply. Questions
scope the work to the next matching episode. Counts fire on crossing a threshold
from below; conditions require all necessary conditions; targets require matching
attributes. Backend delay retains the existing range from
`max(5 seconds, ceil(10% of duration))` to `floor(50% of duration)`. Existing question
position, modality, and initial response-delay sampling are preserved.

Negative samples immediately refuse at `question_time`, explain the missing
capability, and contain no tool calls or simulated results. They remain valid
`status: "ok"` samples, not `skip`. No initial silence or backend delay applies.
They cannot rely on future video evidence or use pure reasoning as a pretext for
refusal. Backend results in positive samples always have `simulated: true`; this
pipeline does not execute external application tools.

## Merge and verification

```bash
python data_generation/merge_qa.py
PYTHONPATH=.:data_generation python -m unittest discover -s data_generation/tests -v
ruff check data_generation/src data_generation/tests data_generation/scripts
```

Merge writes `data_generation/data/all_qa.json`. Options `--data-dir`,
`--videos-dir`, and `--output` select other locations. It validates new task
contracts and flattens positive Agent messages into timestamped `responses`, while
preserving capability metadata and negative refusals. It leaves input files intact.
Caption and routing artifacts are not QA entries.

`build_evidence_samples.py` evaluates factual audiovisual responses only. It rejects
capability refusals and Agent message chains (including flattened simulated tool
results) before creating video clips, because these are not video-only claims.

The old events/objects caption format, `--objects`, `--events-json`, and the five
`triggered_*` Agent tasks have been removed. Old generated JSON files are deleted;
source videos and DEFINITION.md are retained. Old samples are rejected, not migrated.
Regenerate captions and task samples before merging a new dataset.

Structural checks catch malformed captions, unavailable-tool calls, inconsistent
trigger/count metadata, and timestamp violations. They cannot prove perceptual
accuracy, completeness of semantic conditions, or truthfulness of free-text claims;
those require reviewing generated samples against the original video. Unit tests
use mocked API results and do not establish real-model generation quality.
