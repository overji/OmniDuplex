# OmniDuplex Gallery

## Live gallery

```bash
npm install
npm run dev
```

The left sidebar lists final task JSON files at the root of `../data_generation/data`
and `all_qa.json` bundles in its subdirectories. Worker and attempt directories are
excluded. A JSON array contains one sample per entry; a JSON object is one sample.
The current Qwen 44-sample bundle is selected on first use; a valid saved selection
takes precedence. Filters select ability, RTD/PR/Agent mode, positive/negative
sample type, and video. Agent negative subtypes are labeled `missing_tool` or
`no_context`.

The timeline displays **valid response windows**, not event durations:
`window_start` is the earliest supported response time, `trigger_time` is the
recommended response time, and `window_end` is the latest useful response time.
Window bands contain recommended-time diamonds; overlapping windows use separate
lanes. Cards provide three independent seek buttons and highlight only while
playback is inside their window. Zero-length windows appear as instant markers.
Messages without windows use point markers without invented boundaries.

RTD/PR cards show responses, evidence occurrence/availability times, trigger
criteria and task details. Agent samples distinguish delegation, **simulated**
backend results and final replies. Negative samples separate the refusal from
observation `annotations`; these annotations are not Agent actions. Tool JSON is
shown as structured data when parseable and preserved as text otherwise. All
original fields and raw JSON remain available in collapsible sections. The source
files are never modified.

Refresh the page to discover added, changed or deleted files. The last selected
file and sample are restored when they still exist. Invalid JSON is reported
for that file without preventing selection of another file.

## Annotate responses

Use **Browse / Annotate** above the existing sample filters. Browse restores its
selected sample. Annotate selects the first response missing either score in the
current file and filter scope, in original sample/response order. Select annotator
**A** or **B**; each has independent scores and progress. Agent assistant delegation
and completion replies can both be rated. Simulated tool results are inputs and
remain visible without ratings.
Agent negative samples (`missing_tool` and `no_context`) still score refusal responses.
RTD/PR scoring is unchanged. Observation `annotations` remain visible without ratings.
Excluded responses do not contribute to progress or agreement statistics; their
existing saved scores are preserved. Original response indices and fingerprints
remain unchanged.

Each response card accepts **Pass (1) / Fail (0)** for timing and content. Each
click saves immediately to the server. After both scores save, the next unfinished
response is selected and the video seeks to its time. You can select any completed
response to revise it. Save errors leave the response available for retry; stale
source data requires a page reload. Scores refresh every five seconds for other
open annotator sessions. A and B are shared labels, not authenticated accounts.

The summary uses the current file and filters. It shows the selected annotator's
completion and pass rates, plus A/B paired-response counts, direct agreement,
unweighted Cohen's Kappa, and Pearson correlation (Phi for binary scores). Timing
and content pairings are independent; an unscored value is never treated as a fail.
No pairs, zero variance, or an undefined denominator produce **N/A** with a reason.
These metrics measure agreement between two human annotators, not agreement with
an assumed ground truth. Keep A and B independent when evaluating reliability.

Source QA files remain read-only. Scores are written atomically under
`annotations/<A|B>/<source-relative-path>.annotations.json`, with the existing
response fingerprint, two scores and update time. Changed responses invalidate
old scores. Subdirectory bundles with the same filename use separate output paths.
Keep this output directory when rebuilding or moving the app.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `ANNOTATION_DATA_DIR` | `../data_generation/data` | Shared browsing and annotation input directory |
| `ANNOTATION_VIDEO_DIR` | Repository-root `videos`, then `data_generation/videos` | Override the shared video directory |
| `ANNOTATION_OUTPUT_DIR` | `gallery/annotations` | Persistent A/B annotation output |

Set variables before `dev`, `build`, `preview`, or `start`; rebuild when changing
video sources for a production export. The input directory follows the same final
sample and nested `all_qa.json` discovery rules in both modes. If overridden, the
sidebar still labels the source as live data; paths are configured on the server.

The former `annotation_gallery` application is merged into this gallery. To import
an older output directory before deleting it, run:

```bash
node scripts/migrate-annotations.mjs /path/to/old/annotations
```

Legacy files are copied byte-for-byte to annotator A and backed up under
`annotation-backups/<timestamp>`. Conflicting destination files are reported and
never overwritten. Legacy fingerprints are preserved. Annotation outputs and
backups are excluded from Git. Do not infer nested source paths from matching
basenames; old files correspond to their original top-level source files.

## Serve a build with live data

```bash
npm run build
npm start
# Optional network access and port:
npm start -- --host 0.0.0.0 --port 4173
```

`npm start` uses Vite's preview server and the same live data middleware as
development. Host and port follow `preview` in `vite.config.js` unless supplied
on the command line. `npm run preview` also supports live data.

Keep the gallery source/configuration, installed dependencies, and sibling
`data_generation/data` directory on the server. Builds copy only referenced videos into `dist`, preferring the repository-root
`videos` directory and falling back to `data_generation/videos`; keep those source
directories available when rebuilding. Data is read fresh on each request; data changes need no rebuild or
service restart. Task generation writes the input video's filename to each
sample's `video` field.

## Portable snapshot

`npm run build` also embeds a snapshot of final task JSON files, JavaScript, and CSS
in `dist/index.html`. Copy the complete `dist` directory and open `index.html`
directly, or serve it from a static-server subpath. Keep the referenced video
files beside `index.html`. The UI labels this mode as an offline snapshot.
Annotation controls require the live server; portable snapshots are browse-only. Rebuild to update
its data; only the live service can read changes in the source directory.

## Checks

```bash
npm test
npm run build
```

## View model outputs

Select **Model outputs**, choose a sample and a model. The reference responses
remain visible alongside read-only predictions, including unmatched replies.
Prediction cards and a separate timeline track seek to the original fractional
video onset. Scores use the benchmark's official timing matching, not semantic
correctness. Missing samples, unavailable matches and evaluated samples with no
response are shown separately. Offline snapshots do not include this feature.

Export the configured five model runs before starting the live gallery:

```bash
python scripts/export-model-outputs.py
```

`model-runs.json` configures the dataset and experiment directories; paths are
relative to that configuration file. The script supports `--config` and
`--output-dir`. Generated `model_outputs/results.json` contains original
predictions and official scores, without modifying source predictions. Upload
this directory alongside the app for remote viewing; Python and original
benchmark directories are needed only when exporting, not for serving.

`MODEL_OUTPUT_DIR` overrides the server's default `gallery/model_outputs` directory.
The read-only `/api/model-outputs?file=…&sampleIndex=…` endpoint matches samples by
video, task, question, question time and reference responses. Reordered or single
sample files therefore retain the correct outputs; changed or ambiguous samples
are not matched. Export again when results change. Model outputs never write to
`annotations/` or contribute to human annotation statistics.
