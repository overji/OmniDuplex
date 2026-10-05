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
files beside `index.html`. The UI labels this mode as an offline snapshot. Rebuild to update
its data; only the live service can read changes in the source directory.

## Checks

```bash
npm test
npm run build
```
