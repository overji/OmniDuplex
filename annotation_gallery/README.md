# Annotation Gallery

This app reviews each `response` in a QA JSON file against its video. It accepts
a single QA object or an array such as `data_generation/data/all_qa.json`. Both
the trigger time and the response content receive a pass/fail score. Tool
responses are included. Each click saves to a separate server-side JSON file;
the input QA files are never changed.

```bash
cd annotation_gallery
npm install
npm run dev
```

Open the URL printed by Vite. The file sidebar lists QA JSON files from
`../data_generation/data`; videos come from `../data_generation/videos`. Opening
a file starts at its first response missing either score. After both scores
save successfully, the page moves to the next unfinished response. Completed
responses remain available for review and edits.

Set these environment variables before starting the app to use other locations:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANNOTATION_DATA_DIR` | `../data_generation/data` | JSON input directory |
| `ANNOTATION_VIDEO_DIR` | `../data_generation/videos` | Video directory |
| `ANNOTATION_OUTPUT_DIR` | `annotation_gallery/annotations` | Saved annotation files |

Each input file gets `<filename>.annotations.json` in the output directory.
Records are keyed by sample and response index and include a source fingerprint,
the two scores (`0` = fail, `1` = pass), and an update time. If source content
changes, old scores are ignored for that response and the page asks for a reload
before accepting another score. Keep the output directory when moving or
rebuilding the app. This first version uses one shared annotation set and shows
completion and pass rates. It does not calculate Kappa or correlation.

```bash
npm test
npm run build
npm start
```

Use the running `dev` or `start` server for annotation. A static `dist` file
opened directly cannot save scores. `npm start -- --host 0.0.0.0 --port 4173`
can set the preview address and port.
