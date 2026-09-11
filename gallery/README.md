# OmniDuplex Gallery

## Live gallery

```bash
npm install
npm run dev
```

The left sidebar lists JSON files in `../data_generation/data`. Select a file,
then use the sample buttons at the top of the main panel to select a complete
sample. A JSON array contains one sample per entry; a JSON object is one sample.
The gallery shows all fields, raw JSON, and the existing `panorama.mp4` video.
Timestamp buttons seek the video. The source files are never modified.

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
`data_generation/data` directory on the server. Builds copy videos from
`data_generation/videos` into `dist`; keep that source directory available when
rebuilding. Data is read fresh on each request; data changes need no rebuild or
service restart. The current JSON files do not identify a video, so samples
continue to use `panorama.mp4`, visibly labeled in the player.

## Portable snapshot

`npm run build` also embeds a snapshot of all JSON files, JavaScript, and CSS
in `dist/index.html`. Copy the complete `dist` directory and open `index.html`
directly, or serve it from a static-server subpath. Keep `panorama.mp4` beside
`index.html`. The UI labels this mode as an offline snapshot. Rebuild to update
its data; only the live service can read changes in the source directory.

## Checks

```bash
npm test
npm run build
```
