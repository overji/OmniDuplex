import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { tmpdir } from 'node:os'
import { modelOutputsMiddleware, modelResultsForSample } from './model-outputs.mjs'
import { predictionRecords, formatOnset } from '../src/model-output-state.mjs'

const a = { video: 'clip.mp4', task: 'event_rtd', question: 'A?', question_time: '00:01', responses: [{ response: 'A' }] }
const b = { ...a, question: 'B?' }
const prediction = { sample_index: 0, status: 'complete', events: [{ onset_seconds: 1.234, text: '<script>raw text</script>' }] }
const score = { windows: 1, hits: 1, predictions: 1, matches: [{ event_index: 0, response_index: 3, window_start_seconds: 1, window_end_seconds: 2 }] }
const bundle = { version: 1, samples: [a, b], models: [
  { id: 'first', name: 'First', evaluatedSamples: 1, totalSamples: 2, records: [{ prediction, score }] },
  { id: 'silent', name: 'Silent', evaluatedSamples: 1, totalSamples: 2, records: [{ prediction: { ...prediction, events: [] }, score: { ...score, hits: 0, predictions: 0, matches: [] } }] },
] }

test('matches source identity rather than gallery index and distinguishes missing from empty outputs', () => {
  const result = modelResultsForSample(bundle, structuredClone(a))
  assert.equal(result.models[0].prediction.events[0].onset_seconds, 1.234)
  assert.equal(result.models[1].state, 'evaluated')
  assert.deepEqual(result.models[1].prediction.events, [])
  assert.equal(modelResultsForSample(bundle, b).models[0].state, 'not_evaluated')
  assert.equal(modelResultsForSample(bundle, { ...a, responses: [{ response: 'changed' }] }).models[0].state, 'unavailable')
  assert.equal(modelResultsForSample({ ...bundle, samples: [a, a] }, a).models[0].state, 'unavailable')
  assert.equal(modelResultsForSample(bundle, { ...a, question_time: '00:02' }).models[0].state, 'unavailable')
})

test('prediction records preserve raw text, original event order and fractional onset', () => {
  const records = predictionRecords({ prediction, score })
  assert.equal(records[0].seconds, 1.234)
  assert.equal(records[0].trigger_time, '00:01.234')
  assert.equal(records[0].text, '<script>raw text</script>')
  assert.equal(records[0].match.response_index, 3)
  assert.equal(records[0].track, 'predictions')
  assert.equal(predictionRecords({ prediction, score: { matches: [] } })[0].label, 'Unmatched prediction')
  assert.deepEqual(predictionRecords(null), [])
  assert.equal(formatOnset(59.9999), '01:00.000')
  assert.equal(formatOnset(0), '00:00.000')
  assert.equal(formatOnset(-1), '—')
})

test('read-only API handles reordered and single samples, invalid requests and missing or corrupt exports', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'model-output-api-'))
  const dataDir = path.join(root, 'data'), outputDir = path.join(root, 'outputs')
  await mkdir(dataDir); await mkdir(outputDir)
  await writeFile(path.join(dataDir, 'reordered.json'), JSON.stringify([b, a]))
  await writeFile(path.join(dataDir, 'single.json'), JSON.stringify(a))
  const output = path.join(outputDir, 'results.json')
  await writeFile(output, JSON.stringify(bundle))
  const middleware = modelOutputsMiddleware({ dataDir, outputDir })
  const server = createServer((req, res) => middleware(req, res, () => { res.statusCode = 404; res.end() }))
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    const base = `http://127.0.0.1:${server.address().port}/api/model-outputs`
    const request = (file, index) => fetch(`${base}?${new URLSearchParams({ file, sampleIndex: index })}`)
    assert.equal((await (await request('reordered.json', 1)).json()).models[0].prediction.sample_index, 0)
    assert.equal((await (await request('single.json', 0)).json()).models[0].state, 'evaluated')
    assert.equal((await request('single.json', 1)).status, 404)
    assert.equal((await request('../outside.json', 0)).status, 400)
    for (const index of ['-1', '1.2', '', '9007199254740992']) assert.equal((await request('single.json', index)).status, 400)
    assert.equal((await fetch(base + '?file=single.json')).status, 400)
    assert.equal((await fetch(base, { method: 'PATCH' })).status, 405)
    await rm(output)
    assert.deepEqual((await (await request('single.json', 0)).json()).models, [])
    await writeFile(output, '{invalid')
    assert.equal((await request('single.json', 0)).status, 500)
  } finally {
    await new Promise(resolve => server.close(resolve))
    await rm(root, { recursive: true, force: true })
  }
})
