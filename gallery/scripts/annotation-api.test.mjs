import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { annotationMiddleware } from './annotation-api.mjs'
import { dataMiddleware } from './data-source.mjs'

test('lists QA files, saves each score, resumes, and rejects stale source data', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'annotation-gallery-'))
  const dataDir = path.join(root, 'data')
  const videoDir = path.join(root, 'videos')
  const outputDir = path.join(root, 'annotations')
  await mkdir(dataDir)
  await mkdir(videoDir)
  const source = [{ status: 'ok', question: 'What happened?', video: 'clip.mp4', responses: [
    { trigger_time: '00:02', response: 'The car arrived.' },
    { trigger_time: '00:03', response: '{"status":"success"}', role: 'tool' },
  ] }]
  await writeFile(path.join(dataDir, 'bundle.json'), JSON.stringify(source))
  await writeFile(path.join(dataDir, 'single.json'), JSON.stringify(source[0]))
  await writeFile(path.join(dataDir, 'router.json'), '{"ranked_tasks":[]}')
  await writeFile(path.join(dataDir, 'broken.json'), '{bad')
  await writeFile(path.join(videoDir, 'clip.mp4'), 'fake-video')
  const outside = path.join(root, 'outside.json')
  await writeFile(outside, JSON.stringify(source))
  await symlink(outside, path.join(dataDir, 'link.json'))

  const middleware = annotationMiddleware({ dataDir, videoDir, outputDir })
  const server = createServer((req, res) => middleware(
    req, res, () => dataMiddleware(dataDir)(req, res, () => { res.statusCode = 404; res.end() }),
  ))
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    const base = `http://127.0.0.1:${server.address().port}`
    const request = (url, options) => fetch(base + url, options)
    assert.deepEqual((await (await request('/api/files')).json()).files, ['bundle.json', 'single.json'])
    assert.equal((await (await request('/api/data?file=single.json')).json()).data.question, 'What happened?')
    const { fingerprints } = await (await request('/api/data?file=bundle.json')).json()
    assert.equal(fingerprints[0].length, 2)
    const score = (responseIndex, dimension, value, fingerprint = fingerprints[0][responseIndex]) => request('/api/annotations', {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: 'bundle.json', annotator: 'A', sampleIndex: 0, responseIndex,
        fingerprint, dimension, score: value }),
    })
    assert.equal((await score(0, 'time', 1)).status, 200)
    assert.equal((await score(0, 'content', 0)).status, 200)
    assert.equal((await score(1, 'time', 0)).status, 200)
    const saved = await (await request('/api/annotations?file=bundle.json&annotator=A')).json()
    assert.equal(saved.records['0:0'].timeScore, 1)
    assert.equal(saved.records['0:0'].contentScore, 0)
    assert.equal(saved.records['0:1'].timeScore, 0)
    const concurrent = await Promise.all([score(1, 'time', 1), score(1, 'content', 1)])
    assert.deepEqual(concurrent.map(response => response.status), [200, 200])
    const afterConcurrent = await (await request('/api/annotations?file=bundle.json&annotator=A')).json()
    assert.equal(afterConcurrent.records['0:1'].timeScore, 1)
    assert.equal(afterConcurrent.records['0:1'].contentScore, 1)
    assert.equal((await readFile(path.join(outputDir, 'A/bundle.json.annotations.json'), 'utf8')).includes('timeScore'), true)
    assert.equal((await score(0, 'time', 0)).status, 200)
    assert.equal((await (await request('/api/annotations?file=bundle.json&annotator=A')).json()).records['0:0'].timeScore, 0)
    source[0].responses[0].response = 'The bus arrived.'
    await writeFile(path.join(dataDir, 'bundle.json'), JSON.stringify(source))
    assert.equal((await score(0, 'time', 1)).status, 409)
    const current = await (await request('/api/annotations?file=bundle.json&annotator=A')).json()
    assert.equal(current.records['0:0'], undefined)
    assert.equal(current.records['0:1'].timeScore, 1)
    assert.equal((await request('/api/data?file=..%2Foutside.json')).status, 400)
    assert.equal((await request('/api/data?file=link.json')).status, 403)
  } finally {
    await new Promise(resolve => server.close(resolve))
    await rm(root, { recursive: true, force: true })
  }
})


test('save failure returns an error without modifying the input QA', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'annotation-write-failure-'))
  const dataDir = path.join(root, 'data')
  const videoDir = path.join(root, 'videos')
  const outputDir = path.join(root, 'not-a-directory')
  await mkdir(dataDir)
  await mkdir(videoDir)
  await writeFile(outputDir, 'occupied')
  const source = { status: 'ok', question: 'Q', responses: [{ trigger_time: '00:01', response: 'A' }] }
  const input = path.join(dataDir, 'sample.json')
  await writeFile(input, JSON.stringify(source))
  const middleware = annotationMiddleware({ dataDir, videoDir, outputDir })
  const server = createServer((req, res) => middleware(req, res, () => dataMiddleware(dataDir)(req, res, () => { res.statusCode = 404; res.end() })))
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    const base = `http://127.0.0.1:${server.address().port}`
    const { fingerprints } = await (await fetch(base + '/api/data?file=sample.json')).json()
    const response = await fetch(base + '/api/annotations', {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: 'sample.json', annotator: 'A', sampleIndex: 0, responseIndex: 0,
        fingerprint: fingerprints[0][0], dimension: 'time', score: 1 }),
    })
    assert.equal(response.status, 500)
    assert.deepEqual(JSON.parse(await readFile(input, 'utf8')), source)
  } finally {
    await new Promise(resolve => server.close(resolve))
    await rm(root, { recursive: true, force: true })
  }
})

test('Agent completion writes are rejected while legacy ratings and source are preserved', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'agent-annotation-'))
  const dataDir = path.join(root, 'data')
  const outputDir = path.join(root, 'annotations')
  await mkdir(dataDir)
  await mkdir(path.join(outputDir, 'A'), { recursive: true })
  const source = [
    { mode: 'agent', responses: [
      { role: 'assistant', tool_calls: [{ function: { name: 'agent_call' } }] },
      { role: 'tool', response: '{}' }, { role: 'assistant', outcome: 'success' },
    ] },
    { mode: 'agent', sample_type: 'missing_tool', responses: [{ response: 'Cannot call tool' }] },
    { mode: 'agent', sample_type: 'no_context', responses: [{ response: 'No tools' }] },
  ]
  const input = path.join(dataDir, 'sample.json')
  const raw = JSON.stringify(source)
  await writeFile(input, raw)
  const { responseFingerprints } = await import('./annotation-records.mjs')
  const fingerprints = responseFingerprints(source)
  const legacy = { version: 1, records: { '0:2': { fingerprint: fingerprints[0][2], timeScore: 0, contentScore: 1 } } }
  const annotationPath = path.join(outputDir, 'A/sample.json.annotations.json')
  const legacyRaw = JSON.stringify(legacy)
  await writeFile(annotationPath, legacyRaw)
  const middleware = annotationMiddleware({ dataDir, outputDir })
  const server = createServer((req, res) => middleware(req, res, () => { res.statusCode = 404; res.end() }))
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    const base = `http://127.0.0.1:${server.address().port}`
    const rate = (sampleIndex, responseIndex) => fetch(base + '/api/annotations', {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file: 'sample.json', annotator: 'A', sampleIndex, responseIndex,
        fingerprint: fingerprints[sampleIndex][responseIndex], dimension: 'time', score: 1 }),
    })
    assert.equal((await rate(0, 1)).status, 400)
    assert.equal(await readFile(annotationPath, 'utf8'), legacyRaw)
    assert.equal((await rate(0, 2)).status, 400)
    for (const sampleIndex of [0, 1, 2]) assert.equal((await rate(sampleIndex, 0)).status, 200)
    const current = await (await fetch(base + '/api/annotations?file=sample.json&annotator=A')).json()
    assert.equal(current.records['0:2'].timeScore, 0)
    assert.equal(current.records['0:2'].contentScore, 1)
    assert.equal(current.records['0:2'].fingerprint, fingerprints[0][2])
    assert.equal(current.records['0:0'].timeScore, 1)
    assert.equal(await readFile(input, 'utf8'), raw)
  } finally {
    await new Promise(resolve => server.close(resolve))
    await rm(root, { recursive: true, force: true })
  }
})
