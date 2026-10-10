import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { annotationMiddleware } from './annotation-api.mjs'
import { dataMiddleware } from './data-source.mjs'
import { migrateAnnotations } from './migrate-annotations.mjs'
import { responseFingerprints } from './annotation-records.mjs'

const source = [{ status: 'ok', video: 'clip.mp4', question: 'Watch', responses: [
  { response: 'Answer', trigger_time: '00:01' }, { response: '{}', role: 'tool', trigger_time: '00:02' },
] }]

test('nested bundles isolate annotators and paths, with concurrent updates and stale-score rejection', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'gallery-annotation-integration-'))
  const dataDir = path.join(root, 'data'), outputDir = path.join(root, 'output')
  await mkdir(path.join(dataDir, 'first'), { recursive: true })
  await mkdir(path.join(dataDir, 'second'))
  for (const folder of ['first', 'second']) await writeFile(path.join(dataDir, folder, 'all_qa.json'), JSON.stringify(source))
  const annotation = annotationMiddleware({ dataDir, outputDir })
  const data = dataMiddleware(dataDir)
  const server = createServer((req, res) => annotation(req, res, () => data(req, res, () => { res.statusCode = 404; res.end() })))
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    const base = `http://127.0.0.1:${server.address().port}`
    const get = async (file, annotator) => (await (await fetch(`${base}/api/annotations?file=${encodeURIComponent(file)}&annotator=${annotator}`)).json()).records
    const fingerprints = (await (await fetch(base + '/api/data?file=first%2Fall_qa.json')).json()).fingerprints
    const score = (file, annotator, dimension, value, extra = {}) => fetch(base + '/api/annotations', {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file, annotator, sampleIndex: 0, responseIndex: 0, dimension, score: value,
        fingerprint: fingerprints[0][0], ...extra }),
    })
    assert.deepEqual((await (await fetch(base + '/api/files')).json()).files, ['first/all_qa.json', 'second/all_qa.json'])
    const saves = await Promise.all([score('first/all_qa.json', 'A', 'time', 1), score('first/all_qa.json', 'A', 'content', 0), score('first/all_qa.json', 'B', 'time', 0)])
    assert.deepEqual(saves.map(response => response.status), [200, 200, 200])
    assert.equal((await get('first/all_qa.json', 'A'))['0:0'].contentScore, 0)
    assert.equal((await get('first/all_qa.json', 'A'))['0:0'].timeScore, 1)
    assert.equal((await get('first/all_qa.json', 'B'))['0:0'].timeScore, 0)
    assert.equal((await get('first/all_qa.json', 'B'))['0:0'].contentScore, undefined)
    assert.deepEqual(await get('second/all_qa.json', 'A'), {})
    assert.equal((await score('first/all_qa.json', 'C', 'time', 1)).status, 400)
    assert.equal((await score('../escape.json', 'A', 'time', 1)).status, 400)
    assert.equal((await score('first/all_qa.json', 'A', 'time', 2)).status, 400)
    const changed = structuredClone(source); changed[0].responses[0].response = 'Changed'
    await writeFile(path.join(dataDir, 'first/all_qa.json'), JSON.stringify(changed))
    assert.equal((await score('first/all_qa.json', 'A', 'time', 1)).status, 409)
    assert.deepEqual(await get('first/all_qa.json', 'A'), {})
    assert.deepEqual(await get('first/all_qa.json', 'B'), {})
    // Restore the input, then reject output-directory symlinks before any outside writes.
    await writeFile(path.join(dataDir, 'first/all_qa.json'), JSON.stringify(source))
    const outside = path.join(root, 'outside'); await mkdir(outside)
    await symlink(outside, path.join(outputDir, 'A/second'))
    assert.equal((await score('second/all_qa.json', 'A', 'time', 1)).status, 403)
    await assert.rejects(readFile(path.join(outside, 'all_qa.json.annotations.json')), { code: 'ENOENT' })
    assert.deepEqual(JSON.parse(await readFile(path.join(dataDir, 'first/all_qa.json'), 'utf8')), source)
  } finally {
    await new Promise(resolve => server.close(resolve))
    await rm(root, { recursive: true, force: true })
  }
})

test('legacy scores migrate byte-for-byte to A with a backup and without overwriting conflicts', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'gallery-annotation-migration-'))
  try {
    const legacy = path.join(root, 'legacy'), output = path.join(root, 'output')
    await mkdir(legacy)
    const stored = { version: 1, records: { '0:0': { fingerprint: responseFingerprints(source)[0][0], timeScore: 1, contentScore: 0 } } }
    const bytes = JSON.stringify(stored, null, 2) + '\n'
    await writeFile(path.join(legacy, 'old.json.annotations.json'), bytes)
    assert.deepEqual(await migrateAnnotations(legacy, output, path.join(root, 'backup')), { migrated: ['old.json'], conflicts: [] })
    assert.equal(await readFile(path.join(output, 'A/old.json.annotations.json'), 'utf8'), bytes)
    assert.equal(await readFile(path.join(root, 'backup/old.json.annotations.json'), 'utf8'), bytes)
    await writeFile(path.join(output, 'A/old.json.annotations.json'), '{"version":1,"records":{}}')
    assert.deepEqual(await migrateAnnotations(legacy, output, path.join(root, 'backup-2')), { migrated: [], conflicts: ['old.json'] })
    assert.equal(await readFile(path.join(output, 'A/old.json.annotations.json'), 'utf8'), '{"version":1,"records":{}}')
    assert.equal(await readFile(path.join(legacy, 'old.json.annotations.json'), 'utf8'), bytes)
  } finally { await rm(root, { recursive: true, force: true }) }
})
