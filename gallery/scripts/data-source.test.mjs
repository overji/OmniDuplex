import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, writeFile, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { createServer } from 'node:http'
import { dataMiddleware } from './data-source.mjs'

test('live API discovers changes, isolates invalid files and rejects traversal', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'gallery-data-'))
  const outside = await mkdtemp(path.join(tmpdir(), 'gallery-outside-'))
  const middleware = dataMiddleware(root)
  const server = createServer((req, res) => middleware(req, res, () => { res.statusCode = 404; res.end() }))
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
    const origin = `http://127.0.0.1:${server.address().port}`
    const get = (url) => fetch(origin + url)
    assert.deepEqual(await (await get('/api/files')).json(), { files: [] })
    await writeFile(path.join(root, 'b.json'), '{"status":"ok","question":"first","responses":[]}')
    await writeFile(path.join(root, 'a.json'), '{"responses":[]}')
    await writeFile(path.join(root, 'ignored.txt'), 'ignored')
    let response = await get('/api/files')
    assert.equal(response.headers.get('cache-control'), 'no-store')
    assert.deepEqual(await response.json(), { files: ['b.json'] })
    assert.equal((await (await get('/api/data?file=b.json')).json()).data.question, 'first')
    await writeFile(path.join(root, 'b.json'), '[]')
    assert.deepEqual((await (await get('/api/data?file=b.json')).json()).data, [])
    await writeFile(path.join(root, 'b.json'), '{unfinished')
    assert.equal((await get('/api/data?file=b.json')).status, 422)
    assert.equal((await get('/api/data?file=a.json')).status, 200)
    await writeFile(path.join(root, 'bundle.json'), JSON.stringify([
      { status: 'ok', question: 'first', responses: [], video: 'video_550.mp4', task: 'event_alert' },
      { status: 'skip', reason: 'not applicable', question: 'second', responses: [], video: 'video_551.mp4', task: 'event_narration' },
    ]))
    assert.deepEqual((await (await get('/api/files')).json()).files, ['bundle.json'])
    assert.equal((await (await get('/api/data?file=bundle.json')).json()).data.length, 2)
    await rm(path.join(root, 'b.json'))
    await rm(path.join(root, 'bundle.json'))
    assert.deepEqual((await (await get('/api/files')).json()).files, [])
    assert.equal((await get('/api/data?file=b.json')).status, 404)
    for (const name of ['../secret.json', '/etc/private.json', '..\\secret.json']) {
      assert.equal((await get(`/api/data?file=${encodeURIComponent(name)}`)).status, 400)
    }
    await writeFile(path.join(outside, 'secret.json'), '{}')
    await symlink(path.join(outside, 'secret.json'), path.join(root, 'link.json'))
    assert.equal((await get('/api/data?file=link.json')).status, 403)
    assert.deepEqual((await (await get('/api/files')).json()).files, [])
    assert.equal((await fetch(origin + '/api/files', { method: 'POST' })).status, 405)
  } finally {
    await new Promise(resolve => server.close(resolve))
    await rm(root, { recursive: true, force: true })
    await rm(outside, { recursive: true, force: true })
  }
})
