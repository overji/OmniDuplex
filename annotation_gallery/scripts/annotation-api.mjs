import { createHash, randomUUID } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, readdir, realpath, rename, stat, writeFile } from 'node:fs/promises'
import path from 'node:path'

const fail = (status, message) => Object.assign(new Error(message), { status })

function safeName(name, extension = '') {
  if (typeof name !== 'string' || !name || name !== path.basename(name) ||
      name.includes('\\') || (extension && !name.endsWith(extension))) {
    throw fail(400, 'Invalid file name')
  }
  return name
}

async function confinedFile(root, name) {
  const rootPath = await realpath(root)
  const file = await realpath(path.join(rootPath, name))
  if (path.dirname(file) !== rootPath) throw fail(403, 'File outside configured directory')
  if (!(await stat(file)).isFile()) throw fail(400, 'Expected a file')
  return file
}

export function isQaData(data) {
  if (Array.isArray(data)) return data.length > 0 && data.every(isQaData)
  return Boolean(data && typeof data === 'object' &&
    typeof data.question === 'string' && Array.isArray(data.responses) &&
    data.responses.every(response => response && typeof response === 'object' &&
      typeof response.trigger_time === 'string' && typeof response.response === 'string') &&
    !('audio_events' in data) && !('visual_events' in data) && !('items' in data))
}

export function fingerprint(sample, response) {
  return createHash('sha256').update(JSON.stringify([
    sample.video, sample.question, sample.task, response,
  ])).digest('hex')
}

export async function listFiles(dataDir) {
  try {
    const names = []
    for (const entry of await readdir(dataDir, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith('.json')) continue
      try {
        if (isQaData(JSON.parse(await readFile(path.join(dataDir, entry.name), 'utf8')))) {
          names.push(entry.name)
        }
      } catch {
        // Invalid and intermediate JSON files are not annotation inputs.
      }
    }
    return names.sort()
  } catch (error) {
    if (error.code === 'ENOENT') return []
    throw error
  }
}

export async function readInput(dataDir, name) {
  const file = await confinedFile(dataDir, safeName(name, '.json'))
  let data
  try { data = JSON.parse(await readFile(file, 'utf8')) }
  catch { throw fail(422, `Invalid JSON: ${name}`) }
  if (!isQaData(data)) throw fail(422, `Not a QA file: ${name}`)
  const samples = Array.isArray(data) ? data : [data]
  const fingerprints = samples.map(sample => sample.responses.map(
    response => fingerprint(sample, response),
  ))
  return { data, samples, fingerprints }
}

async function readAnnotations(outputDir, name) {
  try {
    const result = JSON.parse(await readFile(path.join(outputDir, `${name}.annotations.json`), 'utf8'))
    if (!result || typeof result !== 'object' || Array.isArray(result) ||
        !result.records || typeof result.records !== 'object' || Array.isArray(result.records)) {
      throw new Error('Invalid annotation file')
    }
    return result
  } catch (error) {
    if (error.code === 'ENOENT') return { version: 1, records: {} }
    throw error
  }
}

function currentAnnotations(stored, fingerprints) {
  const records = {}
  for (const [key, record] of Object.entries(stored.records)) {
    const [sampleIndex, responseIndex] = key.split(':').map(Number)
    if (Number.isInteger(sampleIndex) && Number.isInteger(responseIndex) &&
        fingerprints[sampleIndex]?.[responseIndex] === record?.fingerprint) {
      records[key] = record
    }
  }
  return records
}

async function readBody(req) {
  let text = ''
  for await (const chunk of req) {
    text += chunk
    if (text.length > 16_384) throw fail(413, 'Request body too large')
  }
  try { return JSON.parse(text) }
  catch { throw fail(400, 'Invalid JSON body') }
}

async function serveVideo(req, res, videoDir, name) {
  safeName(name)
  const file = await confinedFile(videoDir, name)
  const { size } = await stat(file)
  const mime = { '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime' }[path.extname(name).toLowerCase()]
  if (!mime) throw fail(415, 'Unsupported video format')
  let start = 0
  let end = size - 1
  if (req.headers.range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range)
    if (!match || (!match[1] && !match[2])) throw fail(416, 'Invalid video range')
    if (!match[1]) {
      const suffix = Number(match[2])
      if (!suffix) throw fail(416, 'Invalid video range')
      start = Math.max(0, size - suffix)
    } else {
      start = Number(match[1])
      if (match[2]) end = Math.min(Number(match[2]), end)
    }
    if (start >= size || end < start) throw fail(416, 'Invalid video range')
    res.statusCode = 206
    res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`)
  }
  res.setHeader('Content-Type', mime)
  res.setHeader('Accept-Ranges', 'bytes')
  res.setHeader('Content-Length', end - start + 1)
  if (req.method === 'HEAD') return res.end()
  createReadStream(file, { start, end }).on('error', () => res.destroy()).pipe(res)
}

export function annotationMiddleware({ dataDir, videoDir, outputDir }) {
  let writeQueue = Promise.resolve()
  return async (req, res, next) => {
    const url = new URL(req.url, 'http://localhost')
    if (!url.pathname.startsWith('/api/')) return next()
    res.setHeader('Cache-Control', 'no-store')
    try {
      if (url.pathname === '/api/video') {
        if (!['GET', 'HEAD'].includes(req.method)) throw fail(405, 'Method not allowed')
        return await serveVideo(req, res, videoDir, url.searchParams.get('name'))
      }
      let result
      if (req.method === 'GET' && url.pathname === '/api/files') {
        result = { files: await listFiles(dataDir) }
      } else if (req.method === 'GET' && url.pathname === '/api/data') {
        const { data, fingerprints } = await readInput(dataDir, url.searchParams.get('file'))
        result = { data, fingerprints }
      } else if (req.method === 'GET' && url.pathname === '/api/annotations') {
        const name = safeName(url.searchParams.get('file'), '.json')
        const { fingerprints } = await readInput(dataDir, name)
        result = { records: currentAnnotations(await readAnnotations(outputDir, name), fingerprints) }
      } else if (req.method === 'PATCH' && url.pathname === '/api/annotations') {
        const body = await readBody(req)
        if (!body || typeof body !== 'object') throw fail(400, 'Invalid annotation')
        const { file, sampleIndex, responseIndex, dimension, score } = body
        safeName(file, '.json')
        if (!Number.isInteger(sampleIndex) || sampleIndex < 0 ||
            !Number.isInteger(responseIndex) || responseIndex < 0 ||
            !['time', 'content'].includes(dimension) || ![0, 1].includes(score) ||
            typeof body.fingerprint !== 'string') throw fail(400, 'Invalid annotation')
        const { fingerprints } = await readInput(dataDir, file)
        const expected = fingerprints[sampleIndex]?.[responseIndex]
        if (!expected) throw fail(404, 'Response not found')
        if (expected !== body.fingerprint) throw fail(409, 'Source response changed; reload the file')
        const save = async () => {
          const current = await readInput(dataDir, file)
          if (current.fingerprints[sampleIndex]?.[responseIndex] !== expected) {
            throw fail(409, 'Source response changed; reload the file')
          }
          const stored = await readAnnotations(outputDir, file)
          const key = `${sampleIndex}:${responseIndex}`
          const previous = stored.records[key]?.fingerprint === expected ? stored.records[key] : {}
          const record = {
            ...previous, fingerprint: expected, [`${dimension}Score`]: score,
            updatedAt: new Date().toISOString(),
          }
          stored.records[key] = record
          await mkdir(outputDir, { recursive: true })
          const temporary = path.join(outputDir, `.${file}.${randomUUID()}.tmp`)
          await writeFile(temporary, JSON.stringify(stored, null, 2) + '\n', 'utf8')
          await rename(temporary, path.join(outputDir, `${file}.annotations.json`))
          return { key, record }
        }
        const pending = writeQueue.then(save)
        writeQueue = pending.catch(() => {})
        result = await pending
      } else {
        throw fail(404, 'Not found')
      }
      res.setHeader('Content-Type', 'application/json; charset=utf-8')
      res.end(JSON.stringify(result))
    } catch (error) {
      res.statusCode = error.status || (error.code === 'ENOENT' ? 404 : 500)
      res.setHeader('Content-Type', 'application/json; charset=utf-8')
      res.end(JSON.stringify({ error: res.statusCode === 500 ? 'Unable to process request' : error.message }))
    }
  }
}
