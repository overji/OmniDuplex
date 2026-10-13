import { randomUUID } from 'node:crypto'
import { mkdir, readFile, realpath, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { readData, validateDataName, dataDir as defaultDataDir } from './data-source.mjs'
import { responseFingerprints, currentAnnotations } from './annotation-records.mjs'
import { isScorableResponse } from '../src/annotation-state.mjs'

const fail = (status, message) => Object.assign(new Error(message), { status })
export const outputDir = path.resolve(process.env.ANNOTATION_OUTPUT_DIR || path.join(import.meta.dirname, '../annotations'))

function annotatorName(value) {
  if (!['A', 'B'].includes(value)) throw fail(400, 'Select annotator A or B')
  return value
}

// Resolve each directory before reading or writing; an output symlink cannot escape the root.
export async function annotationFile(root, annotator, name, create = false) {
  annotatorName(annotator)
  validateDataName(name)
  if (create) await mkdir(root, { recursive: true })
  const resolvedRoot = await realpath(root)
  let directory = resolvedRoot
  for (const part of [annotator, ...name.split('/').slice(0, -1)]) {
    const candidate = path.join(directory, part)
    if (create) await mkdir(candidate, { recursive: true })
    directory = await realpath(candidate)
    const relative = path.relative(resolvedRoot, directory)
    if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) throw fail(403, 'Annotation outside output directory')
  }
  const file = path.join(directory, `${path.basename(name)}.annotations.json`)
  try {
    const resolved = await realpath(file)
    if (path.dirname(resolved) !== directory) throw fail(403, 'Annotation outside output directory')
  } catch (error) { if (error.code !== 'ENOENT') throw error }
  return file
}

export async function readAnnotations(root, annotator, name) {
  try {
    const result = JSON.parse(await readFile(await annotationFile(root, annotator, name), 'utf8'))
    if (!result || typeof result !== 'object' || Array.isArray(result) ||
        !result.records || typeof result.records !== 'object' || Array.isArray(result.records)) throw fail(500, 'Invalid annotation file')
    return result
  } catch (error) {
    if (error.code === 'ENOENT') return { version: 1, records: {} }
    throw error
  }
}

async function readBody(req) {
  let text = ''
  for await (const chunk of req) {
    text += chunk
    if (text.length > 16_384) throw fail(413, 'Request body too large')
  }
  try { return JSON.parse(text) } catch { throw fail(400, 'Invalid JSON body') }
}

export function annotationMiddleware({ dataDir = defaultDataDir, outputDir: root = outputDir } = {}) {
  let writeQueue = Promise.resolve()
  return async (req, res, next) => {
    const url = new URL(req.url, 'http://localhost')
    if (url.pathname !== '/api/annotations') return next()
    res.setHeader('Cache-Control', 'no-store')
    res.setHeader('Content-Type', 'application/json; charset=utf-8')
    try {
      let result
      if (req.method === 'GET') {
        const name = url.searchParams.get('file')
        const annotator = annotatorName(url.searchParams.get('annotator'))
        const fingerprints = responseFingerprints(await readData(name, dataDir))
        result = { records: currentAnnotations(await readAnnotations(root, annotator, name), fingerprints) }
      } else if (req.method === 'PATCH') {
        const body = await readBody(req)
        if (!body || typeof body !== 'object' || Array.isArray(body)) throw fail(400, 'Invalid annotation')
        const { file, sampleIndex, responseIndex, dimension, score } = body
        const annotator = annotatorName(body.annotator)
        validateDataName(file)
        if (!Number.isInteger(sampleIndex) || sampleIndex < 0 || !Number.isInteger(responseIndex) || responseIndex < 0 ||
            !['time', 'content'].includes(dimension) || ![0, 1].includes(score) || typeof body.fingerprint !== 'string') throw fail(400, 'Invalid annotation')
        const save = async () => {
          const data = await readData(file, dataDir)
          const samples = Array.isArray(data) ? data : [data]
          const fingerprints = responseFingerprints(data)
          const expected = fingerprints[sampleIndex]?.[responseIndex]
          if (!expected) throw fail(404, 'Response not found')
          if (expected !== body.fingerprint) throw fail(409, 'Source response changed; reload the file')
          if (!isScorableResponse(samples[sampleIndex], samples[sampleIndex].responses[responseIndex])) {
            throw fail(400, 'Agent tool results are inputs and are not scored')
          }
          const stored = await readAnnotations(root, annotator, file)
          const key = `${sampleIndex}:${responseIndex}`
          const previous = stored.records[key]?.fingerprint === expected ? stored.records[key] : {}
          const record = { ...previous, fingerprint: expected, [`${dimension}Score`]: score, updatedAt: new Date().toISOString() }
          stored.records[key] = record
          const target = await annotationFile(root, annotator, file, true)
          const temporary = path.join(path.dirname(target), `.${randomUUID()}.tmp`)
          try {
            await writeFile(temporary, JSON.stringify(stored, null, 2) + '\n', { encoding: 'utf8', flag: 'wx' })
            await rename(temporary, target)
          } finally { await rm(temporary, { force: true }) }
          return { key, record }
        }
        const pending = writeQueue.then(save)
        writeQueue = pending.catch(() => {})
        result = await pending
      } else throw fail(405, 'Method not allowed')
      res.end(JSON.stringify(result))
    } catch (error) {
      res.statusCode = error.status || (error.code === 'ENOENT' ? 404 : 500)
      res.end(JSON.stringify({ error: res.statusCode === 500 ? 'Unable to save or read annotations' : error.message }))
    }
  }
}
