import { readdir, readFile, realpath, copyFile } from 'node:fs/promises'
import path from 'node:path'
import { responseFingerprints } from './annotation-records.mjs'

export const dataDir = path.resolve(process.env.ANNOTATION_DATA_DIR || path.join(import.meta.dirname, '../../data_generation/data'))
export const videoDirs = process.env.ANNOTATION_VIDEO_DIR
  ? [path.resolve(process.env.ANNOTATION_VIDEO_DIR)]
  : [path.resolve(import.meta.dirname, '../../videos'), path.resolve(import.meta.dirname, '../../data_generation/videos')]

export async function listFiles(root = dataDir) {
  try {
    const files = []
    async function visit(relative = '') {
      for (const entry of await readdir(path.join(root, relative), { withFileTypes: true })) {
        const name = relative ? `${relative}/${entry.name}` : entry.name
        if (entry.isDirectory() && !['workers', 'attempts', 'aborted_attempts'].includes(entry.name) && !entry.name.startsWith('.')) {
          await visit(name)
        } else if (entry.isFile() && entry.name.endsWith('.json') && (!relative || entry.name === 'all_qa.json')) {
          try {
            const data = await readData(name, root)
            if (isFinalTaskResult(data)) files.push(name)
          } catch {
            // Invalid and intermediate JSON files are not gallery samples.
          }
        }
      }
    }
    await visit()
    return files.sort()
  } catch (error) {
    if (error.code === 'ENOENT') return []
    throw error
  }
}

export function isFinalTaskResult(data) {
  if (Array.isArray(data)) {
    return data.length > 0 && data.every((entry) => isFinalTaskResult(entry))
  }
  return Boolean(
    data && typeof data === 'object' && !Array.isArray(data) &&
    'status' in data && 'question' in data && Array.isArray(data.responses) &&
    !('audio_events' in data) && !('visual_events' in data) && !('items' in data),
  )
}

export function validateDataName(name) {
  if (typeof name !== 'string' || !name || path.isAbsolute(name) || name.includes('\\') || name.split('/').some(part => !part || part === '.' || part === '..') || !name.endsWith('.json')) {
    throw Object.assign(new Error('Invalid file name'), { status: 400 })
  }
}

export async function readData(name, root = dataDir) {
  validateDataName(name)
  const resolvedRoot = await realpath(root)
  const file = await realpath(path.join(root, name))
  const relative = path.relative(resolvedRoot, file)
  if (relative.startsWith(`..${path.sep}`) || relative === '..' || path.isAbsolute(relative)) throw Object.assign(new Error('File outside data directory'), { status: 403 })
  const source = await readFile(file, 'utf8')
  try { return JSON.parse(source) } catch {
    throw Object.assign(new Error(`Invalid JSON: ${name}`), { status: 422 })
  }
}

export function dataMiddleware(root = dataDir) {
  return async (req, res, next) => {
    const url = new URL(req.url, 'http://localhost')
    if (!url.pathname.startsWith('/api/')) return next()
    res.setHeader('Content-Type', 'application/json; charset=utf-8')
    res.setHeader('Cache-Control', 'no-store')
    try {
      if (req.method !== 'GET') throw Object.assign(new Error('Method not allowed'), { status: 405 })
      let result
      if (url.pathname === '/api/files') result = { files: await listFiles(root) }
      else if (url.pathname === '/api/data') {
        const data = await readData(url.searchParams.get('file'), root)
        result = { data, fingerprints: responseFingerprints(data) }
      }
      else throw Object.assign(new Error('Not found'), { status: 404 })
      res.end(JSON.stringify(result))
    } catch (error) {
      res.statusCode = error.status || (error.code === 'ENOENT' ? 404 : 500)
      res.end(JSON.stringify({ error: res.statusCode === 500 ? 'Unable to read data directory' : error.message }))
    }
  }
}

export function galleryDataPlugin() {
  const virtualId = '\0gallery-snapshot'
  let config
  return {
    name: 'gallery-data',
    configResolved(value) { config = value },
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        try {
          const name = decodeURIComponent(new URL(req.url, 'http://localhost').pathname).slice(1)
          if (name && name === path.basename(name) && !name.includes('\\') && name.endsWith('.mp4')) {
            for (const root of videoDirs) {
              try {
                await realpath(path.join(root, name))
                req.url = `/@fs/${path.join(root, name)}`
                break
              } catch (error) { if (error.code !== 'ENOENT') throw error }
            }
          }
        } catch { /* Leave invalid URLs to Vite. */ }
        next()
      })
      server.middlewares.use(dataMiddleware())
    },
    configurePreviewServer(server) { server.middlewares.use(dataMiddleware()) },
    resolveId(id) { if (id === 'virtual:gallery-snapshot') return virtualId },
    async load(id) {
      if (id !== virtualId) return
      const snapshot = {}
      // Development always reads live data. Production also embeds an offline snapshot.
      if (process.env.NODE_ENV === 'production') {
        for (const name of await listFiles()) {
          try { snapshot[name] = { data: await readData(name) } }
          catch (error) { snapshot[name] = { error: error.message } }
        }
      }
      return `export default ${JSON.stringify(snapshot)}`
    },
    async closeBundle() {
      if (config.command !== 'build') return
      const videos = new Set()
      for (const name of await listFiles()) {
        const data = await readData(name)
        for (const sample of Array.isArray(data) ? data : [data]) {
          if (typeof sample.video === 'string' && sample.video === path.basename(sample.video)) videos.add(sample.video)
        }
      }
      for (const name of videos) {
        let source
        for (const root of videoDirs) {
          try { source = await realpath(path.join(root, name)); break } catch (error) { if (error.code !== 'ENOENT') throw error }
        }
        if (source) await copyFile(source, path.resolve(config.root, config.build.outDir, name))
        else config.logger.warn(`Gallery video unavailable: ${name}`)
      }
    },
  }
}
