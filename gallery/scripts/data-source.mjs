import { readdir, readFile, realpath } from 'node:fs/promises'
import path from 'node:path'

export const dataDir = path.resolve(import.meta.dirname, '../../data_generation/data')

export async function listFiles(root = dataDir) {
  try {
    const files = []
    for (const entry of await readdir(root, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith('.json')) continue
      try {
        const data = JSON.parse(await readFile(path.join(root, entry.name), 'utf8'))
        if (isFinalTaskResult(data)) files.push(entry.name)
      } catch {
        // Invalid and intermediate JSON files are not gallery samples.
      }
    }
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

export async function readData(name, root = dataDir) {
  if (!name || name !== path.basename(name) || name.includes('\\') || !name.endsWith('.json')) {
    throw Object.assign(new Error('Invalid file name'), { status: 400 })
  }
  const resolvedRoot = await realpath(root)
  const file = await realpath(path.join(root, name))
  if (path.dirname(file) !== resolvedRoot) throw Object.assign(new Error('File outside data directory'), { status: 403 })
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
      else if (url.pathname === '/api/data') result = { data: await readData(url.searchParams.get('file'), root) }
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
  return {
    name: 'gallery-data',
    configureServer(server) { server.middlewares.use(dataMiddleware()) },
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
  }
}
