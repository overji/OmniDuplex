import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { isDeepStrictEqual } from 'node:util'
import { readData, dataDir as defaultDataDir } from './data-source.mjs'

export const modelOutputDir = path.resolve(process.env.MODEL_OUTPUT_DIR || path.join(import.meta.dirname, '../model_outputs'))
const fail = (status, message) => Object.assign(new Error(message), { status })

export function modelResultsForSample(bundle, sample) {
  const identities = ['video', 'task', 'question', 'question_time', 'responses']
  const indices = bundle.samples.flatMap((source, index) =>
    identities.every(key => isDeepStrictEqual(source[key], sample[key])) ? [index] : [])
  const sampleIndex = indices.length === 1 ? indices[0] : null
  return { message: sampleIndex === null ? 'No unique matching benchmark sample for this data.' : '',
    models: bundle.models.map(({ records, ...model }) => {
      const record = sampleIndex === null ? null : records.find(row => row.prediction.sample_index === sampleIndex)
      return { ...model, state: sampleIndex === null ? 'unavailable' : !record ? 'not_evaluated' : 'evaluated',
        prediction: record?.prediction ?? null, score: record?.score ?? null }
    }) }
}

export function modelOutputsMiddleware({ dataDir = defaultDataDir, outputDir = modelOutputDir } = {}) {
  return async (req, res, next) => {
    const url = new URL(req.url, 'http://localhost')
    if (url.pathname !== '/api/model-outputs') return next()
    res.setHeader('Content-Type', 'application/json; charset=utf-8')
    res.setHeader('Cache-Control', 'no-store')
    try {
      if (req.method !== 'GET') throw fail(405, 'Method not allowed')
      const index = url.searchParams.get('sampleIndex')
      if (index === null || !/^\d+$/.test(index) || !Number.isSafeInteger(Number(index))) throw fail(400, 'Invalid sample index')
      const data = await readData(url.searchParams.get('file'), dataDir)
      const samples = Array.isArray(data) ? data : [data]
      const sample = samples[Number(index)]
      if (!sample) throw fail(404, 'Sample not found')
      let bundle
      try { bundle = JSON.parse(await readFile(path.join(outputDir, 'results.json'), 'utf8')) }
      catch (error) {
        if (error.code !== 'ENOENT') throw error
        res.end(JSON.stringify({ models: [], message: 'Model outputs are not available yet.' }))
        return
      }
      if (bundle.version !== 1 || !Array.isArray(bundle.samples) || !Array.isArray(bundle.models)) throw fail(500, 'Invalid model output export')
      res.end(JSON.stringify(modelResultsForSample(bundle, sample)))
    } catch (error) {
      res.statusCode = error.status || (error.code === 'ENOENT' ? 404 : 500)
      res.end(JSON.stringify({ error: res.statusCode === 500 ? 'Unable to read model outputs' : error.message }))
    }
  }
}

export function modelOutputsPlugin() {
  const middleware = modelOutputsMiddleware()
  return { name: 'gallery-model-outputs',
    configureServer(server) { server.middlewares.use(middleware) },
    configurePreviewServer(server) { server.middlewares.use(middleware) } }
}
