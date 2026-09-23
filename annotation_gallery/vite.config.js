import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { annotationMiddleware } from './scripts/annotation-api.mjs'

const root = path.resolve(import.meta.dirname, '..')
const dataDir = path.resolve(process.env.ANNOTATION_DATA_DIR || path.join(root, 'data_generation/data'))
const videoDir = path.resolve(process.env.ANNOTATION_VIDEO_DIR || path.join(root, 'data_generation/videos'))
const outputDir = path.resolve(process.env.ANNOTATION_OUTPUT_DIR || path.join(import.meta.dirname, 'annotations'))

const middleware = annotationMiddleware({ dataDir, videoDir, outputDir })
const annotationPlugin = {
  name: 'annotation-data',
  configureServer(server) { server.middlewares.use(middleware) },
  configurePreviewServer(server) { server.middlewares.use(middleware) },
}

export default defineConfig({
  base: './',
  plugins: [react(), annotationPlugin],
})
