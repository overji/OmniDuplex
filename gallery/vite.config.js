import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { galleryDataPlugin, videoDirs } from './scripts/data-source.mjs'
import { annotationMiddleware } from './scripts/annotation-api.mjs'
import { modelOutputsPlugin } from './scripts/model-outputs.mjs'

const middleware = annotationMiddleware()
const annotations = {
  name: 'gallery-annotations',
  configureServer(server) { server.middlewares.use(middleware) },
  configurePreviewServer(server) { server.middlewares.use(middleware) },
}

export default defineConfig({
  // Keep every generated asset URL relative to index.html so the complete dist
  // directory can be copied to any static-server directory or URL subpath.
  base: './',
  // The data plugin serves and copies only the videos referenced by samples.
  publicDir: false,
  plugins: [react(), annotations, modelOutputsPlugin(), galleryDataPlugin()],
  server: {
    fs: {
      allow: [path.resolve(import.meta.dirname, '..'), ...videoDirs],
    },
  },
})
