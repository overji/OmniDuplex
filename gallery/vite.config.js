import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { galleryDataPlugin } from './scripts/data-source.mjs'

export default defineConfig({
  // Keep every generated asset URL relative to index.html so the complete dist
  // directory can be copied to any static-server directory or URL subpath.
  base: './',
  // Treat the source video directory as public input. Vite copies its contents
  // into dist, so the production gallery has no dependency on the source tree.
  publicDir: path.resolve(import.meta.dirname, '../data_generation/videos'),
  plugins: [react(), galleryDataPlugin()],
  server: {
    fs: {
      allow: [path.resolve(import.meta.dirname, '..')],
    },
  },
})
