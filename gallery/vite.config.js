import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { galleryDataPlugin } from './scripts/data-source.mjs'

export default defineConfig({
  // Keep every generated asset URL relative to index.html so the complete dist
  // directory can be copied to any static-server directory or URL subpath.
  base: './',
  // The data plugin serves and copies only the videos referenced by samples.
  publicDir: false,
  plugins: [react(), galleryDataPlugin()],
  server: {
    fs: {
      allow: [path.resolve(import.meta.dirname, '..')],
    },
  },
})
