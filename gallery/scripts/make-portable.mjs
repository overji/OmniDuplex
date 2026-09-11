import { readFile, readdir, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const galleryDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const distDir = path.join(galleryDir, 'dist')
const indexPath = path.join(distDir, 'index.html')

function resolveDistAsset(relativeUrl) {
  const assetPath = path.resolve(distDir, relativeUrl)
  const relativePath = path.relative(distDir, assetPath)
  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    throw new Error(`refusing to inline asset outside dist: ${relativeUrl}`)
  }
  return assetPath
}

let html = await readFile(indexPath, 'utf8')

const scriptPattern = /<script\s+type="module"\s+crossorigin\s+src="([^"]+)"><\/script>/g
const scriptMatches = [...html.matchAll(scriptPattern)]
if (scriptMatches.length !== 1) {
  throw new Error(`expected one generated module script, found ${scriptMatches.length}`)
}
for (const match of scriptMatches) {
  const source = await readFile(resolveDistAsset(match[1]), 'utf8')
  const safeSource = source.replaceAll('</script', '<\\/script')
  const inlineScript = `<script type="module">${safeSource}</script>`
  html = html.replace(
    match[0],
    () => inlineScript,
  )
  if (!html.includes(inlineScript)) throw new Error('failed to preserve module script')
}

const stylesheetPattern = /<link\s+rel="stylesheet"\s+crossorigin\s+href="([^"]+)">/g
const stylesheetMatches = [...html.matchAll(stylesheetPattern)]
if (stylesheetMatches.length !== 1) {
  throw new Error(
    `expected one generated stylesheet, found ${stylesheetMatches.length}`,
  )
}
for (const match of stylesheetMatches) {
  const source = await readFile(resolveDistAsset(match[1]), 'utf8')
  const safeSource = source.replaceAll('</style', '<\\/style')
  const inlineStylesheet = `<style>${safeSource}</style>`
  html = html.replace(match[0], () => inlineStylesheet)
  if (!html.includes(inlineStylesheet)) {
    throw new Error('failed to preserve stylesheet')
  }
}

await writeFile(indexPath, html, 'utf8')

const assetsDir = path.join(distDir, 'assets')
try {
  const generatedAssets = await readdir(assetsDir)
  if (generatedAssets.every((name) => /\.(?:css|js)$/.test(name))) {
    await rm(assetsDir, { recursive: true })
  }
} catch (error) {
  if (error.code !== 'ENOENT') throw error
}
