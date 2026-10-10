import { copyFile, mkdir, readFile, readdir } from 'node:fs/promises'
import { constants } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { annotationFile } from './annotation-api.mjs'

// Keep an exact backup and copy scores to A. Never replace a conflicting annotation.
export async function migrateAnnotations(legacyDir, outputDir, backupDir) {
  let entries
  try { entries = await readdir(legacyDir, { withFileTypes: true }) }
  catch (error) { if (error.code === 'ENOENT') return { migrated: [], conflicts: [] }; throw error }
  const result = { migrated: [], conflicts: [] }
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.annotations.json')) continue
    const source = path.join(legacyDir, entry.name)
    const name = entry.name.slice(0, -'.annotations.json'.length)
    const bytes = await readFile(source)
    const stored = JSON.parse(bytes)
    if (!stored?.records || typeof stored.records !== 'object' || Array.isArray(stored.records)) throw new Error(`Invalid legacy annotations: ${entry.name}`)
    await mkdir(backupDir, { recursive: true })
    await copyFile(source, path.join(backupDir, entry.name), constants.COPYFILE_EXCL)
    const destination = await annotationFile(outputDir, 'A', name, true)
    try {
      await copyFile(source, destination, constants.COPYFILE_EXCL)
      result.migrated.push(name)
    } catch (error) {
      if (error.code !== 'EEXIST') throw error
      if (!(await readFile(destination)).equals(bytes)) result.conflicts.push(name)
    }
  }
  return result
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const gallery = path.resolve(import.meta.dirname, '..')
  const legacy = process.argv[2] || path.join(gallery, '../annotation_gallery/annotations')
  const output = process.env.ANNOTATION_OUTPUT_DIR || path.join(gallery, 'annotations')
  const backup = path.join(gallery, 'annotation-backups', new Date().toISOString().replaceAll(':', '-'))
  const result = await migrateAnnotations(legacy, output, backup)
  console.log(JSON.stringify(result))
  if (result.conflicts.length) process.exitCode = 1
}
