import { createHash } from 'node:crypto'

// Preserve the original annotation-gallery fingerprints for legacy scores.
export function fingerprint(sample, response) {
  return createHash('sha256').update(JSON.stringify([
    sample.video, sample.question, sample.task, response,
  ])).digest('hex')
}

export function responseFingerprints(data) {
  return (Array.isArray(data) ? data : [data]).map(sample =>
    (Array.isArray(sample?.responses) ? sample.responses : []).map(response => fingerprint(sample, response)))
}

export function currentAnnotations(stored, fingerprints) {
  return Object.fromEntries(Object.entries(stored.records).filter(([key, record]) => {
    if (!/^\d+:\d+$/.test(key)) return false
    const [sampleIndex, responseIndex] = key.split(':').map(Number)
    return typeof record?.fingerprint === 'string' && fingerprints[sampleIndex]?.[responseIndex] === record.fingerprint
  }))
}
