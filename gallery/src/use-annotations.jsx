import { useEffect, useRef, useState } from 'react'

async function request(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options })
  const body = await response.json()
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`)
  return body
}

export function useAnnotations(file, enabled) {
  const [groups, setGroups] = useState({ A: {}, B: {} })
  const [readyFile, setReadyFile] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [status, setStatus] = useState('')
  const busy = useRef(false)
  const revision = useRef(0)
  const failed = useRef(null)

  useEffect(() => {
    setReadyFile('')
    setGroups({ A: {}, B: {} })
    setError('')
    setStatus('')
    failed.current = null
    if (!enabled || !file) return
    const controller = new AbortController()
    async function refresh() {
      if (busy.current) return
      const ticket = ++revision.current
      try {
        const results = await Promise.all(['A', 'B'].map(annotator => request(
          `./api/annotations?file=${encodeURIComponent(file)}&annotator=${annotator}`, { signal: controller.signal })))
        if (controller.signal.aborted || ticket !== revision.current) return
        setGroups({ A: results[0].records, B: results[1].records })
        setReadyFile(file)
        if (!failed.current) setError('')
      } catch (err) {
        if (!controller.signal.aborted && ticket === revision.current) setError(err.message)
      }
    }
    refresh()
    const timer = setInterval(refresh, 5000)
    return () => { controller.abort(); clearInterval(timer); revision.current++ }
  }, [file, enabled])

  async function save(annotator, item, dimension, score) {
    if (busy.current || readyFile !== file || !enabled) return null
    busy.current = true
    revision.current++
    setSaving(true)
    setStatus('Saving…')
    setError('')
    failed.current = { annotator, item, dimension, score }
    try {
      const saved = await request('./api/annotations', {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file, annotator, sampleIndex: item.sampleIndex,
          responseIndex: item.responseIndex, fingerprint: item.fingerprint, dimension, score }),
      })
      const updated = { ...groups[annotator], [saved.key]: saved.record }
      setGroups(previous => ({ ...previous, [annotator]: { ...previous[annotator], [saved.key]: saved.record } }))
      setStatus('Saved')
      failed.current = null
      return updated
    } catch (err) {
      setError(err.message)
      setStatus('Save failed')
      return null
    } finally { busy.current = false; setSaving(false) }
  }

  return { groups, ready: enabled && readyFile === file, error, saving, status, save,
    retry: failed.current ? () => {
      const { annotator, item, dimension, score } = failed.current
      return save(annotator, item, dimension, score)
    } : null }
}
