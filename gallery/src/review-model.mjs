export function parseTime(value) {
  if (typeof value === 'number') return Number.isFinite(value) && value >= 0 ? value : null
  if (typeof value !== 'string' || !/^\d+:\d{2}(?::\d{2})?(?:\.\d+)?$/.test(value.trim())) return null
  const parts = value.trim().split(':').map(Number)
  if (parts.slice(1).some(part => part >= 60)) return null
  return parts.reduce((total, part) => total * 60 + part, 0)
}

export function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

export function sampleKind(sample) {
  const type = sample.sample_type ?? sample.generation_context?.sample_type
  return ['missing_tool', 'no_context'].includes(type) ? 'negative' : 'positive'
}

export function decodeJSON(value) {
  if (typeof value !== 'string') return value
  try { return JSON.parse(value) } catch { return value }
}

export function reviewRecords(sample) {
  const responses = Array.isArray(sample.responses) ? sample.responses : [
    ...(sample.audio_events || []), ...(sample.visual_events || []),
  ]
  const records = responses.map((item, index) => ({ item, index, track: 'responses' }))
  records.push(...(Array.isArray(sample.annotations) ? sample.annotations : []).map((item, index) => ({ item, index, track: 'annotations' })))
  return records.map(({ item, index, track }) => {
    const event = item && typeof item === 'object' ? item : { response: String(item) }
    const seconds = parseTime(event.trigger_time ?? event.start_seconds ?? event.start_time)
    const start = parseTime(event.window_start)
    const end = parseTime(event.window_end)
    const hasWindow = start !== null && end !== null && seconds !== null && start <= seconds && seconds <= end
    return {
      ...event, id: `${track}:${index}`, track, originalIndex: index,
      response: event.response ?? event.event_name,
      trigger_time: event.trigger_time ?? event.start_time ?? formatTime(seconds),
      trigger_type_reason: event.trigger_type_reason ?? event.evidence_for_judgment ?? event.visual_details,
      seconds, start, end, hasWindow,
      label: track === 'annotations' ? 'Observation annotation'
        : sampleKind(sample) === 'negative' ? 'Refusal'
          : event.role === 'tool' ? 'Simulated tool result'
            : event.tool_calls?.length ? 'Agent delegation'
              : event.role === 'assistant' ? 'Agent reply' : 'Response',
    }
  }).sort((a, b) => (a.seconds ?? Infinity) - (b.seconds ?? Infinity))
}

export function isActiveWindow(record, currentTime) {
  return record.hasWindow && record.start <= currentTime && currentTime <= record.end
}

export function windowLanes(records) {
  const ends = new Map()
  return records.filter(record => record.hasWindow).sort((a, b) => a.start - b.start || a.originalIndex - b.originalIndex).map(record => {
    const lanes = ends.get(record.track) || []
    let lane = lanes.findIndex(end => end < record.start)
    if (lane < 0) lane = lanes.length
    lanes[lane] = record.end
    ends.set(record.track, lanes)
    return { ...record, lane }
  })
}

export function matchesFilters(sample, filters) {
  return (!filters.ability || sample.ability === filters.ability)
    && (!filters.mode || sample.mode === filters.mode)
    && (!filters.kind || sampleKind(sample) === filters.kind)
    && (!filters.video || sample.video === filters.video)
}

export function initialFile(files, saved) {
  return files.includes(saved) ? saved
    : files.find(name => name === 'qwen3.8_omni_flash_44/all_qa.json')
      || files.find(name => name.endsWith('/all_qa.json')) || files[0] || ''
}
