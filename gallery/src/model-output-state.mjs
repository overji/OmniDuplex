export function formatOnset(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  const millis = Math.round(seconds * 1000)
  const minutes = Math.floor(millis / 60000)
  const remainder = millis % 60000
  return `${String(minutes).padStart(2, '0')}:${String(Math.floor(remainder / 1000)).padStart(2, '0')}.${String(remainder % 1000).padStart(3, '0')}`
}

export function predictionRecords(model) {
  const matches = new Map((model?.score?.matches || []).map(match => [match.event_index, match]))
  return (model?.prediction?.events || []).map((event, index) => ({
    ...event, id: `prediction:${index}`, track: 'predictions', originalIndex: index,
    seconds: Number.isFinite(event.onset_seconds) && event.onset_seconds >= 0 ? event.onset_seconds : null,
    trigger_time: formatOnset(event.onset_seconds), hasWindow: false,
    label: matches.has(index) ? `Matched reference ${matches.get(index).response_index + 1}` : 'Unmatched prediction',
    match: matches.get(index) || null,
  }))
}
