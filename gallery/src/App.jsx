import { useEffect, useMemo, useRef, useState } from 'react'
import snapshot from 'virtual:gallery-snapshot'
import { flattenSamples, firstIncomplete, isComplete, isScorableResponse } from './annotation-state.mjs'
import { useAnnotations } from './use-annotations.jsx'
import { ResponseRating, AnnotationSummary } from './annotation-controls.jsx'
import { ModelOutputs, useModelOutputs } from './model-outputs.jsx'
import { predictionRecords } from './model-output-state.mjs'

import { parseTime, formatTime, sampleKind, decodeJSON, reviewRecords, isActiveWindow, windowLanes, matchesFilters, initialFile } from './review-model.mjs'
function getVideoUrl(videoName) {
  return videoName ? `${import.meta.env.BASE_URL}${encodeURIComponent(videoName)}` : ''
}

function Icon({ name, size = 18 }) {
  const paths = {
    arrow: <path d="m9 18 6-6-6-6" />,
    eye: <><path d="M2.1 12S5.7 5 12 5s9.9 7 9.9 7-3.6 7-9.9 7-9.9-7-9.9-7Z"/><circle cx="12" cy="12" r="3"/></>,
    spark: <path d="m12 3-1.2 4.1a5.3 5.3 0 0 1-3.7 3.7L3 12l4.1 1.2a5.3 5.3 0 0 1 3.7 3.7L12 21l1.2-4.1a5.3 5.3 0 0 1 3.7-3.7L21 12l-4.1-1.2a5.3 5.3 0 0 1-3.7-3.7L12 3Z" />,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    audio: <><path d="M11 5 6 9H3v6h3l5 4V5Z"/><path d="M15 9.5a4 4 0 0 1 0 5"/></>,
    check: <path d="m5 12 4 4L19 6" />,
    alert: <><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.7 2.5 17.2A2 2 0 0 0 4.2 20h15.6a2 2 0 0 0 1.7-2.8L13.7 3.7a2 2 0 0 0-3.4 0Z"/></>,
  }
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      {paths[name]}
    </svg>
  )
}

function MetaPill({ icon, label, value, accent = false }) {
  return (
    <div className={`meta-pill${accent ? ' meta-pill--accent' : ''}`}>
      <span className="meta-pill__icon"><Icon name={icon} size={15} /></span>
      <span className="meta-pill__label">{label}</span>
      <strong>{value || 'Not specified'}</strong>
    </div>
  )
}

function TimeButton({ label, seconds, onSeek }) {
  return <button className="time-button" type="button" disabled={seconds === null} onClick={() => onSeek(seconds)}>{label} <strong>{formatTime(seconds)}</strong></button>
}

function EventCard({ event, index, isActive, onSeek, cardRef, annotation }) {
  const body = event.role === 'tool' ? decodeJSON(event.response) : event.response
  return <article ref={cardRef} className={`event-card${isActive ? ' event-card--active' : ''}${event.track === 'annotations' ? ' event-card--annotation' : ''}${annotation?.selected ? ' event-card--selected' : ''}`} aria-current={isActive ? 'true' : undefined}>
    <button className="event-card__button" type="button" disabled={event.seconds === null} onClick={() => onSeek(event.seconds)}>
      <span className="event-card__rail"><span className="event-card__dot" /></span>
      <span className="event-card__content">
        <span className="event-card__topline"><span className="timecode">{event.trigger_time}</span><span className="event-number">{event.label} · {index + 1}</span><Icon name="arrow" /></span>
        <strong className="event-card__response">{typeof body === 'object' && body !== null ? body.summary || body.status || 'Tool result' : body || 'No response provided'}</strong>
        {event.trigger_type_reason && <span className="event-card__reason">{event.trigger_type} · {event.trigger_type_reason}</span>}
        {event.event_description && <span className="event-card__description">{event.event_description}</span>}
      </span>
    </button>
    <div className="event-card__details">
      {event.hasWindow ? <div className="window-times"><span>Valid response window</span><TimeButton label="Earliest" seconds={event.start} onSeek={onSeek} /><TimeButton label="Recommended" seconds={event.seconds} onSeek={onSeek} /><TimeButton label="Latest" seconds={event.end} onSeek={onSeek} /></div>
        : <TimeButton label="Message time" seconds={event.seconds} onSeek={onSeek} />}
      {(event.window_start !== undefined || event.window_end !== undefined) && !event.hasWindow && <p role="alert">Invalid or incomplete window — see raw fields below.</p>}
      {event.refusal_reason && <p className="refusal-reason">{event.refusal_reason}</p>}
      {event.missing_tools && <p>Missing tools: {event.missing_tools.join(', ') || 'None'}</p>}
      {event.evidence?.length > 0 && <details><summary>Evidence ({event.evidence.length})</summary>{event.evidence.map((evidence, i) => <div className="evidence-record" key={i}><TimeButton label="Observed" seconds={parseTime(evidence.time)} onSeek={onSeek} /><TimeButton label="Available" seconds={parseTime(evidence.available_time)} onSeek={onSeek} /><p>{evidence.fact}</p></div>)}</details>}
      {event.details && <details><summary>Task details</summary><ValueView value={event.details} /></details>}
      {event.trigger && <details><summary>Trigger criterion</summary><ValueView value={event.trigger} /></details>}
      {event.tool_calls?.map((call, i) => <details key={i}><summary>Tool call · {call.function?.name || call.name}</summary><ValueView value={decodeJSON(call.function?.arguments ?? call.arguments)} /></details>)}
      {event.role === 'tool' && <details><summary>Simulated backend result</summary><ValueView value={body} /></details>}
      {event.outcome && <p>Outcome: <strong>{event.outcome}</strong></p>}
      {annotation && <>
        <button className="time-button" type="button" disabled={annotation.disabled} onClick={annotation.onSelect}>
          {annotation.selected ? 'Current response' : 'Review this response'} · {isComplete(annotation.record) ? 'Complete' : 'Unfinished'}
        </button>
        <ResponseRating record={annotation.record} disabled={annotation.disabled} onRate={annotation.onRate} />
      </>}
    </div>
  </article>
}

function WindowTimeline({ events, currentTime, duration, questionSeconds, onSeek }) {
  const windows = windowLanes(events)
  const position = seconds => `${Math.max(0, Math.min(seconds / duration * 100, 100))}%`
  if (!duration) return <p className="data-null">Timeline unavailable until video duration is known.</p>
  return <>
    <p className="timeline-legend">Band: valid response window · Diamond: recommended time · Dot: message time</p>
    <div className="scrubber">
      <div className="scrubber__track" /><div className="scrubber__fill" style={{ width: position(currentTime) }} />
      {questionSeconds !== null && <div className="question-marker" style={{ left: position(questionSeconds) }} title={`Question time ${formatTime(questionSeconds)}`}><span>QUESTION</span></div>}
      <input aria-label="Video timeline" type="range" min="0" max={duration} step="0.1" value={Math.min(currentTime, duration)} onChange={e => onSeek(Number(e.target.value))} />
    </div>
    {['responses', 'annotations', 'predictions'].map(track => {
      const records = events.filter(event => event.track === track)
      if (!records.length) return null
      const bands = windows.filter(event => event.track === track)
      const laneCount = bands.length ? Math.max(...bands.map(event => event.lane)) + 1 : 0
      const messages = records.filter(event => !event.hasWindow && event.seconds !== null)
      const messageLanes = messages.map((event, index) => messages.slice(0, index).filter(previous => previous.seconds === event.seconds).length)
      const messageLaneCount = messages.length ? Math.max(...messageLanes) + 1 : 0
      return <div className={`window-track window-track--${track}`} key={track}>
        <span>{track === 'annotations' ? 'Observation annotations · not Agent actions' : track === 'predictions' ? 'Model predictions' : 'Responses / messages'}</span>
        <div className="window-track__lanes" style={{ height: `${Math.max(1, laneCount + messageLaneCount) * 26}px` }}>
          {bands.map(event => <div key={event.id} className={`window-band${isActiveWindow(event, currentTime) ? ' is-active' : ''}`} style={{ left: position(event.start), width: `${Math.max(0, Math.min(event.end, duration) - Math.min(event.start, duration)) / duration * 100}%`, top: `${event.lane * 26 + 6}px` }}>
            <button className="window-band__hit" type="button" aria-label={`Window ${event.id}: ${formatTime(event.start)} to ${formatTime(event.end)}`} title={`${event.label}: ${formatTime(event.start)} → ${formatTime(event.end)}`} onClick={() => onSeek(event.start)} />
            <button className="window-trigger" type="button" style={{ left: event.end === event.start ? '0%' : `${(event.seconds - event.start) / (event.end - event.start) * 100}%` }} title={`Recommended: ${event.trigger_time}`} aria-label={`Recommended time ${event.trigger_time} for ${event.id}`} onClick={() => onSeek(event.seconds)} />
          </div>)}
          {messages.map((event, index) => <button key={event.id} className="message-marker" type="button" style={{ left: position(event.seconds), top: `${(laneCount + messageLanes[index]) * 26 + 6}px` }} title={`${event.label} · ${event.trigger_time}`} aria-label={`${event.label} at ${event.trigger_time}`} onClick={() => onSeek(event.seconds)} />)}
        </div>
      </div>
    })}
    <div className="timeline-scale"><span>00:00</span><span>{formatTime(duration / 4)}</span><span>{formatTime(duration / 2)}</span><span>{formatTime(duration * .75)}</span><span>{formatTime(duration)}</span></div>
  </>
}

function SampleReview({ sampleData, annotation, modelOutputs }) {
  const videoName = typeof sampleData.video === 'string' && sampleData.video.trim() ? sampleData.video.trim() : ''
  const videoUrl = getVideoUrl(videoName)
  const videoRef = useRef(null)
  const cardRefs = useRef(new Map())
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(Number.isFinite(sampleData.generation_context?.duration_seconds) && sampleData.generation_context.duration_seconds > 0 ? sampleData.generation_context.duration_seconds : null)
  const [videoError, setVideoError] = useState(!videoName)
  const [isPlaying, setIsPlaying] = useState(false)

  const events = useMemo(() => reviewRecords(sampleData), [sampleData])
  const outputModel = modelOutputs?.state.models.find(model => model.id === modelOutputs.modelId) || modelOutputs?.state.models[0]
  const timelineEvents = modelOutputs ? [...events, ...predictionRecords(outputModel)] : events
  const activeIds = events.filter(event => isActiveWindow(event, currentTime)).map(event => event.id).join('|')
  const activeIndex = events.findIndex(event => isActiveWindow(event, currentTime))

  useEffect(() => {
    if (annotation || activeIndex < 0) return
    const card = cardRefs.current.get(activeIndex)
    if (card) card.parentElement.scrollTo({ top: card.offsetTop, behavior: 'smooth' })
  }, [activeIndex, activeIds])

  useEffect(() => {
    if (!annotation?.activeKey) return
    const index = events.findIndex(event => event.track === 'responses' && `${annotation.sampleIndex}:${event.originalIndex}` === annotation.activeKey)
    const card = cardRefs.current.get(index)
    if (card) card.parentElement.scrollTo({ top: card.offsetTop, behavior: 'smooth' })
    if (index >= 0 && videoRef.current?.readyState > 0) seekTo(events[index].seconds)
  }, [annotation?.activeKey])

  function seekTo(seconds) {
    if (seconds === null || !videoRef.current || videoError) return
    const safeTime = duration ? Math.min(Math.max(seconds, 0), duration) : Math.max(seconds, 0)
    videoRef.current.currentTime = safeTime
    setCurrentTime(safeTime)
  }

  const questionSeconds = parseTime(sampleData.question_time)

  return (
    <div className="app-shell">
      <div className="ambient ambient--one" />
      <div className="ambient ambient--two" />

      <header className="site-header">
        <a className="brand" href="#top" aria-label="OmniDuplex event review home">
          <span className="brand__mark"><span /></span>
          <span>OMNI<span>DUPLEX</span></span>
        </a>
        <div className="header-context">
          <span className="header-context__eyebrow">GALLERY / SELECTED SAMPLE</span>
          <span className="header-context__title">Multimodal Sample Review</span>
        </div>
        <div className={`status-badge status-badge--${sampleData.status === 'ok' ? 'ok' : 'warning'}`}>
          <span className="status-badge__dot" />
          {sampleData.status === 'ok' ? 'READY TO REVIEW' : 'CHECK SAMPLE'}
        </div>
      </header>

      <section className="intro" id="top">
        <div className="intro__copy">
          <p className="kicker"><span>01</span> REVIEW PROMPT</p>
          <h1>Browse this sample.<br /><em>Explore every detail.</em></h1>
          <div className="question-card">
            <span className="question-card__icon"><Icon name="spark" /></span>
            <div>
              <span className="question-card__label">USER REQUEST · {sampleData.question_time || '00:00'}</span>
              <p>“{sampleData.question || 'No review question provided.'}”</p>
            </div>
          </div>
        </div>
        <div className="intro__meta">
          <MetaPill icon="eye" label="Vision" value={sampleData.vision_dependency} accent />
          <MetaPill icon="audio" label="Audio" value={sampleData.audio_dependency} />
          <MetaPill icon="check" label="Task" value={sampleData.ability || sampleData.task} />
          <MetaPill icon="check" label="Mode" value={sampleData.mode} />
          <MetaPill icon="check" label="Sample" value={`${sampleKind(sampleData)}${sampleKind(sampleData) === 'negative' ? ` · ${sampleData.sample_type}` : ''}`} />
        </div>
      </section>

      <section className="review-grid">
        <div className="player-column">
          <div className="section-heading">
            <div><span className="section-heading__number">01</span><h2>{videoName || 'Video unavailable'}</h2></div>
            <span className={`play-state${isPlaying ? ' play-state--live' : ''}`}><i />{isPlaying ? 'PLAYING' : 'PAUSED'}</span>
          </div>

          <div className="video-frame">
            {videoUrl && <video
              ref={videoRef}
              src={videoUrl}
              controls
              preload="metadata"
              playsInline
              onLoadedMetadata={(event) => {
                const value = event.currentTarget.duration
                setDuration(Number.isFinite(value) && value > 0 ? value : duration)
                setVideoError(false)
                if (annotation?.activeKey) {
                  const event = events.find(item => item.track === 'responses' && `${annotation.sampleIndex}:${item.originalIndex}` === annotation.activeKey)
                  if (event) seekTo(event.seconds)
                }
              }}
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
              onError={() => setVideoError(true)}
            >
              Your browser does not support HTML video.
            </video>}
            <div className="video-frame__corner video-frame__corner--tl" />
            <div className="video-frame__corner video-frame__corner--br" />
            {videoError && (
              <div className="video-error" role="alert">
                <Icon name="alert" size={26} />
                <strong>Video unavailable</strong>
                <span>{videoName ? `Check that ${videoName} is available beside this gallery.` : 'This sample does not specify a video file.'}</span>
              </div>
            )}
          </div>

          <div className="timeline-panel">
            <div className="timeline-panel__readout">
              <span><strong>{formatTime(currentTime)}</strong> / {formatTime(duration)}</span>
              <span>{events.length} RECORDS</span>
            </div>
            <WindowTimeline events={timelineEvents} currentTime={currentTime} duration={duration} questionSeconds={questionSeconds} onSeek={seekTo} />
          </div>
        </div>

        <aside className="events-column" aria-label="Detected event timeline">
          <div className="section-heading section-heading--events">
            <div><span className="section-heading__number">02</span><h2>Responses & Annotations</h2></div>
            <span className="event-count">{String(events.length).padStart(2, '0')}</span>
          </div>
          {modelOutputs && <ModelOutputs state={modelOutputs.state} model={outputModel} modelId={modelOutputs.modelId} onModelChange={modelOutputs.onModelChange} onSeek={seekTo} />}
          <div className="events-list">
            {events.length ? events.map((event, index) => (
              <EventCard
                key={event.id}
                event={event}
                index={index}
                isActive={isActiveWindow(event, currentTime)}
                onSeek={seekTo}
                annotation={annotation && event.track === 'responses' && isScorableResponse(sampleData, event) ? {
                  selected: annotation.activeKey === `${annotation.sampleIndex}:${event.originalIndex}`,
                  record: annotation.records[`${annotation.sampleIndex}:${event.originalIndex}`],
                  disabled: annotation.disabled,
                  onSelect: () => annotation.onSelect(event.originalIndex),
                  onRate: (dimension, score) => annotation.onRate(event.originalIndex, dimension, score),
                } : null}
                cardRef={(node) => node ? cardRefs.current.set(index, node) : cardRefs.current.delete(index)}
              />
            )) : (
              <div className="empty-state"><Icon name="alert" /><strong>No events found</strong><span>This sample contains no responses or annotations.</span></div>
            )}
          </div>
          <div className="events-footer"><span>SELECT AN EVENT TO SEEK</span><span>SYNCED WITH VIDEO</span></div>
        </aside>
      </section>

      <DataSections data={sampleData} onSeek={seekTo} />
      <footer className="site-footer">
        <span>OMNIDUPLEX · MULTIMODAL EVENT DATASET</span>
        <span>VISION-DEPENDENT SAMPLE <i /> 2026</span>
      </footer>
    </div>
  )
}


const sectionLabels = {
  video_summary: 'Video summary', responses: 'Responses', audio_events: 'Audio events',
  visual_events: 'Visual events', items: 'Items', uncertainties: 'Uncertainties', annotations: 'Observation annotations',
}

function ValueView({ value }) {
  if (value === null) return <span className="data-null">null</span>
  if (Array.isArray(value)) return value.length
    ? <ol className="data-array">{value.map((item, index) => <li key={index}><ValueView value={item} /></li>)}</ol>
    : <span>[]</span>
  if (typeof value === 'object') return <dl className="data-fields">{Object.entries(value).map(([key, item]) => (
    <div key={key}><dt>{key}</dt><dd><ValueView value={item} /></dd></div>
  ))}</dl>
  return <span>{String(value)}</span>
}

function DataSections({ data, onSeek }) {
  return <section className="data-sections" aria-label="All sample data">
    <h2>All sample data</h2>
    {data && typeof data === 'object' && !Array.isArray(data)
      ? Object.entries(data).map(([key, value]) => <details className="data-section" key={key}>
        <summary>{sectionLabels[key] || key}{Array.isArray(value) ? ` (${value.length})` : ''}</summary>
        {Array.isArray(value) ? (value.length ? value.map((item, index) => {
          const seconds = parseTime(item?.start_seconds ?? item?.trigger_time ?? item?.start_time ?? item?.first_appearance_time)
          return <article className="data-record" key={index}>
            <div className="data-record__heading"><strong>{item?.event_name || item?.name || item?.event_id || `#${index + 1}`}</strong>
              {seconds !== null && <button type="button" onClick={() => onSeek(seconds)}>Jump to {formatTime(seconds)}</button>}
            </div><ValueView value={item} />
          </article>
        }) : <p className="data-null">No entries</p>) : <ValueView value={value} />}
      </details>) : <ValueView value={data} />}
    <details className="raw-data"><summary>Raw JSON</summary><pre>{JSON.stringify(data, null, 2)}</pre></details>
  </section>
}

function readSelection() {
  try { return JSON.parse(localStorage.getItem('gallery-selection')) || {} } catch { return {} }
}
function saveSelection(file, point) {
  try { localStorage.setItem('gallery-selection', JSON.stringify({ file, point })) } catch { /* Storage may be unavailable. */ }
}
async function getJSON(url, signal) {
  const response = await fetch(url, { signal, cache: 'no-store' })
  const body = await response.json()
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`)
  return body
}

export default function App() {
  const [saved] = useState(readSelection)
  const [files, setFiles] = useState([])
  const [file, setFile] = useState('')
  const [points, setPoints] = useState([])
  const [fingerprints, setFingerprints] = useState([])
  const [mode, setMode] = useState('browse')
  const [modelId, setModelId] = useState('minicpmo')
  const [annotator, setAnnotator] = useState('A')
  const [activeKey, setActiveKey] = useState(null)
  const [point, setPoint] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filters, setFilters] = useState({ ability: '', mode: '', kind: '', video: '' })
  const [offline, setOffline] = useState(location.protocol === 'file:')
  const annotations = useAnnotations(file, mode === 'annotate' && !offline && !loading && !error)

  useEffect(() => {
    const controller = new AbortController()
    async function initialize() {
      try {
        let names
        if (location.protocol === 'file:') names = Object.keys(snapshot).sort()
        else {
          // A static export has no API. Only fall back when there is clearly no JSON API.
          const response = await fetch('./api/files', { signal: controller.signal, cache: 'no-store' })
          if (response.status === 404 || (response.ok && !response.headers.get('content-type')?.includes('application/json'))) {
            names = Object.keys(snapshot).sort()
            setOffline(true)
          } else {
            const body = await response.json()
            if (!response.ok) throw new Error(body.error || 'Unable to load files')
            names = body.files
          }
        }
        if (controller.signal.aborted) return
        setFiles(names)
        setFile(initialFile(names, saved.file))
        if (!names.length) setLoading(false)
      } catch (err) {
        if (!controller.signal.aborted) { setError(err.message); setLoading(false) }
      }
    }
    initialize()
    return () => controller.abort()
  }, [saved])

  useEffect(() => {
    if (!file) return
    const controller = new AbortController()
    setLoading(true)
    setError('')
    setPoints([])
    setFingerprints([])
    async function load() {
      try {
        const result = offline ? snapshot[file] : await getJSON(`./api/data?file=${encodeURIComponent(file)}`, controller.signal)
        if (result.error) throw new Error(result.error)
        const entries = Array.isArray(result.data) ? result.data : [result.data]
        if (controller.signal.aborted) return
        const remembered = readSelection()
        const selected = remembered.file === file && Number.isInteger(remembered.point) && remembered.point >= 0 && remembered.point < entries.length ? remembered.point : 0
        setPoints(entries)
        setFingerprints(result.fingerprints || [])
        setPoint(selected)
        saveSelection(file, selected)
      } catch (err) { if (!controller.signal.aborted) setError(err.message) }
      finally { if (!controller.signal.aborted) setLoading(false) }
    }
    load()
    return () => controller.abort()
  }, [file, offline])

  function selectFile(name) {
    if (name === file) return
    setLoading(true)
    setPoints([])
    setError('')
    setFile(name)
  }

  const visiblePoints = points.map((entry, index) => ({ entry, index })).filter(({ entry }) => matchesFilters(entry, filters))
  const items = useMemo(() => flattenSamples(points, fingerprints), [points, fingerprints])
  const visibleItems = useMemo(() => items.filter(item => matchesFilters(item.sample, filters)), [items, filters])
  const annotationPoint = activeKey === null ? undefined : Number(activeKey.split(':')[0])
  const targetPoint = mode === 'annotate' ? annotationPoint : point
  const selectedPoint = visiblePoints.some(item => item.index === targetPoint) ? targetPoint : visiblePoints[0]?.index
  const complete = annotations.ready && firstIncomplete(visibleItems, annotations.groups[annotator]) < 0

  useEffect(() => {
    if (mode !== 'annotate' || !annotations.ready) { setActiveKey(null); return }
    const index = firstIncomplete(visibleItems, annotations.groups[annotator])
    setActiveKey(index < 0 ? null : visibleItems[index].key)
  }, [mode, file, annotator, filters, annotations.ready])

  function selectSample(index) {
    if (mode !== 'annotate') { setPoint(index); saveSelection(file, index); return }
    const candidates = visibleItems.filter(item => item.sampleIndex === index)
    const next = firstIncomplete(candidates, annotations.groups[annotator])
    setActiveKey(candidates[next < 0 ? 0 : next]?.key || null)
  }

  async function rateResponse(responseIndex, dimension, score) {
    const item = items.find(entry => entry.sampleIndex === selectedPoint && entry.responseIndex === responseIndex)
    if (!item) return
    setActiveKey(item.key)
    const updated = await annotations.save(annotator, item, dimension, score)
    if (updated && isComplete(updated[item.key])) {
      const next = firstIncomplete(visibleItems, updated)
      setActiveKey(next < 0 ? null : visibleItems[next].key)
    }
  }

  async function retrySave() {
    const updated = await annotations.retry?.()
    if (updated && isComplete(updated[activeKey])) {
      const next = firstIncomplete(visibleItems, updated)
      setActiveKey(next < 0 ? null : visibleItems[next].key)
    }
  }
  const modelResults = useModelOutputs(file, selectedPoint, mode === 'models' && !offline && !loading && !error)
  const selected = points[selectedPoint]
  const reviewData = selected && typeof selected === 'object' && !Array.isArray(selected) ? selected : null
  return <div className="gallery-layout">
    <aside className="file-sidebar" aria-label="Data files">
      <a className="brand" href="#">OMNI<span>DUPLEX</span></a>
      <h1>Data files <span>{files.length}</span></h1>
      <p className="source-label">data_generation/data</p>
      <p className="source-label">{offline ? 'Offline snapshot · rebuild to update' : 'Live directory · refresh to update'}</p>
      <nav>{files.map(name => <button type="button" key={name} disabled={annotations.saving} className={`file-button${file === name ? ' is-selected' : ''}`} aria-current={file === name ? 'true' : undefined} onClick={() => selectFile(name)}>{name}</button>)}</nav>
    </aside>
    <main className="gallery-main">
      <header className="point-picker">
        <h2>{file || 'Gallery'} <span>{points.length} samples</span></h2>
        <div className="annotation-toolbar">
          <div role="group" aria-label="Gallery mode">{['browse', 'annotate', 'models'].map(value => <button type="button" key={value}
            aria-pressed={mode === value} disabled={annotations.saving || (value !== 'browse' && offline)}
            onClick={() => setMode(value)}>{value === 'browse' ? 'Browse' : value === 'annotate' ? 'Annotate' : 'Model outputs'}</button>)}</div>
          {mode === 'annotate' && <label>Annotator <select aria-label="Annotator" disabled={annotations.saving}
            value={annotator} onChange={event => setAnnotator(event.target.value)}><option>A</option><option>B</option></select></label>}
          <span role="status">{offline ? 'Annotation and model outputs require the live gallery server' : mode === 'annotate' ? annotations.status || (annotations.ready ? 'Synced' : 'Loading scores…') : ''}</span>
        </div>
        <div className="sample-filters">
          {['ability', 'mode', 'kind', 'video'].map(field => <label key={field}>{field === 'kind' ? 'Sample type' : field}<select aria-label={`Filter ${field}`} disabled={annotations.saving} value={filters[field]} onChange={e => setFilters({ ...filters, [field]: e.target.value })}><option value="">All</option>{[...new Set(points.map(entry => field === 'kind' ? sampleKind(entry) : entry?.[field]).filter(Boolean))].sort().map(value => <option key={value} value={value}>{value}</option>)}</select></label>)}
          <span>{visiblePoints.length} / {points.length} samples</span>
        </div>
        <div className="point-buttons" aria-label="Data points">{visiblePoints.map(({ entry, index }) => {
          const title = entry?.question || entry?.name || entry?.id || entry?.video_summary?.main_content_summary
          return <button type="button" key={index} disabled={annotations.saving} aria-pressed={selectedPoint === index} title={title ? String(title) : `Sample ${index + 1}`} onClick={() => selectSample(index)}>
            <strong>{String(index + 1).padStart(2, '0')}</strong><span>{entry?.ability || entry?.task} · {entry?.mode} · {sampleKind(entry)}{sampleKind(entry) === 'negative' ? ` (${entry.sample_type})` : ''}<br />{title ? String(title) : `Sample ${index + 1}`}</span>
          </button>
        })}</div>
      </header>
      {mode === 'annotate' && !loading && !error && <>
        {annotations.error && <p className="gallery-message" role="alert">{annotations.error}. {annotations.retry
          ? <button className="time-button" type="button" disabled={annotations.saving} onClick={retrySave}>Retry save</button>
          : 'Scores will retry loading automatically. Reload if source data changed.'}</p>}
        {annotations.ready && <AnnotationSummary items={visibleItems} groups={annotations.groups} annotator={annotator} />}
        {complete && <p className="gallery-message" role="status">All responses in this file and filter selection are complete for annotator {annotator}. Select any response to edit its scores.</p>}
      </>}
      {loading ? <p className="gallery-message" role="status">Loading data…</p>
        : error ? <p className="gallery-message" role="alert">{error}</p>
        : !files.length ? <p className="gallery-message">No JSON files found in data_generation/data.</p>
        : !points.length ? <p className="gallery-message">This file contains no samples.</p>
        : !visiblePoints.length ? <p className="gallery-message">No samples match these filters.</p>
        : reviewData ? <SampleReview key={`${file}:${selectedPoint}`} sampleData={reviewData} modelOutputs={mode === 'models' ? { state: modelResults, modelId, onModelChange: setModelId } : null} annotation={mode === 'annotate' ? {
            sampleIndex: selectedPoint, activeKey, records: annotations.groups[annotator],
            disabled: annotations.saving || !annotations.ready,
            onSelect: responseIndex => setActiveKey(`${selectedPoint}:${responseIndex}`), onRate: rateResponse,
          } : null} />
        : <DataSections data={selected} onSeek={() => {}} />}
    </main>
  </div>
}
