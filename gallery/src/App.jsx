import { useEffect, useMemo, useRef, useState } from 'react'
import snapshot from 'virtual:gallery-snapshot'

const FALLBACK_DURATION = 273
const videoUrl = `${import.meta.env.BASE_URL}panorama.mp4`

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

function parseTime(value) {
  if (typeof value === 'number') return Number.isFinite(value) && value >= 0 ? value : null
  if (typeof value !== 'string' || !value.trim()) return null
  const parts = value.split(':').map(Number)
  if (![2, 3].includes(parts.length) || parts.some(part => !Number.isFinite(part) || part < 0)) return null
  const seconds = parts.reduce((total, part) => total * 60 + part, 0)
  return seconds >= 0 ? seconds : null
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '00:00'
  const minutes = Math.floor(seconds / 60)
  const remaining = Math.floor(seconds % 60)
  return `${String(minutes).padStart(2, '0')}:${String(remaining).padStart(2, '0')}`
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

function EventCard({ event, index, isActive, onSelect, cardRef }) {
  return (
    <article
      ref={cardRef}
      className={`event-card${isActive ? ' event-card--active' : ''}`}
      aria-current={isActive ? 'true' : undefined}
    >
      <button className="event-card__button" type="button" onClick={onSelect} disabled={!onSelect}>
        <span className="event-card__rail" aria-hidden="true">
          <span className="event-card__dot" />
        </span>
        <span className="event-card__content">
          <span className="event-card__topline">
            <span className="timecode"><Icon name="clock" size={14} />{event.trigger_time || '—'}</span>
            <span className="event-number">EVENT {String(index + 1).padStart(2, '0')}</span>
            <span className="event-card__arrow"><Icon name="arrow" size={17} /></span>
          </span>
          <strong className="event-card__response">{event.response || 'No response provided'}</strong>
          <span className="event-card__description">{event.event_description || 'No event description available.'}</span>
          <span className="event-card__reason">
            <Icon name="eye" size={14} />
            <span>{event.trigger_type_reason || 'No trigger reason provided.'}</span>
          </span>
        </span>
      </button>
    </article>
  )
}

function SampleReview({ sampleData }) {
  const videoRef = useRef(null)
  const cardRefs = useRef(new Map())
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(FALLBACK_DURATION)
  const [videoError, setVideoError] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)

  const events = useMemo(() => {
    const source = Array.isArray(sampleData.responses) ? sampleData.responses : [
      ...(Array.isArray(sampleData.audio_events) ? sampleData.audio_events : []),
      ...(Array.isArray(sampleData.visual_events) ? sampleData.visual_events : []),
    ]
    return source
      .map((value, originalIndex) => {
        const event = value && typeof value === 'object' ? value : { response: String(value) }
        return {
          ...event,
          trigger_time: event.trigger_time ?? event.start_time,
          response: event.response ?? event.event_name,
          event_description: event.event_description ?? event['detailed description'],
          trigger_type_reason: event.trigger_type_reason ?? event.evidence_for_judgment ?? event.visual_details,
          seconds: parseTime(event.start_seconds ?? event.trigger_time ?? event.start_time),
          originalIndex,
        }
      })
      .sort((a, b) => (a.seconds ?? Infinity) - (b.seconds ?? Infinity))
  }, [sampleData])

  const activeIndex = useMemo(() => {
    let found = -1
    events.forEach((event, index) => {
      if (event.seconds !== null && event.seconds <= currentTime) found = index
    })
    return found
  }, [currentTime, events])

  useEffect(() => {
    if (activeIndex < 0) return
    const card = cardRefs.current.get(activeIndex)
    if (card) card.parentElement.scrollTo({ top: card.offsetTop, behavior: 'smooth' })
  }, [activeIndex])

  function seekTo(seconds) {
    if (seconds === null || !videoRef.current || videoError) return
    const safeTime = Math.min(Math.max(seconds, 0), duration)
    videoRef.current.currentTime = safeTime
    setCurrentTime(safeTime)
  }

  function handleScrub(event) {
    seekTo(Number(event.target.value))
  }

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0

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
          <span className="header-context__title">Proactive Event Review</span>
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
          <MetaPill icon="check" label="Detected" value={`${events.length} events`} />
        </div>
      </section>

      <section className="review-grid">
        <div className="player-column">
          <div className="section-heading">
            <div><span className="section-heading__number">01</span><h2>panorama.mp4</h2></div>
            <span className={`play-state${isPlaying ? ' play-state--live' : ''}`}><i />{isPlaying ? 'PLAYING' : 'PAUSED'}</span>
          </div>

          <div className="video-frame">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              preload="metadata"
              playsInline
              onLoadedMetadata={(event) => {
                const value = event.currentTarget.duration
                setDuration(Number.isFinite(value) && value > 0 ? value : FALLBACK_DURATION)
                setVideoError(false)
              }}
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
              onError={() => setVideoError(true)}
            >
              Your browser does not support HTML video.
            </video>
            <div className="video-frame__corner video-frame__corner--tl" />
            <div className="video-frame__corner video-frame__corner--br" />
            {videoError && (
              <div className="video-error" role="alert">
                <Icon name="alert" size={26} />
                <strong>Video unavailable</strong>
                <span>Check that the packaged video asset is present beside this gallery.</span>
              </div>
            )}
          </div>

          <div className="timeline-panel">
            <div className="timeline-panel__readout">
              <span><strong>{formatTime(currentTime)}</strong> / {formatTime(duration)}</span>
              <span>{events.length} EVENTS</span>
            </div>
            <div className="scrubber">
              <div className="scrubber__track" />
              <div className="scrubber__fill" style={{ width: `${progress}%` }} />
              {events.map((event, index) => event.seconds !== null && (
                <button
                  key={`${event.trigger_time}-${index}`}
                  type="button"
                  className={`timeline-marker${index === activeIndex ? ' timeline-marker--active' : ''}`}
                  style={{ left: `${Math.min((event.seconds / duration) * 100, 100)}%` }}
                  onClick={() => seekTo(event.seconds)}
                  aria-label={`Jump to event ${index + 1} at ${event.trigger_time}`}
                  title={`${event.trigger_time} — ${event.response || 'Event'}`}
                />
              ))}
              <input
                aria-label="Video timeline"
                type="range"
                min="0"
                max={duration}
                step="0.1"
                value={Math.min(currentTime, duration)}
                onChange={handleScrub}
              />
            </div>
            <div className="timeline-scale" aria-hidden="true">
              <span>00:00</span><span>{formatTime(duration / 4)}</span><span>{formatTime(duration / 2)}</span><span>{formatTime(duration * 0.75)}</span><span>{formatTime(duration)}</span>
            </div>
          </div>
        </div>

        <aside className="events-column" aria-label="Detected event timeline">
          <div className="section-heading section-heading--events">
            <div><span className="section-heading__number">02</span><h2>Event Timeline</h2></div>
            <span className="event-count">{String(events.length).padStart(2, '0')}</span>
          </div>
          <div className="events-list">
            {events.length ? events.map((event, index) => (
              <EventCard
                key={`${event.trigger_time}-${event.originalIndex}`}
                event={event}
                index={index}
                isActive={index === activeIndex}
                onSelect={event.seconds === null ? undefined : () => seekTo(event.seconds)}
                cardRef={(node) => node ? cardRefs.current.set(index, node) : cardRefs.current.delete(index)}
              />
            )) : (
              <div className="empty-state"><Icon name="alert" /><strong>No events found</strong><span>This sample does not contain responses, audio events, or visual events.</span></div>
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
  visual_events: 'Visual events', items: 'Items', uncertainties: 'Uncertainties',
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
      ? Object.entries(data).map(([key, value]) => <section className="data-section" key={key}>
        <h3>{sectionLabels[key] || key}{Array.isArray(value) ? ` (${value.length})` : ''}</h3>
        {Array.isArray(value) ? (value.length ? value.map((item, index) => {
          const seconds = parseTime(item?.start_seconds ?? item?.trigger_time ?? item?.start_time ?? item?.first_appearance_time)
          return <article className="data-record" key={index}>
            <div className="data-record__heading"><strong>{item?.event_name || item?.name || item?.event_id || `#${index + 1}`}</strong>
              {seconds !== null && <button type="button" onClick={() => onSeek(seconds)}>Jump to {formatTime(seconds)}</button>}
            </div><ValueView value={item} />
          </article>
        }) : <p className="data-null">No entries</p>) : <ValueView value={value} />}
      </section>) : <ValueView value={data} />}
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
  const [point, setPoint] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [offline, setOffline] = useState(location.protocol === 'file:')

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
        setFile(names.includes(saved.file) ? saved.file : names[0] || '')
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
    async function load() {
      try {
        const result = offline ? snapshot[file] : await getJSON(`./api/data?file=${encodeURIComponent(file)}`, controller.signal)
        if (result.error) throw new Error(result.error)
        const entries = Array.isArray(result.data) ? result.data : [result.data]
        if (controller.signal.aborted) return
        const remembered = readSelection()
        const selected = remembered.file === file && Number.isInteger(remembered.point) && remembered.point >= 0 && remembered.point < entries.length ? remembered.point : 0
        setPoints(entries)
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

  const selected = points[point]
  const reviewData = selected && typeof selected === 'object' && !Array.isArray(selected) ? selected : null
  return <div className="gallery-layout">
    <aside className="file-sidebar" aria-label="Data files">
      <a className="brand" href="#">OMNI<span>DUPLEX</span></a>
      <h1>Data files <span>{files.length}</span></h1>
      <p className="source-label">data_generation/data</p>
      <p className="source-label">{offline ? 'Offline snapshot · rebuild to update' : 'Live directory · refresh to update'}</p>
      <nav>{files.map(name => <button type="button" key={name} className={`file-button${file === name ? ' is-selected' : ''}`} aria-current={file === name ? 'true' : undefined} onClick={() => selectFile(name)}>{name}</button>)}</nav>
    </aside>
    <main className="gallery-main">
      <header className="point-picker">
        <h2>{file || 'Gallery'} <span>{points.length} samples</span></h2>
        <div className="point-buttons" aria-label="Data points">{points.map((entry, index) => {
          const title = entry?.question || entry?.name || entry?.id || entry?.video_summary?.main_content_summary
          return <button type="button" key={index} aria-pressed={point === index} title={title ? String(title) : `Sample ${index + 1}`} onClick={() => { setPoint(index); saveSelection(file, index) }}>
            <strong>{String(index + 1).padStart(2, '0')}</strong><span>{title ? String(title) : `Sample ${index + 1}`}</span>
          </button>
        })}</div>
      </header>
      {loading ? <p className="gallery-message" role="status">Loading data…</p>
        : error ? <p className="gallery-message" role="alert">{error}</p>
        : !files.length ? <p className="gallery-message">No JSON files found in data_generation/data.</p>
        : !points.length ? <p className="gallery-message">This file contains no samples.</p>
        : reviewData ? <SampleReview key={`${file}:${point}`} sampleData={reviewData} />
        : <DataSections data={selected} onSeek={() => {}} />}
    </main>
  </div>
}
