import { useEffect, useMemo, useRef, useState } from 'react'
import { firstIncomplete, flattenSamples, progress } from './annotation-state.mjs'

async function getJson(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options })
  const body = await response.json()
  if (!response.ok) throw new Error(body.error || `请求失败 (${response.status})`)
  return body
}

function parseTime(value) {
  if (typeof value !== 'string') return null
  const parts = value.split(':').map(Number)
  if (![2, 3].includes(parts.length) || parts.some(part => !Number.isInteger(part) || part < 0)) return null
  return parts.reduce((total, part) => total * 60 + part, 0)
}

function rate(passed, rated) {
  return rated ? `${Math.round((passed / rated) * 100)}%` : '—'
}

function ScoreGroup({ title, hint, value, onRate, disabled }) {
  return <fieldset className="score-group" disabled={disabled}>
    <legend>{title}</legend>
    <p>{hint}</p>
    <div className="score-options">
      <button type="button" className={value === 1 ? 'selected pass' : ''} aria-pressed={value === 1} onClick={() => onRate(1)}>通过</button>
      <button type="button" className={value === 0 ? 'selected fail' : ''} aria-pressed={value === 0} onClick={() => onRate(0)}>不通过</button>
    </div>
  </fieldset>
}

export default function App() {
  const [files, setFiles] = useState([])
  const [file, setFile] = useState('')
  const [data, setData] = useState(null)
  const [fingerprints, setFingerprints] = useState([])
  const [records, setRecords] = useState({})
  const [activeIndex, setActiveIndex] = useState(-1)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [saveError, setSaveError] = useState('')
  const [saveStatus, setSaveStatus] = useState('')
  const [videoError, setVideoError] = useState(false)
  const videoRef = useRef(null)

  useEffect(() => {
    const controller = new AbortController()
    getJson('./api/files', { signal: controller.signal }).then(({ files: names }) => {
      if (controller.signal.aborted) return
      setFiles(names)
      setFile(names[0] || '')
      if (!names.length) setLoading(false)
    }).catch(err => {
      if (!controller.signal.aborted) { setError(err.message); setLoading(false) }
    })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!file) return
    const controller = new AbortController()
    setLoading(true)
    setError('')
    setSaveError('')
    setSaveStatus('')
    setData(null)
    Promise.all([
      getJson(`./api/data?file=${encodeURIComponent(file)}`, { signal: controller.signal }),
      getJson(`./api/annotations?file=${encodeURIComponent(file)}`, { signal: controller.signal }),
    ]).then(([source, saved]) => {
      if (controller.signal.aborted) return
      const items = flattenSamples(source.data, source.fingerprints)
      setData(source.data)
      setFingerprints(source.fingerprints)
      setRecords(saved.records)
      setActiveIndex(firstIncomplete(items, saved.records))
    }).catch(err => {
      if (!controller.signal.aborted) setError(err.message)
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [file])

  const samples = useMemo(() => data ? (Array.isArray(data) ? data : [data]) : [], [data])
  const items = useMemo(() => data ? flattenSamples(data, fingerprints) : [], [data, fingerprints])
  const stats = useMemo(() => progress(items, records), [items, records])
  const active = items[activeIndex]
  const activeRecord = active ? records[active.key] : null
  const videoName = active?.sample.video

  useEffect(() => setVideoError(false), [videoName])

  function seek() {
    const seconds = parseTime(active?.response.trigger_time)
    if (seconds !== null && videoRef.current) videoRef.current.currentTime = seconds
  }

  async function saveScore(dimension, score) {
    if (!active || saving) return
    setSaving(true)
    setSaveError('')
    setSaveStatus('保存中…')
    try {
      const saved = await getJson('./api/annotations', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file, sampleIndex: active.sampleIndex,
          responseIndex: active.responseIndex, fingerprint: active.fingerprint,
          dimension, score }),
      })
      const updated = { ...records, [saved.key]: saved.record }
      setRecords(updated)
      setSaveStatus('已保存')
      if ([0, 1].includes(saved.record.timeScore) && [0, 1].includes(saved.record.contentScore)) {
        setActiveIndex(firstIncomplete(items, updated))
      }
    } catch (err) {
      setSaveError(err.message)
      setSaveStatus('保存失败')
    } finally {
      setSaving(false)
    }
  }

  return <div className="layout">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">O</span><span>OMNI<span className="brand-accent">DUPLEX</span></span></div>
      <div className="sidebar-title"><span>标注工作台</span><strong>数据文件</strong><small>{files.length} 个文件</small></div>
      <nav aria-label="数据文件列表" className="file-list">
        {files.map(name => <button type="button" key={name} disabled={saving}
          className={`file-item ${name === file ? 'active' : ''}`} aria-current={name === file ? 'true' : undefined}
          onClick={() => setFile(name)}><span className="file-icon">▤</span><span>{name}</span></button>)}
      </nav>
      <div className="sidebar-foot"><span className="live-dot" />实时保存至服务端</div>
    </aside>

    <main className="main">
      <header className="topbar"><div><span className="eyebrow">ANNOTATION GALLERY / {file || 'NO FILE'}</span><h1>逐条检查视频响应</h1></div>
        <div className="status-pill">{loading ? '加载中' : saving ? '保存中' : saveStatus || '已同步'}</div></header>
      {error ? <div className="notice error" role="alert">{error}</div>
        : loading ? <div className="notice" role="status">正在加载标注数据…</div>
        : !files.length ? <div className="notice">数据目录中没有可标注的 QA JSON 文件。</div>
        : <>
          <section className="metrics" aria-label="标注进度">
            <div className="metric"><span>已完成</span><strong>{stats.complete}<small> / {stats.total}</small></strong><div className="mini-track"><i style={{ width: `${stats.total ? stats.complete / stats.total * 100 : 0}%` }} /></div></div>
            <div className="metric"><span>时间点通过率</span><strong>{rate(stats.timePassed, stats.timeRated)}</strong><small>{stats.timeRated} 条已评分</small></div>
            <div className="metric"><span>回复内容通过率</span><strong>{rate(stats.contentPassed, stats.contentRated)}</strong><small>{stats.contentRated} 条已评分</small></div>
          </section>

          <section className="sample-nav" aria-label="样本列表"><div className="section-label">样本 <span>{samples.length}</span></div>
            <div className="sample-tabs">{samples.map((sample, index) => {
              const sampleItems = items.filter(item => item.sampleIndex === index)
              const done = sampleItems.filter(item => [0, 1].includes(records[item.key]?.timeScore) && [0, 1].includes(records[item.key]?.contentScore)).length
              return <button type="button" key={index} disabled={!sampleItems.length || saving}
                className={active?.sampleIndex === index ? 'active' : ''}
                onClick={() => { setActiveIndex(items.indexOf(sampleItems.find(item => ![0, 1].includes(records[item.key]?.timeScore) || ![0, 1].includes(records[item.key]?.contentScore)) || sampleItems[0])); setSaveError('') }}
                title={sample.question}><span>{String(index + 1).padStart(2, '0')}</span><strong>{sample.question}</strong><small>{done}/{sampleItems.length}</small></button>
            })}</div>
          </section>

          {saveError && <div className="notice error" role="alert">{saveError}。请检查服务端后重试；如果源数据已变化，请刷新页面。</div>}
          {active ? <div className="work-grid">
            <div className="review-column">
              <section className="video-card"><div className="card-head"><div><span className="eyebrow">SOURCE VIDEO</span><h2>{videoName || '未指定视频'}</h2></div><span className="time-badge">{active.response.trigger_time || '无时间'}</span></div>
                {videoName ? <video key={videoName} ref={videoRef} src={`./api/video?name=${encodeURIComponent(videoName)}`} controls preload="metadata" playsInline
                  onError={() => setVideoError(true)} onLoadedMetadata={() => setVideoError(false)} /> : <div className="video-missing">当前样本没有视频文件名</div>}
                {videoError && <div className="video-warning" role="alert">无法加载视频，请检查视频目录中是否存在 {videoName}。</div>}
                <div className="video-actions"><button type="button" onClick={seek}>跳到响应时间 {active.response.trigger_time || '—'}</button><span>播放视频，核对时间与内容</span></div>
              </section>
              <section className="context-card"><div className="eyebrow">USER REQUEST · {active.sample.question_time || '—'}</div><h2>{active.sample.question}</h2><div className="context-meta"><span>任务：{active.sample.task || '未指定'}</span><span>视频：{active.sample.video_folder || videoName || '未指定'}</span></div></section>
            </div>

            <div className="annotation-column"><section className="response-card"><div className="card-head"><div><span className="eyebrow">RESPONSE {activeIndex + 1} / {items.length}</span><h2>当前响应</h2></div><span className="role-badge">{active.response.role || '普通回复'}</span></div>
              <div className="response-time">触发时间 <strong>{active.response.trigger_time || '—'}</strong></div>
              <div className="response-text">{active.response.response || '无回复内容'}</div>
              <details><summary>查看完整响应字段</summary><pre>{JSON.stringify(active.response, null, 2)}</pre></details>
            </section>
            <section className="rating-card"><div className="card-head"><div><span className="eyebrow">HUMAN REVIEW</span><h2>人工评分</h2></div><span className="save-indicator">{saveStatus || '等待评分'}</span></div>
              <ScoreGroup title="时间点" hint="该响应是否在合适的视频时间点触发？" value={activeRecord?.timeScore} disabled={saving} onRate={score => saveScore('time', score)} />
              <ScoreGroup title="回复内容" hint="回复内容是否准确且符合用户请求？" value={activeRecord?.contentScore} disabled={saving} onRate={score => saveScore('content', score)} />
              <p className="rating-note">每次点击立即保存。两项都评分后自动进入下一条未完成的响应。</p>
            </section></div>
          </div> : <div className="notice complete"><strong>{items.length ? '当前文件全部标注完成' : '当前文件没有 response'}</strong><p>可以从样本列表返回检查或修改已保存的评分。</p></div>}
          <section className="response-nav" aria-label="响应列表"><div className="section-label">全部响应 <span>{items.length}</span></div><div className="response-tabs">{items.map((item, index) => {
            const record = records[item.key]
            const done = [0, 1].includes(record?.timeScore) && [0, 1].includes(record?.contentScore)
            return <button type="button" key={item.key} disabled={saving} className={`${index === activeIndex ? 'active' : ''} ${done ? 'done' : ''}`}
              onClick={() => { setActiveIndex(index); setSaveError('') }}><span>{String(index + 1).padStart(2, '0')}</span><strong>{item.response.response || '无回复内容'}</strong><small>{done ? '已完成' : '待标注'}</small></button>
          })}</div></section>
        </>}
    </main>
  </div>
}
