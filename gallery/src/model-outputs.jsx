import { useEffect, useState } from 'react'
import { formatOnset, predictionRecords } from './model-output-state.mjs'

export function useModelOutputs(file, sampleIndex, enabled) {
  const key = `${file}:${sampleIndex}`
  const [state, setState] = useState({ key: null, models: [], loading: false, error: '', message: '' })
  useEffect(() => {
    if (!enabled || !file || !Number.isInteger(sampleIndex)) return
    const controller = new AbortController()
    setState({ key, models: [], loading: true, error: '', message: '' })
    async function load() {
      try {
        const response = await fetch(`./api/model-outputs?${new URLSearchParams({ file, sampleIndex })}`, { signal: controller.signal, cache: 'no-store' })
        const body = await response.json()
        if (!response.ok) throw new Error(body.error || 'Unable to load model outputs')
        if (!controller.signal.aborted) setState({ key, models: body.models, loading: false, error: '', message: body.message })
      } catch (error) {
        if (!controller.signal.aborted) setState({ key, models: [], loading: false, error: error.message, message: '' })
      }
    }
    load()
    return () => controller.abort()
  }, [file, sampleIndex, enabled])
  return enabled && state.key === key ? state : { models: [], loading: enabled, error: '', message: '' }
}

const percent = value => value === null ? 'N/A' : `${(value * 100).toFixed(2)}%`

export function ModelOutputs({ state, model, modelId, onModelChange, onSeek }) {
  return <section className="model-outputs" aria-label="Model outputs">
    <h2>Model outputs</h2>
    {state.loading && <p role="status">Loading model outputs…</p>}
    {state.error && <p role="alert">{state.error}</p>}
    {state.models.length > 0 && <label>Model <select aria-label="Output model" value={model?.id || modelId} onChange={event => onModelChange(event.target.value)}>
      {state.models.map(row => <option key={row.id} value={row.id}>{row.name} · {row.evaluatedSamples}/{row.totalSamples}</option>)}
    </select></label>}
    {state.message && <p role="status">{state.message}</p>}
    {model?.state === 'not_evaluated' && <p role="status">This sample has not been evaluated by {model.name}.</p>}
    {model?.state === 'evaluated' && <>
      <p>Run coverage: {model.evaluatedSamples}/{model.totalSamples} samples · Sample status: {model.prediction.status}</p>
      <p>Timing only · {model.score.hits}/{model.score.windows} windows hit · {model.score.predictions} predictions</p>
      <div className="model-score-summary">
        <span>Window hit rate: {percent(model.score.windows ? model.score.hits / model.score.windows : null)}</span>
        <span>Precision: {percent(model.score.predictions ? model.score.hits / model.score.predictions : 0)}</span>
        <span>F1: {percent(model.score.windows + model.score.predictions ? 2 * model.score.hits / (model.score.windows + model.score.predictions) : 0)}</span>
      </div>
      {!model.prediction.events.length && <p role="status">Evaluated — no response generated.</p>}
      <div className="model-predictions">{predictionRecords(model).map(event => <article className={`model-prediction${event.match ? ' is-matched' : ''}`} key={event.id}>
        <button className="time-button" type="button" disabled={event.seconds === null} onClick={() => onSeek(event.seconds)}>
          Seek <strong>{event.trigger_time}</strong>
        </button>
        <span className="model-match-label">{event.label}</span>
        {event.match && <p>Reference window: {formatOnset(event.match.window_start_seconds)} → {formatOnset(event.match.window_end_seconds)}</p>}
        <p className="model-prediction-text">{event.text || '(Empty response)'}</p>
      </article>)}</div>
    </>}
  </section>
}
