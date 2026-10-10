import { agreement, progress } from './annotation-state.mjs'

const percent = value => value === null ? 'N/A' : `${(value * 100).toFixed(1)}%`
const coefficient = value => value === null ? 'N/A' : value.toFixed(3)

export function ResponseRating({ record, disabled, onRate }) {
  return <div className="response-rating" aria-label="Human response scores">
    {['time', 'content'].map(dimension => <fieldset key={dimension} disabled={disabled}>
      <legend>{dimension === 'time' ? 'Response timing' : 'Response content'}</legend>
      <span>{dimension === 'time' ? 'Is this an appropriate response time?' : 'Is this accurate and appropriate to the request?'}</span>
      <div>{[1, 0].map(score => <button type="button" key={score}
        aria-pressed={record?.[`${dimension}Score`] === score}
        onClick={() => onRate(dimension, score)}>{score ? 'Pass' : 'Fail'}</button>)}</div>
    </fieldset>)}
  </div>
}

export function AnnotationSummary({ items, groups, annotator }) {
  const stats = progress(items, groups[annotator])
  return <section className="annotation-summary" aria-label="Annotation progress and agreement">
    <p>Current file and filters · Annotator {annotator} · {stats.complete} / {stats.total} responses complete</p>
    <div className="annotation-progress"><span>Timing pass rate: {percent(stats.timeRated ? stats.timePassed / stats.timeRated : null)} ({stats.timeRated} rated)</span>
      <span>Content pass rate: {percent(stats.contentRated ? stats.contentPassed / stats.contentRated : null)} ({stats.contentRated} rated)</span></div>
    <div className="agreement-table"><table><caption>Annotator A ↔ B · scores paired separately for each dimension</caption>
      <thead><tr><th>Dimension</th><th>Paired responses</th><th>Agreement</th><th>Cohen’s Kappa</th><th>Pearson / Phi</th></tr></thead>
      <tbody>{['time', 'content'].map(dimension => {
        const result = agreement(items, groups.A, groups.B, dimension)
        return <tr key={dimension}><th>{dimension === 'time' ? 'Timing' : 'Content'}</th><td>{result.n}</td>
          <td>{percent(result.rate)}</td><td>{coefficient(result.kappa)}{result.kappaReason && <small>{result.kappaReason}</small>}</td>
          <td>{coefficient(result.pearson)}{result.pearsonReason && <small>{result.pearsonReason}</small>}</td></tr>
      })}</tbody></table></div>
    <p>Each score saves immediately. Both scores must save before moving to the next unfinished response. Scores sync every five seconds.</p>
  </section>
}
