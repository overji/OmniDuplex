export function flattenSamples(data, fingerprints) {
  const samples = Array.isArray(data) ? data : [data]
  return samples.flatMap((sample, sampleIndex) => (Array.isArray(sample?.responses) ? sample.responses : []).map(
    (response, responseIndex) => ({
      key: `${sampleIndex}:${responseIndex}`,
      sampleIndex,
      responseIndex,
      sample,
      response,
      fingerprint: fingerprints[sampleIndex]?.[responseIndex],
    }),
  ))
}

export function isComplete(record) {
  return [0, 1].includes(record?.timeScore) && [0, 1].includes(record?.contentScore)
}

export function firstIncomplete(items, records) {
  return items.findIndex(item => !isComplete(records[item.key]))
}

export function progress(items, records) {
  const result = { total: items.length, complete: 0, timeRated: 0, timePassed: 0,
    contentRated: 0, contentPassed: 0 }
  for (const item of items) {
    const record = records[item.key]
    if (isComplete(record)) result.complete++
    if ([0, 1].includes(record?.timeScore)) {
      result.timeRated++
      result.timePassed += record.timeScore
    }
    if ([0, 1].includes(record?.contentScore)) {
      result.contentRated++
      result.contentPassed += record.contentScore
    }
  }
  return result
}

// A response contributes independently to each dimension once both annotators rated it.
export function agreement(items, a, b, dimension) {
  const field = `${dimension}Score`
  const pairs = items.map(item => [a[item.key]?.[field], b[item.key]?.[field]])
    .filter(pair => pair.every(score => [0, 1].includes(score)))
  const n = pairs.length
  if (!n) return { n, rate: null, kappa: null, pearson: null, kappaReason: 'No paired scores', pearsonReason: 'No paired scores' }
  const sumA = pairs.reduce((sum, pair) => sum + pair[0], 0)
  const sumB = pairs.reduce((sum, pair) => sum + pair[1], 0)
  const sumAB = pairs.reduce((sum, pair) => sum + pair[0] * pair[1], 0)
  const rate = pairs.filter(([x, y]) => x === y).length / n
  const chance = (sumA * sumB + (n - sumA) * (n - sumB)) / (n * n)
  const denominator = Math.sqrt(sumA * (n - sumA) * sumB * (n - sumB))
  return { n, rate, kappa: chance < 1 ? (rate - chance) / (1 - chance) : null,
    pearson: n >= 2 && denominator > 0 ? (n * sumAB - sumA * sumB) / denominator : null,
    kappaReason: chance === 1 ? 'Expected agreement is 100%; Kappa is undefined' : '',
    pearsonReason: n < 2 ? 'At least two paired scores are required' : denominator === 0 ? 'One or both annotators have constant scores' : '' }
}
