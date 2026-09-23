export function flattenSamples(data, fingerprints) {
  const samples = Array.isArray(data) ? data : [data]
  return samples.flatMap((sample, sampleIndex) => sample.responses.map(
    (response, responseIndex) => ({
      key: `${sampleIndex}:${responseIndex}`,
      sampleIndex,
      responseIndex,
      sample,
      response,
      fingerprint: fingerprints[sampleIndex][responseIndex],
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
