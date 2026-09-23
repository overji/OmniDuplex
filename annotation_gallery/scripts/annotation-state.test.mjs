import { test } from 'node:test'
import assert from 'node:assert/strict'
import { firstIncomplete, flattenSamples, progress } from '../src/annotation-state.mjs'

test('resume chooses first incomplete response and progress counts only saved scores', () => {
  const data = [{ question: 'Q1', responses: [{ response: 'A' }, { response: 'B' }] },
    { question: 'Q2', responses: [{ role: 'tool', response: '{}' }] }]
  const items = flattenSamples(data, [['a', 'b'], ['c']])
  assert.deepEqual(items.map(item => item.key), ['0:0', '0:1', '1:0'])
  const records = { '0:0': { timeScore: 1, contentScore: 0 },
    '0:1': { timeScore: 1 }, '1:0': { timeScore: 0, contentScore: 1 } }
  assert.equal(firstIncomplete(items, records), 1)
  assert.deepEqual(progress(items, records), { total: 3, complete: 2, timeRated: 3,
    timePassed: 2, contentRated: 2, contentPassed: 1 })
  records['0:1'].contentScore = 1
  assert.equal(firstIncomplete(items, records), -1)
})
