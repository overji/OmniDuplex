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

test('agreement uses per-dimension paired scores and handles undefined coefficients', async () => {
  const { agreement } = await import('../src/annotation-state.mjs')
  const items = ['0:0', '0:1', '0:2', '1:0'].map(key => ({ key }))
  const records = values => Object.fromEntries(items.map((item, index) => [item.key, { timeScore: values[index] }]))
  const a = records([0, 0, 1, 1])
  assert.deepEqual(agreement(items, a, a, 'time'), { n: 4, rate: 1, kappa: 1, pearson: 1, kappaReason: '', pearsonReason: '' })
  const inverse = agreement(items, a, records([1, 1, 0, 0]), 'time')
  assert.equal(inverse.kappa, -1)
  assert.equal(inverse.pearson, -1)
  const mixed = agreement(items, a, records([0, 1, 0, 1]), 'time')
  assert.equal(mixed.kappa, 0)
  assert.equal(mixed.pearson, 0)
  const unbalanced = agreement(items, records([0, 0, 0, 1]), records([0, 0, 1, 1]), 'time')
  assert.equal(unbalanced.rate, 0.75)
  assert.equal(unbalanced.kappa, 0.5)
  assert.ok(Math.abs(unbalanced.pearson - 1 / Math.sqrt(3)) < 1e-12)
  assert.equal(agreement(items, a, a, 'content').n, 0)
  const constant = agreement(items, records([1, 1, 1, 1]), records([1, 1, 1, 1]), 'time')
  assert.equal(constant.kappa, null)
  assert.equal(constant.pearson, null)
  assert.equal(constant.rate, 1)
  assert.match(constant.kappaReason, /undefined/)
  const single = agreement(items, { '0:0': { timeScore: 0 } }, { '0:0': { timeScore: 1 } }, 'time')
  assert.equal(single.n, 1)
  assert.equal(single.kappa, 0)
  assert.equal(single.pearson, null)
})

test('filtered resume retains source indices and observation annotations are excluded', async () => {
  const { matchesFilters } = await import('../src/review-model.mjs')
  const data = [
    { mode: 'rtd', responses: [{}, {}], annotations: [{}] },
    { mode: 'agent', responses: [{ role: 'tool' },
      { role: 'assistant', tool_calls: [{ function: { name: 'agent_call' } }] },
      { role: 'assistant', tool_calls: [{ function: { name: 'agent_call' } }] }] },
  ]
  const items = flattenSamples(data, [['a', 'b'], ['c', 'd', 'e']]).filter(item => matchesFilters(item.sample, { mode: 'agent' }))
  assert.deepEqual(items.map(item => item.key), ['1:1', '1:2'])
  const a = { '1:1': { timeScore: 1, contentScore: 0 } }
  assert.equal(firstIncomplete(items, a), 1)
  assert.equal(firstIncomplete(items, {}), 0)
})

test('Agent scoring restores completion replies and excludes only tool inputs', async () => {
  const { agreement } = await import('../src/annotation-state.mjs')
  const call = { role: 'assistant', tool_calls: [{ function: { name: 'agent_call' } }] }
  const data = [
    { mode: 'agent', responses: [call, { role: 'tool' }, { role: 'assistant', outcome: 'success' }, call] },
    { mode: 'agent', sample_type: 'missing_tool', responses: [{ response: 'Cannot do that' }] },
    { mode: 'agent', generation_context: { sample_type: 'no_context' }, responses: [{}] },
    { mode: 'rtd', responses: [{}] }, { mode: 'pr', responses: [{}] },
  ]
  const items = flattenSamples(data, [['a', 'b', 'c', 'd'], ['e'], ['f'], ['g'], ['h']])
  assert.deepEqual(items.map(item => item.key), ['0:0', '0:2', '0:3', '1:0', '2:0', '3:0', '4:0'])
  assert.equal(items[1].fingerprint, 'c')
  const records = { '0:0': { timeScore: 1, contentScore: 1 },
    '0:1': { timeScore: 0, contentScore: 0 }, '0:2': { timeScore: 0, contentScore: 0 } }
  assert.equal(firstIncomplete(items, records), 2)
  assert.equal(progress(items, records).total, 7)
  assert.equal(progress(items, records).timeRated, 2)
  assert.equal(agreement(items, records, records, 'time').n, 2)
})
