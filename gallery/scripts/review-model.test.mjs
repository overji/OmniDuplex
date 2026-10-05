import { test } from 'node:test'
import assert from 'node:assert/strict'
import { parseTime, formatTime, reviewRecords, isActiveWindow, windowLanes, matchesFilters, initialFile, decodeJSON } from '../src/review-model.mjs'

const response = (start, time, end) => ({ window_start: start, trigger_time: time, window_end: end, response: 'Observed change' })

test('windows highlight their complete valid interval, including boundaries and overlap', () => {
  const records = reviewRecords({ responses: [response('00:10', '00:12', '00:20'), response('00:15', '00:16', '00:25'), response('00:30', '00:30', '00:30')] })
  assert.equal(isActiveWindow(records[0], 9.9), false)
  assert.equal(isActiveWindow(records[0], 10), true)
  assert.equal(isActiveWindow(records[0], 20), true)
  assert.equal(isActiveWindow(records[0], 20.1), false)
  assert.equal(records.filter(record => isActiveWindow(record, 17)).length, 2)
  assert.equal(isActiveWindow(records[2], 30), true)
  assert.equal(isActiveWindow(records[2], 30.1), false)
  assert.deepEqual(windowLanes(records).map(record => record.lane), [0, 1, 0])
})

test('Agent messages retain equal-time order and do not inherit eligibility windows', () => {
  const sample = { mode: 'agent', sample_type: 'positive', responses: [
    { ...response('00:10', '00:12', '00:20'), role: 'assistant', tool_calls: [{ function: { name: 'agent_call' } }] },
    { trigger_time: '00:25', role: 'tool', response: '{"simulated":true,"status":"success"}' },
    { trigger_time: '00:25', role: 'assistant', response: 'Completed', outcome: 'success' },
  ] }
  const records = reviewRecords(sample)
  assert.deepEqual(records.map(record => record.label), ['Agent delegation', 'Simulated tool result', 'Agent reply'])
  assert.deepEqual(records.map(record => record.hasWindow), [true, false, false])
  assert.equal(decodeJSON(records[1].response).simulated, true)
  assert.equal(decodeJSON('{broken'), '{broken')
})

test('negative refusals and review annotations remain separate', () => {
  for (const sample_type of ['missing_tool', 'no_context']) {
    const sample = { mode: 'agent', sample_type, responses: [response('00:00', '00:00', '00:00')], annotations: [response('00:10', '00:12', '00:20')] }
    const records = reviewRecords(sample)
    assert.deepEqual(records.map(record => record.label), ['Refusal', 'Observation annotation'])
    assert.deepEqual(windowLanes(records).map(record => [record.track, record.lane]), [['responses', 0], ['annotations', 0]])
    assert.equal(matchesFilters(sample, { kind: 'negative', mode: 'agent' }), true)
    assert.equal(matchesFilters(sample, { kind: 'positive' }), false)
  }
})

test('legacy, missing and malformed timestamps stay usable without invented windows', () => {
  assert.equal(parseTime('01:02:03.5'), 3723.5)
  assert.equal(parseTime('00:00'), 0)
  assert.equal(parseTime('00:99'), null)
  assert.equal(parseTime(''), null)
  assert.equal(parseTime(-1), null)
  assert.equal(formatTime(null), '—')
  const records = reviewRecords({ responses: [{ trigger_time: '00:10' }, response('00:20', '00:10', '00:30'), { response: 'No time' }] })
  assert.deepEqual(records.map(record => record.hasWindow), [false, false, false])
  assert.equal(records[2].seconds, null)
  assert.equal(reviewRecords({ audio_events: [{ start_seconds: 5, event_name: 'Speech' }] })[0].seconds, 5)
})

test('filters combine and default selection preserves valid history', () => {
  const sample = { ability: 'grounding', mode: 'pr', video: 'video_11.mp4' }
  assert.equal(matchesFilters(sample, { ability: 'grounding', mode: 'pr', kind: 'positive', video: 'video_11.mp4' }), true)
  assert.equal(matchesFilters(sample, { ability: 'grounding', video: 'video_1.mp4' }), false)
  const files = ['old.json', 'qwen3.8_omni_flash_44/all_qa.json']
  assert.equal(initialFile(files, 'old.json'), 'old.json')
  assert.equal(initialFile(files, 'deleted.json'), files[1])
  assert.equal(initialFile([], ''), '')
})
