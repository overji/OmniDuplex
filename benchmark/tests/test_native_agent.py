"""CPU contracts for native actions; no model weights or backend execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from benchmark.cli.main import build_parser
from benchmark.data.agent_inputs import input_protocol
from benchmark.data.agent_schema import TASK_START
from benchmark.data.native_agent import NativeAgentEvents, VERSION, build_prompt, call_event, joyai_event, tools
from benchmark.evaluation.agent_calls import parse_event
from benchmark.runtime.runner import SampleContext


SAMPLE = dict(mode='agent', task='event_agent', video='v.mp4', question='Save it',
              question_time='00:01', _agent_call_schema='task-start', available_tools=['note_write'])
CALL = dict(name='task_start', arguments=dict(instruction='Save the observed event.'))


def speech(text='Saving it.', ended=True):
    return dict(text=text, end_of_turn=ended, is_listen=False)


def action(**overrides):
    return dict(is_tool_call=True, tool_calls=[CALL], tool_generation_complete=True,
                raw_tool_text=json.dumps(CALL), **overrides)


def collect(*items):
    stream = NativeAgentEvents()
    events = []
    for clock, item in enumerate(items, 1):
        events.extend(stream.feed(item, clock))
    return events + stream.finish()


class NativeAgentTests(unittest.TestCase):
    def test_cli_defaults_explicit_old_modes_and_schema_guard(self):
        parser = build_parser()
        for model in ('joyai', 'gander'):
            command = ['parallel', model, '--model-path', '/model', '--gpus', '0', '1']
            if model == 'gander':
                command += ['--base-model-path', '/base', '--official-code', '/code']
            args = parser.parse_args(command)
            self.assertEqual(args.agent_output_protocol, 'native')
            self.assertIn(VERSION, input_protocol(args))
            self.assertEqual(parser.parse_args(command + ['--agent-output-protocol', 'task-text']).agent_output_protocol, 'task-text')
        with self.assertRaises(ValueError):
            input_protocol(NS(agent_output_protocol='native', agent_call_schema='legacy'))

    def test_prompt_and_schema_no_reference_or_legacy_format_leakage(self):
        sample = dict(SAMPLE, responses=[{'response': 'SECRET'}],
                      frontend_context={'system_prompt': 'OLD JSON FORMAT'})
        for model in ('gander', 'joyai'):
            prompt = build_prompt(sample, model)
            self.assertNotIn('SECRET', prompt)
            self.assertNotIn('OLD JSON FORMAT', prompt)
            self.assertIn('- note_write:', prompt)
            self.assertNotIn('- web_search:', prompt)
        self.assertEqual(tools(sample), [TASK_START])
        self.assertEqual(tools(dict(sample, available_tools=[])), [])
        clone = tools(sample)
        clone[0]['function']['name'] = 'changed'
        self.assertEqual(TASK_START['function']['name'], 'task_start')

    def test_speech_then_call_and_call_then_speech_share_earliest_onset(self):
        for items, response_time, call_time in (([speech(), action()], 1, 2),
                                               ([action(), speech()], 2, 1)):
            events = collect(*items)
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event['onset_seconds'], 1)
            self.assertEqual(event['response_onset_seconds'], response_time)
            self.assertEqual(event['agent_call_onset_seconds'], call_time)
            self.assertEqual(event['agent_call_complete_seconds'], call_time)
            self.assertEqual(event['response_text'], 'Saving it.')
            self.assertEqual(parse_event(event)['calls'][0]['function'], CALL)
            self.assertEqual(json.loads(event['native_raw_tool_text']), CALL)

    def test_multichunk_speech_in_both_orders(self):
        for items in ((speech('Save ', False), speech('it.'), action()),
                      (action(), speech('Save ', False), speech('it.'))):
            events = collect(*items)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['response_text'], 'Save it.')
            self.assertEqual(events[0]['onset_seconds'], 1)

    def test_barriers_do_not_pair_over_a_gap(self):
        for barrier in ({'is_listen': True}, {'is_interrupt': True}, {}, {'text': '', 'other_action': True}):
            for first, last in ((speech(), action()), (action(), speech())):
                events = collect(first, barrier, last)
                self.assertEqual(len(events), 2)
                self.assertFalse(any(e['response_text'] and e['tool_calls'] for e in events))
        events = collect(speech('unfinished', False), {}, action())
        self.assertEqual(len(events), 2)
        self.assertFalse(events[0]['response_generation_complete'])

    def test_repeated_actions_and_unpaired_components_are_preserved(self):
        events = collect(speech(), action(), action(), speech())
        self.assertEqual(len(events), 2)
        self.assertEqual(sum(len(e['tool_calls']) for e in events), 2)
        self.assertEqual(collect(action())[0]['response_text'], '')
        event = collect(speech('unfinished', False))[0]
        self.assertEqual(event['tool_calls'], [])
        self.assertFalse(event['response_generation_complete'])

    def test_invalid_calls_are_attempts_not_recovered_calls(self):
        cases = [dict(tool_generation_complete=False), dict(tool_generation_complete=None),
                 dict(tool_error='bad'), dict(tool_parse_valid=False), dict(tool_schema_valid=False),
                 dict(tool_calls=[CALL, CALL]), dict(tool_calls=[{'name': 'task_start', 'arguments': {'name': 'note_write'}}]),
                 dict(tool_calls=[{'name': 'task_start', 'arguments': '{broken'}])]
        for overrides in cases:
            result = action()
            result.update(overrides)
            event = call_event(result, 3)
            self.assertEqual(parse_event(event)['calls'], [])
            self.assertTrue(parse_event(event)['attempted'])
            self.assertEqual(event['native_tool_calls'], result['tool_calls'])

    def test_joyai_native_delegation_raw_payload_and_missing_fields(self):
        for marker in ('</delegation>', '<delegation>'):
            event = joyai_event('</response> Saving it. ' + marker + ' Save it.', 2)
            self.assertEqual(event['response_text'], 'Saving it.')
            self.assertEqual(parse_event(event)['calls'][0]['function']['arguments']['instruction'], 'Save it.')
            self.assertIsNone(event['native_generation_complete'])
            self.assertIsNone(event['agent_call_complete_seconds'])
        self.assertIsNone(joyai_event('</silence>', 1))
        self.assertEqual(joyai_event('</delegation> Save it.', 1)['response_text'], '')
        for text in ('</response> Saving. </delegation>', '</delegation> A </delegation> B'):
            self.assertEqual(joyai_event(text, 1)['tool_calls'], [])
        event = joyai_event('</response> Saving. </delegation> Save it.', 1, complete=False)
        self.assertEqual(event['agent_parse_status'], 'truncated')
        self.assertEqual(event['tool_calls'], [])
        for text in ('I will call task_start now', json.dumps(CALL)):
            self.assertEqual(parse_event(joyai_event(text, 1))['calls'], [])

    def test_existing_judge_and_scored_f1_consume_real_components(self):
        from benchmark.evaluation.judge import prepare_rows, evaluate
        from benchmark.evaluation.scored_f1 import normalized_quality, scored_metrics
        sample = dict(SAMPLE, responses=[dict(response='Saving it.', window_start=1, window_end=4,
                                             role='assistant', tool_calls=[{'function': CALL}])])
        event = collect(speech(), action())[0]
        prediction = dict(sample_index=0, status='complete', events=[event],
                          **{k: sample[k] for k in ('task', 'video', 'question')})
        rows, _ = prepare_rows([sample], [prediction])
        self.assertEqual(rows[0]['payload']['candidate_response'], 'Saving it.')
        self.assertTrue(rows[0]['payload']['judge_call'])
        self.assertEqual(rows[0]['payload']['candidate_tool_calls'][0]['function'], CALL)
        def request(config, key, system, payload):
            return json.dumps(dict(response_score=2, call_score=2 if payload['judge_call'] else None, response_reason='ok',
                                   call_reason='ok', error_types=[])), {}, True
        quality = normalized_quality(evaluate(rows[0], {'retries': 0}, '', '', request))
        self.assertEqual(scored_metrics(quality, 1, 1)['scored_f1'], 1)
        for component in (speech(), action()):
            prediction['events'] = collect(component)
            rows, _ = prepare_rows([sample], [prediction])
            result = evaluate(rows[0], {'retries': 0}, '', '', request)
            # A valid judge cannot award a score to a missing call; empty speech
            # is forced to zero even if the judge tries to award full credit.
            self.assertEqual(normalized_quality(result), .5)

    def test_resume_rejects_old_protocol_before_model_load(self):
        from benchmark.runtime.runner import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data.json'
            data.write_text('[]')
            prompt = root / 'prompt.txt'
            prompt.write_text('Base')
            (root / 'predictions.jsonl').write_text('')
            (root / 'run_config.json').write_text(json.dumps({'agent_input_protocol': 'old-task-text'}))
            args = NS(data=data, system_prompt=prompt, output_dir=root,
                      agent_output_protocol='native', agent_call_schema='task-start')
            adapter = NS(run_config=Mock(return_value={}), load=Mock())
            with self.assertRaisesRegex(ValueError, 'Agent input protocol changed'):
                run(args, adapter)
            adapter.load.assert_not_called()

    def test_native_only_changes_agent_prompts_and_hash_tracks_template(self):
        from benchmark.data.native_agent import identity
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('pr', 'rtd'):
                sample = dict(SAMPLE, mode=mode)
                old = SampleContext(0, sample, 'Base', Path(tmp), task_text_model='joyai')
                new = SampleContext(0, sample, 'Base', Path(tmp), agent_output_protocol='native', task_text_model='joyai')
                self.assertEqual(old.build_prompt('Native controls'), new.build_prompt('Native controls'))
        baseline = identity()
        read = Path.read_bytes
        def changed(path):
            return read(path) + (b'changed' if path.name == 'agent_native_joyai.txt' else b'')
        with patch.object(Path, 'read_bytes', changed):
            self.assertNotEqual(identity(), baseline)

    def test_gander_media_question_order_and_native_runtime_schema(self):
        import numpy as np
        from benchmark.adapters import gander, minicpmo
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = []
            replies = iter([speech('PREQUESTION'), speech(), action()])
            def prefill(**kwargs):
                trace.append('question' if 'text_list' in kwargs else 'media')
                return {'success': True}
            def generate(**kwargs):
                trace.append('generate')
                return next(replies)
            model = NS(prepare=Mock(), streaming_prefill=prefill, streaming_generate=generate)
            args = NS(seed=1, videos_dir=root, cache_dir=root, max_seconds=0,
                      agent_output_protocol='native', decode_mode='greedy', max_speak_tokens=11)
            context = SampleContext(0, SAMPLE, 'Base', root, agent_output_protocol='native', task_text_model='gander')
            meta = dict(duration=3., frame_times=[], frames=[])
            with patch.object(minicpmo, 'media_cache', return_value=(root, meta, np.zeros(48000))):
                result = gander.run_sample(args, NS(model=model), context)
            self.assertEqual(model.tools, [TASK_START])
            self.assertEqual(trace, ['media', 'generate', 'question', 'media', 'generate', 'media', 'generate'])
            self.assertEqual(len(result.events), 1)
            self.assertEqual(result.events[0]['onset_seconds'], 2)
            self.assertEqual(result.events[0]['agent_call_onset_seconds'], 3)
            self.assertNotIn('PREQUESTION', result.events[0]['text'])
            raw = list(root.rglob('*.chunks.jsonl'))
            self.assertEqual(len(raw), 1)
            self.assertEqual(len(raw[0].read_text().splitlines()), 3)

    def test_joyai_http_uses_native_prompt_and_raw_not_normalized_instruction(self):
        from benchmark.adapters import joyai
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = SampleContext(0, SAMPLE, 'Base', root, agent_output_protocol='native', task_text_model='joyai')
            sent = []
            raw = '</response> Saving it. </delegation> Save the observed event.'
            def post(url, **kwargs):
                body = {}
                if not url.endswith('/reset'):
                    sent.append(kwargs['json'])
                    body = {'choices': [{'message': {'content': 'Saving it.'}, 'finish_reason': 'stop'}],
                            'streamingharness': {'raw_content': raw}}
                return NS(raise_for_status=lambda: None, json=lambda: body)
            args = NS(videos_dir=root, cache_dir=root, adapter_url='http://local/v1/chat/completions',
                      model='joyai', agent_output_protocol='native')
            meta = dict(duration=3., frames=['0.jpg', '1.jpg', '2.jpg'], frame_times=[0., 1., 2.])
            with patch.object(joyai, 'media_cache', return_value=(root, meta, None)):
                result = joyai.run_sample(args, NS(requests=NS(post=post)), context)
            self.assertEqual(len(result.events), 2)  # Repeated delegations are not deduplicated.
            self.assertEqual(result.events[0]['onset_seconds'], 2)
            self.assertEqual(result.events[0]['native_raw_text'], raw)
            self.assertEqual(parse_event(result.events[0])['calls'][0]['function'], CALL)
            self.assertIn('</delegation>', sent[0]['messages'][0]['content'])
            self.assertEqual(sum(c['type'] == 'text' for p in sent for c in p['messages'][1]['content']), 1)
            self.assertTrue(all(p['messages'][1]['content'][-1]['type'] == 'image_url' for p in sent))


if __name__ == '__main__':
    unittest.main()
