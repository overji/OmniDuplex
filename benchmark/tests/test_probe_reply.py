import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from benchmark.cli.main import build_parser
from benchmark.probe.output import PROTOCOL, GeneratedText, decoded_output, parse_reply
from benchmark.probe import metrics, runner
from benchmark.probe.prompts import model_input
from benchmark.tests.test_probe import probes, prediction
from benchmark.adapters.livestar import narration_events, reuse_predictions, NATIVE_PROMPT
from benchmark.data.prompts import build_system_prompt


class ReplyProbeTests(unittest.TestCase):
    def test_decision_is_independent_of_agent_call_validity(self):
        for text, decision, valid in [('<silent>', 'NO', True),
                ('I will check.<|task_start|>Search for caps.', 'YES', True),
                ('I will check.', 'YES', False),
                ('I will check.<|task_start|>', 'YES', False),
                ('<|task_start|>Search.', 'YES', False),
                ('Reply<|task_start|>A<|task_start|>B', 'YES', False),
                ('<silent> Reply', None, False), ('', None, False), ('NO', None, False),
                ('YES: old explanation', None, False)]:
            with self.subTest(text=text):
                result = parse_reply(text, 'agent')
                self.assertEqual(result['decision'], decision)
                self.assertEqual(result['output_format_valid'], valid)
        self.assertTrue(parse_reply('A red cup appears.', 'pr')['output_format_valid'])
        self.assertFalse(parse_reply('A<|task_start|>B', 'rtd')['output_format_valid'])
        truncated = parse_reply('A<|task_start|>B', 'agent', complete=False)
        self.assertEqual(truncated['decision'], 'YES')
        self.assertEqual(truncated['output_status'], 'truncated')
        self.assertFalse(truncated['tool_calls'])

    def test_generated_ids_provide_stop_information(self):
        class Tokens:
            def __init__(self, ids): self.ids = ids
            def __getitem__(self, key): return self
            def tolist(self): return self.ids
        tokenizer = NS(eos_token_id=9)
        self.assertTrue(decoded_output('A', Tokens([1, 9]), 0, tokenizer, 2).generation_complete)
        self.assertFalse(decoded_output('A', Tokens([1, 2]), 0, tokenizer, 2).generation_complete)
        self.assertIsNone(decoded_output('A', Tokens([1]), 0, tokenizer, 2).generation_complete)

    def test_prompt_does_not_leak_old_schema_or_reference(self):
        rows, _ = probes()
        row = rows[0]
        row['request'].update(mode='agent', available_tools=['note_write'],
            frontend_context={'system_prompt': 'OLD agent_call JSON'}, responses=['SECRET'])
        result = model_input(row, Path('p.mp4'), 'NEW CONTRACT', 'reply-task-text')
        self.assertNotIn('OLD', result.system_prompt)
        self.assertNotIn('SECRET', result.system_prompt)
        self.assertEqual(result.system_prompt.count('- note_write:'), 1)
        self.assertNotIn('- web_search:', result.system_prompt)
        self.assertIn('<silent>', result.user_prompt)
        changed = {**row, 'expected': 'CHANGED', 'side': 'post', 'window_start_seconds': 999}
        self.assertEqual(result, model_input(changed, Path('p.mp4'), 'NEW CONTRACT', 'reply-task-text'))

    def test_format_failure_does_not_change_binary_accuracy_and_no_mixed_protocols(self):
        rows, _ = probes()
        for row in rows: row['mode'] = 'agent'
        records = [dict(prediction(rows[0], '<silent>'), output_protocol=PROTOCOL),
                   dict(prediction(rows[1], 'I will check.'), output_protocol=PROTOCOL)]
        score = metrics.summarize(rows, records)['groups']['overall']
        self.assertEqual(score['accuracy'], 1)
        self.assertEqual(score['paired_accuracy'], 1)
        self.assertEqual(score['output_format_valid_rate'], .5)
        self.assertEqual(score['agent_valid_calls'], 0)
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            metrics.summarize(rows, [records[0], prediction(rows[1], 'YES')])

    def test_runner_saves_payload_and_resumes_without_regeneration(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, _ = probes(); rows = rows[:2]
            for row in rows:
                row['mode'] = row['request']['mode'] = 'agent'
                row['request']['available_tools'] = ['note_write']
            manifest = root/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            model = root/'model'; model.mkdir(); (model/'config.json').write_text('{}')
            args = parser.parse_args(['probe','run','qwen25_omni','--manifest',str(manifest),
                '--model-path',str(model),'--output-dir',str(root/'out'),
                '--output-protocol','reply-task-text','--system-prompt',
                'benchmark/configs/prompts/probe_reply_task_text.txt'])
            with patch('benchmark.probe.models.load'), patch.object(runner,'prefix_clip',return_value=root/'p.mp4'), \
                    patch('benchmark.probe.models.generate',side_effect=[GeneratedText('<silent>',complete=True),
                        GeneratedText('Saving it.<|task_start|>Save the observed event.',complete=True)]) as generate:
                self.assertEqual(runner.run(args),0)
                self.assertEqual(runner.run(args),0)
                self.assertEqual(generate.call_count,2)
            records = [json.loads(line) for line in (root/'out/probe_predictions.jsonl').read_text().splitlines()]
            self.assertEqual(records[1]['response_text'],'Saving it.')
            self.assertEqual(records[1]['tool_calls'][0]['function']['arguments']['instruction'],'Save the observed event.')
            self.assertEqual(records[0]['decision'],'NO')
            args.output_protocol = 'yes-no'
            with self.assertRaisesRegex(ValueError,'configuration changed'): runner.run(args)


class LiveStarTextTests(unittest.TestCase):
    def test_native_clock_complete_and_truncated_narrations(self):
        result = narration_events([(0,'Before'),(3,'Reply<|task_start|>Search.'),
            (4,'Reply<|task_start|>Partial')], 1, 5., task_text=True, completions={3:True,4:False})
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['onset_seconds'],4.)
        self.assertEqual(result[0]['response_text'],'Reply')
        self.assertEqual(result[0]['tool_calls'][0]['function']['arguments']['instruction'],'Search.')
        self.assertFalse(result[1]['tool_calls'])
        self.assertEqual(narration_events([(3,'Reply')],1,5.),[dict(onset_seconds=4.,text='Reply')])

    def test_reuse_rejects_changed_settings_and_never_imports_agent(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);worker=root/'worker_0';worker.mkdir()
            args=NS(reuse_non_agent=root,model_path=Path('/model'),videos_dir=Path('/videos'),
                    official_code=Path('/native'),agent_call_schema='task-start')
            config=dict(data_sha256='data',model_path='/model',agent_call_schema='task-start',
                        sample_fps=1.,agent_input_protocol='new')
            plan=dict(model='livestar',data_sha256='data',prompt_sha256=hashlib.sha256(b'base').hexdigest(),
                      arguments=dict(model_path='/model',videos_dir='/videos',official_code='/native',agent_call_schema='task-start'))
            (root/'parallel_plan.json').write_text(json.dumps(plan))
            (worker/'run_config.json').write_text(json.dumps({**config,'agent_input_protocol':'old'}))
            samples=[dict(mode=mode,task=mode,video='v.mp4',question='Notify',available_tools=[]) for mode in ['pr','agent']]
            records=[dict(sample_index=i,status='complete',task=s['task'],video=s['video'],question=s['question'],
                system_prompt_sha256=hashlib.sha256(build_system_prompt('base',s,NATIVE_PROMPT).encode()).hexdigest())
                for i,s in enumerate(samples)]
            (worker/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            reused=reuse_predictions(args,samples,[0,1],config,'base')
            self.assertEqual([r['sample_index'] for r in reused],[0])
            self.assertIn('reuse_source',reused[0])
            with self.assertRaisesRegex(ValueError,'sampling configuration'):
                reuse_predictions(args,samples,[0,1],{**config,'sample_fps':2.},'base')


if __name__ == '__main__':
    unittest.main()
