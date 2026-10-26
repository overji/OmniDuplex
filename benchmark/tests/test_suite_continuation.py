import os
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.scripts.resume_after_agent import process_identity, remaining_indices
from benchmark.scripts.run_prepared_suite import build_plan
from benchmark.scripts.prepare_reply_queue import prepare_commands


class SuiteContinuationTests(unittest.TestCase):
    def test_reply_queue_order_protocols_and_gpu_settings(self):
        config=json.loads(Path('benchmark/configs/suites/reply_task_text.json').read_text())
        source=[]
        for name in config['probe_first']+config['probe_last']:
            source.append(dict(mode='probe',adapter=name,gpus=['0','1'],argv=['python','-m','benchmark','probe','run',name]))
        for item in config['ablations']:
            source.append(dict(**item,gpus=['0','1'],argv=['python','-m','benchmark','run',item['adapter']]))
        source.append(dict(mode='realtime',adapter='livestar',argv=['python','-m','benchmark','run','livestar']))
        commands=prepare_commands(source,config,Path('/old'),Path('/agent.json'))
        self.assertEqual([c['adapter'] for c in commands[:3]],config['probe_first'])
        self.assertEqual(commands[7]['adapter'],'livestar')
        self.assertEqual([c['adapter'] for c in commands[8:]],config['probe_last'])
        self.assertEqual([c['sequence'] for c in commands],list(range(13)))
        for entry in commands:
            argv=entry['argv']
            flag='--output-protocol' if entry['mode']=='probe' else '--agent-output-protocol'
            expected='reply-task-text' if entry['mode']=='probe' else 'task-text'
            self.assertEqual(argv[argv.index(flag)+1],expected)
            if entry['mode']=='probe':
                self.assertEqual(argv[argv.index('--prompt-layout')+1],'task-specific')
                self.assertEqual(argv[argv.index('--max-new-tokens')+1],'512')
            else:
                self.assertNotIn('--prompt-layout',argv)
                self.assertNotIn('--max-new-tokens',argv)
        self.assertEqual(commands[0]['gpus'],['0','1'])
        self.assertEqual(commands[7]['smoke_sample_indices'],'/agent.json')
        self.assertIn('--reuse-non-agent',commands[7]['argv'])

    def test_livestar_smoke_uses_agent_subset_and_full_keeps_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry=dict(mode='realtime',adapter='livestar',variant='reply_task_text',
                smoke_sample_indices='/agent.json',argv=['python','-m','benchmark','run','livestar',
                '--agent-output-protocol','task-text','--sample-indices','/all.json'])
            task,=build_plan([entry],['0','1'],Path(tmp))
            full=task['commands'][0];smoke=task['smoke']
            self.assertEqual(full[full.index('--sample-indices')+1],'/all.json')
            self.assertEqual(smoke[smoke.index('--sample-indices')+1],'/agent.json')

    def test_explicit_task_text_protocol_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = [dict(mode='realtime', adapter=name, variant='task_text_agent_full',
                            sequence=i, skip_smoke=True,
                            argv=['python', '-m', 'benchmark', 'run', name,
                                  '--agent-output-protocol', 'task-text', '--sample-indices', '/agent723.json'])
                       for i, name in enumerate(('minicpmo', 'gander', 'joyai'))]
            for task in build_plan(entries, ['0', '1'], Path(tmp)):
                command = task['commands'][0]
                self.assertEqual(command.count('--agent-output-protocol'), 1)
                self.assertEqual(command[command.index('--agent-output-protocol') + 1], 'task-text')
                self.assertEqual(command[command.index('--sample-indices') + 1], '/agent723.json')
                self.assertIsNone(task['smoke'])

    def test_pending_keeps_partial_excludes_agent_and_missing_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            videos = Path(tmp)
            (videos / 'present.mp4').touch()
            samples = [dict(mode=mode, video=video) for mode, video in (
                ('rtd', 'present.mp4'), ('pr', 'present.mp4'),
                ('agent', 'present.mp4'), ('rtd', 'missing.mp4'))]
            records = [dict(sample_index=0, status='complete'), dict(sample_index=1, status='partial')]
            self.assertEqual(remaining_indices(samples, videos, records), [1])

    def test_suite_preserves_explicit_subset_for_full_and_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            entry = dict(mode='realtime', adapter='gander', argv=[
                'python', '-m', 'benchmark', 'run', 'gander', '--sample-indices', '/selected.json'])
            task, = build_plan([entry], ['0', '1'], output)
            for command in (task['commands'][0], task['smoke']):
                self.assertEqual(command.count('--sample-indices'), 1)
                self.assertEqual(command[command.index('--sample-indices') + 1], '/selected.json')

    def test_suite_default_still_uses_available_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = dict(mode='realtime', adapter='minicpmo', argv=['python', '-m', 'benchmark', 'run', 'minicpmo'])
            task, = build_plan([entry], ['0'], Path(tmp))
            command = task['commands'][0]
            self.assertEqual(command[command.index('--sample-indices') + 1], str(Path(tmp) / 'available_sample_indices.json'))

    def test_process_identity(self):
        self.assertIsNotNone(process_identity(os.getpid()))
        self.assertIsNone(process_identity(999999999))


if __name__ == '__main__':
    unittest.main()
