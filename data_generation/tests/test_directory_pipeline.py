import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from data_generation.src.cli.main import main
from data_generation.src.cli.pipeline import parse_args
from data_generation.src.cli.options import application_options
from data_generation.src.application.options import PipelineOptions
from data_generation.src.application import pipeline
from data_generation.src.domain.catalog import split_task
from data_generation.src.domain import selection
from data_generation.tests import test_batch


class DirectoryPipelineTests(unittest.TestCase):
    def setup(self, root, count=3, videos=2, provider='qwen'):
        source = root/'videos'
        source.mkdir()
        for index in range(videos):
            folder = source/str(index)
            folder.mkdir()
            (folder/'same.mp4').write_bytes(f'video-{index}'.encode())
        args = parse_args(['--videos-dir', str(source), '--output-dir', str(root/'out'),
                           '--num-attempts', str(count), '--provider', provider,
                           '--api-key', 'private-test-key', '--model', 'test',
                           '--pr-probability', '1', '--agent-probability', '1',
                           '--agent-positive-probability', '1'])
        return application_options(PipelineOptions, args)

    def api(self, fail_agent=False, skip=False):
        native = test_batch.BatchTests().api(['object_perception_pr', 'object_perception_agent'])
        def respond(system, user, **kwargs):
            kwargs['usage_callback']({'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12})
            if fail_agent and 'Task: object_perception_agent' in system:
                raise RuntimeError('request failed with private-test-key')
            if skip and '\nTask: ' in system:
                return json.dumps({'status': 'skip', 'reason': 'no matching event'})
            return native(system, user, **kwargs)
        return respond

    def test_limit_mid_video_and_resume_do_not_call_again(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup(Path(directory), count=1)
            stack, call = test_batch.BatchTests().patches(self.api())
            with stack:
                self.assertEqual(pipeline.run(args), 0)
                self.assertEqual(call.call_count, 3)
                run = next(args.output_dir.iterdir())
                args.resume=True
                args.output_dir=run
                call.reset_mock()
                self.assertEqual(pipeline.run(args), 0)
                call.assert_not_called()
            report=json.loads((run/'coverage.json').read_text())
            self.assertEqual((report['attempted'],report['accepted'],report['stop_reason']), (1,1,'attempt_limit'))
            self.assertEqual(len(json.loads((run/'all_qa.json').read_text())),1)
            usage=json.loads((run/'api_usage.json').read_text())
            self.assertEqual((usage['calls'],usage['total_tokens']), (3,36))
            self.assertEqual(usage['calls_by_stage'], {'caption':1,'route':1,'qa':1})
            folders=[p for p in run.iterdir() if p.is_dir() and p.name!='.state']
            self.assertEqual(len(folders),1)
            self.assertTrue((folders[0]/'caption.json').is_file())
            self.assertTrue((folders[0]/'route.json').is_file())
            self.assertNotIn('private-test-key', (run/'.state/manifest.json').read_text())

    def test_multiple_abilities_budget_statistics_and_resume(self):
        tasks = ['object_perception_pr', 'object_perception_agent',
                 'action_perception_pr', 'action_perception_agent']
        with tempfile.TemporaryDirectory() as directory, patch.object(
                selection, 'PIPELINE_ABILITY_COUNT_WEIGHTS', (0, 1, 0, 0, 0, 0, 0, 0)):
            args = self.setup(Path(directory), count=3, videos=1)
            stack, call = test_batch.BatchTests().patches(test_batch.BatchTests().api(tasks))
            with stack:
                self.assertEqual(pipeline.run(args), 0)
                self.assertEqual(call.call_count, 5)
                run = next(args.output_dir.iterdir())
                report = json.loads((run/'coverage.json').read_text())
                summary = report['videos']['0/same.mp4']
                self.assertEqual(summary['selected_ability_count'], 2)
                self.assertEqual(summary['attempted_ability_count'], 2)
                self.assertEqual(summary['successful_ability_count'], 2)
                self.assertEqual(report['average_successful_abilities'], 2)
                bundle = json.loads((run/'all_qa.json').read_text())
                self.assertEqual(len(bundle), 3)
                first = split_task(bundle[0]['task'])[0]
                second = split_task(bundle[-1]['task'])[0]
                self.assertNotEqual(first, second)
                self.assertEqual(report['ability_counts'][first], 2)
                self.assertEqual(report['ability_counts'][second], 1)
                manifest = (run/'.state/manifest.json').read_text()
                args.resume = True
                args.output_dir = run
                call.reset_mock()
                self.assertEqual(pipeline.run(args), 0)
                call.assert_not_called()
                self.assertEqual((run/'.state/manifest.json').read_text(), manifest)
                resumed = json.loads((run/'coverage.json').read_text())
                self.assertEqual(resumed['videos'], report['videos'])
                self.assertEqual(resumed['ability_counts'], report['ability_counts'])

    def test_balancing_uses_history_and_current_successes(self):
        tasks = ['object_perception_pr', 'object_perception_agent',
                 'action_perception_pr', 'action_perception_agent']
        with tempfile.TemporaryDirectory() as directory, patch.object(
                selection, 'PIPELINE_ABILITY_COUNT_WEIGHTS', (0, 1, 0, 0, 0, 0, 0, 0)):
            args = self.setup(Path(directory), count=20, videos=2)
            stack, call = test_batch.BatchTests().patches(test_batch.BatchTests().api(tasks))
            with stack:
                self.assertEqual(pipeline.run(args), 0)
                first_run = next(args.output_dir.iterdir())
                self.assertEqual(json.loads((first_run/'coverage.json').read_text())['accepted'], 8)
                self.assertEqual(pipeline.run(args), 0)
                run = next(path for path in args.output_dir.iterdir() if path != first_run)
            manifest = json.loads((run/'.state/manifest.json').read_text())
            decisions = list(manifest['selections'].values())
            for index, decision in enumerate(decisions):
                counts = {item['task']: item['count'] for item in decision['ability_draws'][0]['candidates']}
                self.assertEqual(counts, {'object_perception': 4+index*2, 'action_perception': 4+index*2})
            report = json.loads((run/'coverage.json').read_text())
            self.assertEqual(report['ability_counts']['object_perception'], 4)
            self.assertEqual(report['balancing_ability_counts']['object_perception'], 8)

    def test_budget_can_stop_before_all_selected_abilities_are_attempted(self):
        tasks = ['object_perception_pr', 'action_perception_pr']
        with tempfile.TemporaryDirectory() as directory, patch.object(
                selection, 'PIPELINE_ABILITY_COUNT_WEIGHTS', (0, 1, 0, 0, 0, 0, 0, 0)):
            args = self.setup(Path(directory), count=1, videos=1)
            stack, call = test_batch.BatchTests().patches(test_batch.BatchTests().api(tasks))
            with stack:
                self.assertEqual(pipeline.run(args), 0)
                self.assertEqual(call.call_count, 3)
            run = next(args.output_dir.iterdir())
            video = json.loads((run/'coverage.json').read_text())['videos']['0/same.mp4']
            self.assertEqual(video['selected_ability_count'], 2)
            self.assertEqual(video['attempted_ability_count'], 1)
            self.assertEqual(video['successful_ability_count'], 1)

    def test_multiple_same_names_exhaustion_skip_and_failure_budget(self):
        for count,skip,fail,expected_attempts,expected_calls in [(3,False,False,3,7),
                                                               (9,True,False,4,8),
                                                               (2,False,True,2,4)]:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                args=self.setup(Path(directory), count=count)
                stack,call=test_batch.BatchTests().patches(self.api(fail_agent=fail,skip=skip))
                with stack:
                    self.assertEqual(pipeline.run(args), int(fail))
                    self.assertEqual(call.call_count,expected_calls)
                run=next(args.output_dir.iterdir())
                report=json.loads((run/'coverage.json').read_text())
                self.assertEqual(report['attempted'],expected_attempts)
                self.assertEqual(json.loads((run/'api_usage.json').read_text())['calls'],expected_calls)
                if count==9:
                    self.assertEqual(report['stop_reason'],'source_exhausted')
                    self.assertEqual(report['remaining_attempts'],5)
                    self.assertEqual(report['accepted'],0)
                    self.assertEqual(sum(report['ability_counts'].values()), 0)
                    self.assertEqual(report['average_successful_abilities'], 0)
                    self.assertEqual(report['average_attempted_abilities'], 1)
                if fail:
                    self.assertEqual(report['tasks']['object_perception_agent']['failed'],1)
                    self.assertNotIn('private-test-key',(run/'coverage.json').read_text())
                if not fail:
                    self.assertEqual(len([p for p in run.iterdir() if p.is_dir() and p.name!='.state']),2)

    def test_failed_stage_resume_uses_remaining_budget_and_keeps_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.setup(Path(directory),count=3,videos=1)
            stack,call=test_batch.BatchTests().patches(self.api(fail_agent=True))
            with stack:
                self.assertEqual(pipeline.run(args),1)
            run=next(args.output_dir.iterdir())
            args.resume=True
            args.output_dir=run
            stack,call=test_batch.BatchTests().patches(self.api())
            with stack:
                self.assertEqual(pipeline.run(args),0)
                self.assertEqual(call.call_count,1)
            report=json.loads((run/'coverage.json').read_text())
            self.assertEqual((report['attempted'],report['accepted']), (3,2))
            usage=json.loads((run/'api_usage.json').read_text())
            self.assertEqual((usage['calls'],usage['total_tokens']), (5,60))

    def test_interruption_preserves_partial_results_usage_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup(Path(directory), count=3, videos=1)
            native = self.api()
            def interrupt(system, user, **kwargs):
                result = native(system, user, **kwargs)
                if 'Task: object_perception_agent' in system:
                    raise KeyboardInterrupt()
                return result
            stack, call = test_batch.BatchTests().patches(interrupt)
            with stack, self.assertRaises(KeyboardInterrupt):
                pipeline.run(args)
            run = next(args.output_dir.iterdir())
            self.assertEqual(len(json.loads((run/'all_qa.json').read_text())), 1)
            report = json.loads((run/'coverage.json').read_text())
            self.assertEqual(report['attempted'], 2)
            self.assertEqual(report['tasks']['object_perception_agent']['interrupted'], 1)
            self.assertEqual(json.loads((run/'api_usage.json').read_text())['total_tokens'], 48)
            args.resume = True
            args.output_dir = run
            stack, call = test_batch.BatchTests().patches(self.api())
            with stack:
                self.assertEqual(pipeline.run(args), 0)
                self.assertEqual(call.call_count, 1)
            self.assertEqual(json.loads((run/'coverage.json').read_text())['attempted'], 3)
            self.assertEqual(json.loads((run/'api_usage.json').read_text())['total_tokens'], 60)

    def test_changed_resume_configuration_is_rejected_before_api_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup(Path(directory), count=1)
            stack, call = test_batch.BatchTests().patches(self.api())
            with stack:
                self.assertEqual(pipeline.run(args), 0)
                args.resume = True
                args.output_dir = next(args.output_dir.iterdir())
                before = (args.output_dir/'.state/manifest.json').read_text()
                args.num_attempts = 2
                call.reset_mock()
                with self.assertRaisesRegex(ValueError, 'settings or implementation changed'):
                    pipeline.run(args)
                call.assert_not_called()
                self.assertEqual((args.output_dir/'.state/manifest.json').read_text(), before)

    def test_mapping_validation_before_calls_and_mapping_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=self.setup(root,provider='openai',count=1)
            stack,call=test_batch.BatchTests().patches(self.api())
            with stack, self.assertRaisesRegex(ValueError,'URL mapping'):
                pipeline.run(args)
            call.assert_not_called()
            self.assertFalse(args.output_dir.exists())
            urls=root/'urls.json'
            urls.write_text(json.dumps({
                '0/same.mp4':'https://example.invalid/zero.mp4',
                '1/same.mp4':'https://example.invalid/one.mp4'}))
            args.video_url_map=urls
            stack,call=test_batch.BatchTests().patches(self.api())
            with stack:
                self.assertEqual(pipeline.run(args),0)
            self.assertTrue(all(c.kwargs['video_url']=='https://example.invalid/zero.mp4' for c in call.call_args_list))

    def test_cli_default_dispatch_and_all_help_commands(self):
        from unittest.mock import patch
        with patch('data_generation.src.cli.pipeline.main',return_value=0) as command:
            self.assertEqual(main(['--videos-dir','videos','--output-dir','out','--num-attempts','1']),0)
            command.assert_called_once()
        root=Path(__file__).resolve().parents[1]
        for command in ['pipeline','sample','batch','render-prompts','merge']:
            result=subprocess.run([sys.executable,str(root/'generate.py'),command,'--help'],
                                  cwd='/tmp',capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parse_args(['--videos-dir','videos','--output-dir','out','--num-attempts','0'])
