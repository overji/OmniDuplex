import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark.data.agent_windows import execute, migrate
from benchmark.evaluation.timing import match_windows


class AgentWindowMigrationTests(unittest.TestCase):
    def samples(self):
        return [{'mode': 'agent', 'sample_type': 'positive', 'question_time': '00:00',
                 'generation_context': {'duration_seconds': 20}, 'responses': [
                    {'role': 'assistant', 'trigger_time': '00:01', 'window_start': '00:01',
                     'window_end': '00:01', 'tool_calls': [{}], 'response': 'call'},
                    {'role': 'tool', 'trigger_time': '00:05', 'response': 'result'},
                    {'role': 'assistant', 'trigger_time': '00:05', 'outcome': 'success', 'response': 'final'},
                 ]},
                {'mode': 'agent', 'sample_type': 'missing_tool', 'question_time': '00:02',
                 'generation_context': {'duration_seconds': 20}, 'responses': [
                    {'trigger_time': '00:02', 'window_start': '00:02', 'window_end': '00:02', 'response': 'no'}]},
                {'mode': 'pr', 'responses': [{'trigger_time': '00:01'}]}]

    def test_migration_is_pure_idempotent_and_preserves_inputs(self):
        samples = self.samples(); original = copy.deepcopy(samples)
        updated, changes = migrate(samples)
        self.assertEqual(samples, original)
        self.assertEqual(len(changes), 3)
        self.assertEqual(updated[0]['responses'][0]['window_end'], '00:04')
        self.assertEqual(updated[0]['responses'][1], original[0]['responses'][1])
        final = updated[0]['responses'][2]
        self.assertEqual((final['trigger_time'], final['window_start'], final['window_end'],
                          final['agent_result_time']), ('00:06', '00:06', '00:08', '00:05'))
        self.assertEqual(updated[1]['responses'][0]['window_end'], '00:05')
        self.assertEqual(updated[2], original[2])
        self.assertEqual(migrate(updated), (updated, []))
        bad = self.samples(); bad[0]['generation_context']['duration_seconds'] = 7.9
        with self.assertRaises(ValueError): migrate(bad)

    def test_interleaved_completion_windows_remain_in_data_but_are_not_scored(self):
        sample = {'mode': 'agent', 'responses': [
            {'role': 'tool', 'trigger_time': '00:05'},
            {'role': 'tool', 'trigger_time': '00:07'},
            {'role': 'assistant', 'trigger_time': '00:08', 'agent_result_time': '00:05',
             'window_start': '00:06', 'window_end': '00:08'},
        ]}
        self.assertEqual(match_windows(sample, [{'onset_seconds': 6}])['hits'], 0)
        self.assertEqual(match_windows(sample, [{'onset_seconds': 6}])['windows'], 0)
        self.assertEqual(match_windows(sample, [{'onset_seconds': 5}])['hits'], 0)

    def test_backups_manifest_dry_run_and_rollback(self):
        from benchmark.data import agent_windows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / 'all_qa.json'; manifest = root / 'manifest.json'
            original = json.dumps(self.samples()).encode(); data.write_bytes(original)
            manifest.write_text(json.dumps({'sha256': hashlib.sha256(original).hexdigest(), 'source': 'original'}))
            original_manifest = manifest.read_bytes()
            args = SimpleNamespace(data=data, manifest=manifest, backup_dir=root/'backup', dry_run=True)
            execute(args)
            self.assertEqual(data.read_bytes(), original)
            self.assertFalse(args.backup_dir.exists())
            args.dry_run = False
            execute(args)
            self.assertEqual((args.backup_dir/'all_qa.original.json').read_bytes(), original)
            self.assertEqual((args.backup_dir/'data_manifest.original.json').read_bytes(), original_manifest)
            self.assertEqual(json.loads(manifest.read_text())['sha256'], hashlib.sha256(data.read_bytes()).hexdigest())
            migrated = data.read_bytes(); execute(args)
            self.assertEqual(data.read_bytes(), migrated)
            data.write_bytes(original); manifest.write_bytes(original_manifest)
            args.backup_dir = root/'failed_backup'
            native_write = agent_windows.write_json
            def fail_manifest(path, value):
                if path == manifest: raise OSError('write failed')
                return native_write(path, value)
            with patch.object(agent_windows, 'write_json', side_effect=fail_manifest), self.assertRaises(OSError):
                execute(args)
            self.assertEqual(data.read_bytes(), original)
            self.assertEqual(manifest.read_bytes(), original_manifest)
