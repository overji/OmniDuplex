from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.cli.main import build_parser
from benchmark.config.paths import CONFIGS
from benchmark.probe.prompts import model_input
from benchmark.probe.runner import prompt_path, run_config
from benchmark.tests.test_probe import probes


class TaskPromptTests(unittest.TestCase):
    def setUp(self):
        self.common=(CONFIGS/'prompts/probe_common.txt').read_text()
        self.row=probes()[0][0]

    def test_non_agent_prompts_and_user_instructions_have_no_agent_format(self):
        for mode in ('rtd','pr'):
            with self.subTest(mode=mode):
                row=deepcopy(self.row)
                row['request'].update(mode=mode,available_tools=['note_write'],
                    frontend_context={'system_prompt':'OLD AGENT SCHEMA'},responses=['SECRET GOLD'])
                before=deepcopy(row)
                probe=model_input(row,Path('p.mp4'),self.common,'reply-task-text','task-specific')
                text=probe.system_prompt+'\n'+probe.user_prompt
                for forbidden in ('<|task_start|>','note_write','backend capabilities','OLD AGENT SCHEMA','SECRET GOLD'):
                    self.assertNotIn(forbidden,text)
                self.assertIn('Example 1',probe.system_prompt)
                self.assertIn('Example 2',probe.system_prompt)
                self.assertIn('<silent>',probe.system_prompt)
                self.assertEqual(row,before)

    def test_agent_examples_only_use_an_available_capability(self):
        row=deepcopy(self.row);row['request'].update(mode='agent',available_tools=['note_write'])
        probe=model_input(row,Path('p.mp4'),self.common,'reply-task-text','task-specific')
        self.assertIn('Example 1',probe.system_prompt)
        self.assertIn('parcel',probe.system_prompt)
        self.assertEqual(probe.system_prompt.count('- note_write:'),1)
        self.assertIn('<|task_start|>',probe.user_prompt)
        row['request']['available_tools']=['web_search']
        probe=model_input(row,Path('p.mp4'),self.common,'reply-task-text','task-specific')
        self.assertNotIn('note_write',probe.system_prompt)
        self.assertNotIn('parcel',probe.system_prompt)
        self.assertEqual(probe.system_prompt.count('- web_search:'),1)

    def test_untrusted_reference_fields_do_not_change_rendering(self):
        row=deepcopy(self.row)
        before=model_input(row,Path('p.mp4'),self.common,'reply-task-text','task-specific')
        row.update(expected='SECRET',side='post',window_start_seconds=999)
        row['request']['responses']=[{'response':'ANOTHER SECRET'}]
        self.assertEqual(before,model_input(row,Path('p.mp4'),self.common,'reply-task-text','task-specific'))

    def test_shared_layout_and_protocol_validation(self):
        a=model_input(self.row,Path('p.mp4'),'Original template','reply-task-text')
        b=model_input(self.row,Path('p.mp4'),'Original template','reply-task-text','shared')
        self.assertEqual(a,b)
        self.assertIn('for Agent also append',a.user_prompt)
        with self.assertRaisesRegex(ValueError,'requires reply-task-text'):
            model_input(self.row,Path('p.mp4'),self.common,'yes-no','task-specific')

    def test_cli_defaults_remain_shared_and_256_new_queue_overrides_explicitly(self):
        parser=build_parser()
        base=['probe','run','minicpmo','--manifest','/manifest','--model-path','/model','--output-dir','/out']
        original=parser.parse_args(base)
        self.assertEqual(original.prompt_layout,'shared')
        self.assertEqual(original.max_new_tokens,256)
        updated=parser.parse_args(base+['--output-protocol','reply-task-text','--prompt-layout','task-specific',
                                       '--max-new-tokens','512'])
        self.assertEqual(prompt_path(updated),CONFIGS/'prompts/probe_common.txt')

    def test_task_file_changes_invalidate_run_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.mkdir();(model/'config.json').write_text('{}')
            manifest=root/'manifest';manifest.write_text('manifest')
            file=root/'task.txt';file.write_text('original')
            args=build_parser().parse_args(['probe','run','minicpmo','--manifest',str(manifest),
                '--model-path',str(model),'--output-dir',str(root/'out'),'--output-protocol','reply-task-text',
                '--prompt-layout','task-specific','--max-new-tokens','512'])
            with patch('benchmark.probe.prompts.task_prompt_files',return_value=[file]):
                before=run_config(args);file.write_text('changed');after=run_config(args)
            self.assertNotEqual(before,after)
            self.assertEqual(before['prompt_layout'],'task-specific')
            self.assertEqual(before['max_new_tokens'],512)


if __name__=='__main__':
    unittest.main()
