import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from benchmark.data.task_text import SEPARATOR, TaskTextStream, event, build_prompt
from benchmark.data.agent_inputs import input_protocol
from benchmark.evaluation.agent_calls import parse_event
from benchmark.runtime.runner import SampleContext
from benchmark.cli.main import build_parser
from benchmark.scripts.run_task_text_trial import stratified_indices


class Tokenizer:
    all_special_ids = [1000, 1001, 1002]
    unk_token_id = -1
    def convert_tokens_to_ids(self, token):
        return 1003 if token == '<|turn_eos|>' else -1
    def decode(self, ids, **kwargs):
        return bytes(ids).decode('utf-8', errors='replace')


class TaskTextTests(unittest.TestCase):
    def test_split_separator_first_token_unicode_and_native_controls(self):
        stream = TaskTextStream(Tokenizer())
        self.assertIsNone(stream.feed([1000] + list('看到了。<|task'.encode()), 1))
        self.assertIsNone(stream.feed(list('_start'.encode()) + [1002], 2))
        result = stream.feed(list('|>保存所见事件。'.encode()) + [1003, 1001], 3, ended=True)
        self.assertEqual(result['response_text'], '看到了。')
        self.assertEqual(result['agent_instruction'], '保存所见事件。')
        self.assertEqual(result['onset_seconds'], 1)
        self.assertEqual(result['agent_call_onset_seconds'], 3)
        self.assertEqual(parse_event(result)['calls'][0]['function']['arguments']['instruction'], '保存所见事件。')
        self.assertIsNone(stream.finish())

    def test_bad_formats_and_video_end_do_not_make_calls(self):
        for text, status in [('A', 'absent'), (SEPARATOR+'B', 'empty_field'),
                             ('A'+SEPARATOR, 'empty_field'), ('A'+SEPARATOR+'B'+SEPARATOR+'C', 'duplicate_separator')]:
            result = event(text, 1, complete=True)
            self.assertEqual(result['task_text_status'], status)
            self.assertFalse(result['tool_calls'])
        result = event('A'+SEPARATOR+'B', 1, complete=False)
        self.assertTrue(result['task_text_format_valid'])
        self.assertEqual(result['task_text_status'], 'truncated')
        self.assertFalse(result['tool_calls'])
        self.assertTrue(event('A'+SEPARATOR+'B', 1, complete=None)['tool_calls'])

    def test_prompt_has_no_old_schema_or_reference_leak(self):
        sample = {'mode':'agent', '_agent_call_schema':'task-start', 'available_tools':['note_write'],
                  'frontend_context': {'system_prompt':'OLD JSON FORMAT', 'tools':[{'name':'agent_call'}]},
                  'responses':[{'response':'SECRET LABEL'}]}
        for name in ['minicpmo','gander','joyai']:
            prompt = build_prompt(sample, name)
            self.assertNotIn('OLD JSON FORMAT', prompt)
            self.assertNotIn('SECRET LABEL', prompt)
            self.assertNotIn('Available function schema', prompt)
            self.assertEqual(prompt.count('- note_write:'), 1)
            self.assertNotIn('- web_search:', prompt)
            with tempfile.TemporaryDirectory() as directory:
                context = SampleContext(0,sample,'OLD BASE',Path(directory),agent_output_protocol='task-text',task_text_model=name)
                self.assertEqual(context.build_prompt('OLD NATIVE JSON'), prompt)

    def test_cli_and_old_protocol_identity(self):
        parser = build_parser()
        for name in ['minicpmo','gander','joyai']:
            command=['parallel',name,'--model-path','/models/test','--gpus','0','1','--agent-output-protocol','task-text']
            if name=='gander':command+=['--base-model-path','/base','--official-code','/code']
            args=parser.parse_args(command)
            self.assertIn('task_text_v1',input_protocol(args))
        self.assertEqual(input_protocol(NS(agent_output_protocol='text',agent_call_schema='legacy')),
                         'post_media_generation_without_agent_results_v3')

    def test_gander_disables_native_schema_only_for_trial(self):
        from benchmark.adapters import gander
        context=NS(sample={'mode':'agent','available_tools':['note_write'],'_agent_call_schema':'task-start'},native_protocol='')
        state=NS(model=NS(tools=None))
        with patch('benchmark.adapters.gander.minicpmo.run_sample',return_value=NS(events=[])):
            gander.run_sample(NS(agent_output_protocol='task-text'),state,context)
        self.assertEqual(state.model.tools,[])

    def test_minicpm_uses_raw_ids_even_when_native_text_loses_first_token(self):
        import numpy as np
        from benchmark.adapters import minicpmo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = NS(total_ids=[], tokenizer=Tokenizer(), prepare=Mock(),
                       streaming_prefill=Mock(return_value={'success': True}))
            parts = ['A<|task', '_start|>Save it.']
            def generate(**kwargs):
                part = parts.pop(0)
                model.total_ids.extend([1000] + list(part.encode()))
                return {'text': part[1:], 'is_listen': False, 'end_of_turn': not parts}
            model.streaming_generate = generate
            args = NS(seed=42, videos_dir=root, cache_dir=root, max_seconds=0,
                      agent_output_protocol='task-text', decode_mode='greedy', max_speak_tokens=20)
            context = SampleContext(0, {'mode':'agent','task':'event_agent','video':'v.mp4',
                                      'question':'Save it','question_time':'00:00','available_tools':['note_write']},
                                    'Base',root,agent_output_protocol='task-text')
            meta = {'duration':2.,'frame_times':[],'frames':[]}
            with patch.object(minicpmo,'media_cache',return_value=(root,meta,np.zeros(32000))):
                result = minicpmo.run_sample(args,NS(model=model),context)
            self.assertEqual(result.events[0]['response_text'],'A')
            self.assertEqual(result.events[0]['agent_instruction'],'Save it.')
            self.assertEqual(model.streaming_prefill.call_count,3)

    def test_stratification_is_shared_reproducible_and_capped(self):
        samples=[{'ability':a} for a in ['a']*70+['b']*30]
        selected=stratified_indices(samples,set(range(100)),20,42)
        self.assertEqual(len(selected),20)
        self.assertEqual(sum(i<70 for i in selected),14)
        self.assertEqual(selected,stratified_indices(samples,set(range(100)),20,42))

    def test_joyai_single_line_payload_retains_both_fields(self):
        from benchmark.adapters import joyai
        with tempfile.TemporaryDirectory() as tmp:
            context=SampleContext(0,{'mode':'agent','video':'v.mp4','question':'Save it','question_time':'00:00',
                                    'available_tools':['note_write']},'Base',Path(tmp),agent_output_protocol='task-text',task_text_model='joyai')
            raw='</response> Saving it.'+SEPARATOR+'Save the observed event.'
            sent=[]
            def post(url,**kwargs):
                if url.endswith('/reset'):body={}
                else:
                    sent.append(kwargs['json'])
                    body={'choices':[{'message':{'content':raw},'finish_reason':'stop'}]}
                return NS(raise_for_status=lambda:None,json=lambda:body)
            args=NS(videos_dir=Path(tmp),cache_dir=Path(tmp),adapter_url='http://local/v1/chat/completions',model='joyai',agent_output_protocol='task-text')
            with patch.object(joyai,'media_cache',return_value=(Path(tmp),{'duration':1.,'frames':['f.jpg'],'frame_times':[0.]},None)):
                result=joyai.run_sample(args,NS(requests=NS(post=post)),context)
            self.assertEqual(result.events[0]['agent_instruction'],'Save the observed event.')
            self.assertIn(SEPARATOR,sent[0]['messages'][0]['content'])
            self.assertNotIn('Available function schema',sent[0]['messages'][0]['content'])


if __name__=='__main__':unittest.main()
