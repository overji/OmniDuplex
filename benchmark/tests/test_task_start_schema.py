from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from benchmark.data.agent_schema import adapt_sample, run_schema, validate_task_start
from benchmark.data.agent_inputs import input_protocol
from benchmark.data.prompts import available_function_schemas
from benchmark.evaluation.agent_calls import parse_event
from benchmark.evaluation.judge import prepare_rows
from benchmark.evaluation.timing import match_windows
from benchmark.runtime.runner import SampleContext


def sample():
    return dict(mode='agent', task='test_agent', ability='test', sample_type='positive',
        video='v.mp4', question='Search when the cap is removed', question_time='00:00',
        available_tools=['product_search'], frontend_context={
            'system_prompt': 'Delegate through agent_call.', 'tools': []},
        responses=[dict(role='assistant', response='Searching.', window_start='00:03', window_end='00:06',
            tool_calls=[dict(type='function', function=dict(name='agent_call', arguments=json.dumps({
                'instruction': 'Search for replacement duduk caps', 'observed_context': 'PRIVATE REFERENCE'})))])])


class TaskStartSchemaTests(unittest.TestCase):
    def test_conversion_is_pure_and_keeps_windows(self):
        original=sample();before=deepcopy(original);adapted=adapt_sample(original,'task-start')
        self.assertEqual(original,before)
        call=adapted['responses'][0]['tool_calls'][0]
        self.assertEqual(call['function']['name'],'task_start')
        self.assertEqual(json.loads(call['function']['arguments']), {'instruction':'Search for replacement duduk caps'})
        events=[dict(onset_seconds=4.,text='Search')]
        self.assertEqual(match_windows(original,events),match_windows(adapted,events))
        self.assertIs(adapt_sample(original,'legacy'),original)
        rtd=dict(original,mode='rtd');self.assertIs(adapt_sample(rtd,'task-start'),rtd)

    def test_prompt_exposes_only_schema_not_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapted=adapt_sample(sample(),'task-start')
            model_sample={k:v for k,v in adapted.items() if k!='responses'}
            prompt=SampleContext(0,model_sample,'Base',Path(tmp),agent_output_protocol='structured').build_prompt()
            self.assertNotIn('PRIVATE REFERENCE',prompt)
            self.assertNotIn('Search for replacement duduk caps',prompt)
            self.assertNotIn('agent_call',prompt)
            self.assertNotIn('extracted_parameters',prompt)
            schemas=available_function_schemas(model_sample)
            self.assertEqual([s['function']['name'] for s in schemas],['task_start'])
            self.assertEqual(schemas[0]['function']['parameters']['required'],['instruction'])
            adapted['available_tools']=[]
            self.assertEqual(available_function_schemas(adapted),[])

    def test_schema_rejects_missing_extra_empty_wrong_name(self):
        good=dict(name='task_start',arguments={'instruction':'Search for caps'})
        self.assertTrue(validate_task_start(good))
        for bad in [dict(good,arguments={}),dict(good,arguments={'instruction':' '}),
                    dict(good,arguments={'instruction':'Search','name':'caps'}),
                    dict(good,name='agent_call')]:
            self.assertFalse(validate_task_start(bad))

    def test_envelopes_and_truncation(self):
        call=dict(name='task_start',arguments={'instruction':'Search'})
        for text in [json.dumps(call),json.dumps({'function':call}),
                     '<tool_call>'+json.dumps(call)+'</tool_call>',
                     json.dumps({'response':'Searching','tool_calls':[{'function':call}]})]:
            parsed=parse_event({'text':text});self.assertEqual(parsed['calls'][0]['function'],call)
        self.assertEqual(parse_event({'text':'{"name":"task_start",'})['status'],'malformed')
        self.assertEqual(parse_event({'text':'I will start a task'})['calls'],[])

    def test_judge_sees_simplified_reference_and_rejects_invalid_call(self):
        s=adapt_sample(sample(),'task-start')
        event=dict(onset_seconds=4.,text='',tool_calls=[dict(name='task_start',arguments={'instruction':'Search for caps'})])
        p=dict(sample_index=0,status='complete',task=s['task'],video=s['video'],question=s['question'],events=[event])
        rows,_=prepare_rows([s],[p]);payload=rows[0]['payload']
        self.assertTrue(payload['judge_call'])
        self.assertEqual(payload['reference']['tool_calls'][0]['function']['name'],'task_start')
        event['tool_calls'][0]['arguments']['extra']='wrong'
        rows,_=prepare_rows([s],[p]);self.assertFalse(rows[0]['payload']['judge_call'])

    def test_cross_chunk_and_tool_only_are_kept_after_media(self):
        from benchmark.tests.test_agent_protocols import AgentProtocolTests
        call=dict(name='task_start',arguments={'instruction':'Search for caps'})
        text=json.dumps({'response':'Searching','tool_calls':[{'function':call}]})
        middle=len(text)//2
        events=AgentProtocolTests.run_stream(self,[
            dict(text=text[:middle],is_listen=False,end_of_turn=False),
            dict(text=text[middle:],is_listen=False,end_of_turn=True)])
        self.assertEqual(events[0]['onset_seconds'],1.)
        self.assertEqual(parse_event(events[0])['calls'][0]['function'],call)
        events=AgentProtocolTests.run_stream(self,[dict(text='',is_listen=False,end_of_turn=False,
                                                       is_tool_call=True,tool_calls=[call])])
        self.assertEqual(events[0]['onset_seconds'],1.)
        self.assertEqual(parse_event(events[0])['calls'][0]['function'],call)

    def test_identity_and_old_results(self):
        legacy=NS(agent_output_protocol='structured',agent_call_schema='legacy')
        simple=NS(agent_output_protocol='structured',agent_call_schema='task-start')
        self.assertNotEqual(input_protocol(legacy),input_protocol(simple))
        with self.assertRaises(ValueError):
            input_protocol(NS(agent_output_protocol='agent-token',agent_call_schema='task-start'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.assertEqual(run_schema(root),'legacy')
            (root/'parallel_plan.json').write_text(json.dumps({'arguments':{'agent_call_schema':'task-start'}}))
            self.assertEqual(run_schema(root),'task-start')


if __name__=='__main__':unittest.main()
