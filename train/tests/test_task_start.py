from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.data.task_text import TaskTextStream as AgentStream
from train.src.agent import PROTOCOL
from train.src.checkpoints import adapter_identity, full_model_identity
from benchmark.evaluation.timing import match_windows
from train.src.agent import training_sample, training_record, validate_prepared
from train.src.data import target_chunks, load_config
from train.src.rewards import reward
from train.src.stream import system_prompt
from train.tests.test_agent_token import ByteTokenizer


def source_sample():
    return dict(mode='agent', sample_type='positive', available_tools=['product_search'],
        responses=[dict(role='assistant', response='I will check.', window_start='00:01',window_end='00:03',
            tool_calls=[dict(type='function',function=dict(name='agent_call',arguments=json.dumps({
                'instruction':'search for caps','observed_context':'private reference evidence',
                'user_goal':'find caps','constraints_and_authorization':'read only',
                'suggested_capabilities':['product_search'],'expected_output':'options',
                'extracted_parameters':[{'name':'query','value':'caps'}]})))])])


class TaskStartTrainingTests(unittest.TestCase):
    def test_full_export_records_plain_separator_without_vocab_expansion(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        from train.src.full import finalize_export
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.mkdir();output=root/'export';output.mkdir()
            raw=Tokenizer(WordLevel({'[UNK]':0,'<|':1,'task_start':2,'|>':3},unk_token='[UNK]'))
            raw.pre_tokenizer=Whitespace()
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]')
            model=Qwen2ForCausalLM(Qwen2Config(vocab_size=4,hidden_size=8,intermediate_size=16,
                num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1))
            model.save_pretrained(base);model.save_pretrained(output);tokenizer.save_pretrained(output)
            finalize_export(output,root/'checkpoint',{'model_path':str(base),'agent_output_protocol':PROTOCOL},tokenizer)
            meta=full_model_identity(output)
            self.assertEqual(meta['vocab_size'],4)
            self.assertEqual(meta['separator'],'<|task_start|>')
            self.assertEqual(meta['separator_token_ids'],[1,2,3])
            meta['separator_token_ids']=[0]
            (output/'full_training_model.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):full_model_identity(output)

    def test_adapter_is_pure_idempotent_and_preserves_timing(self):
        source=source_sample();before=deepcopy(source);adapted=training_sample(source)
        self.assertEqual(source,before);self.assertEqual(training_sample(adapted),adapted)
        call=adapted['responses'][0]['tool_calls'][0]['function']
        self.assertEqual(call,dict(name='task_start',arguments={'instruction':'search for caps'}))
        self.assertEqual(match_windows(source,[dict(onset_seconds=2)]),match_windows(adapted,[dict(onset_seconds=2)]))
        rtd=dict(source,mode='rtd');self.assertIs(training_sample(rtd),rtd)

    def test_training_target_and_parser_round_trip(self):
        s=source_sample();r=dict(sample=s,reference=s['responses'][0],response_index=0)
        before=deepcopy(r);t=ByteTokenizer();chunks=target_chunks(r,t,12)
        self.assertEqual(r,before);self.assertEqual(sum(ids.count(t.agent_id) for ids,_ in chunks),0)
        parser=AgentStream(t);event=None
        for clock,(ids,groups) in enumerate(chunks,1):
            event=parser.feed(ids,clock,ended=clock==len(chunks))
        self.assertEqual(event['response_text'],'I will check.')
        self.assertEqual(event['tool_calls'][0]['function'],dict(name='task_start',arguments={'instruction':'search for caps'}))
        row=training_record(r)
        self.assertEqual(row['reference'],row['sample']['responses'][0])
        with self.assertRaisesRegex(ValueError,'requires'):
            target_chunks(r,t,12,protocol='structured')

    def test_invalid_new_calls_and_old_model_output_are_rejected(self):
        t=ByteTokenizer()
        for function in [dict(name='task_start',arguments={'instruction':' '}),
                         dict(name='task_start',arguments={}),
                         dict(name='task_start',arguments={'instruction':'search','extra':'x'}),
                         dict(name='agent_call',arguments={'instruction':'search'})]:
            event=AgentStream(t).feed([t.agent_id]+t.encode(json.dumps(function)),2,ended=True)
            self.assertEqual(event['tool_calls'],[])
        s=training_sample(source_sample());s['responses'][0]['tool_calls'][0]['function']['arguments']['extra']='x'
        with self.assertRaises(ValueError):training_sample(s)

    def test_reward_uses_converted_reference(self):
        s=source_sample();call=dict(name='task_start',arguments={'instruction':'search for caps'})
        event=dict(onset_seconds=2,text='I will check.',response_text='I will check.',tool_calls=[call])
        self.assertEqual(reward(s,[event],{})['call'],1.)
        event['tool_calls']=[dict(call,name='agent_call')]
        self.assertEqual(reward(s,[event],{})['call'],0.)
        event['tool_calls']=[dict(call,arguments={'instruction':'search for caps','extra':'x'})]
        self.assertEqual(reward(s,[event],{})['call'],0.)

    def test_old_prepared_configs_and_exports_are_rejected(self):
        config={'agent_output_protocol':PROTOCOL}
        with self.assertRaisesRegex(ValueError,'prepare again'):validate_prepared({'config':config},config)
        validate_prepared({'protocol':PROTOCOL,'config':config},config)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'config.json').write_text(json.dumps({'agent_output_protocol':'structured'}))
            with self.assertRaisesRegex(ValueError,'requires'):load_config(p/'config.json')
            (p/'agent_protocol.json').write_text(json.dumps({'protocol':'agent_token_v1'}))
            with self.assertRaisesRegex(ValueError,'Unsupported adapter'):adapter_identity(p)
            (p/'full_training_model.json').write_text(json.dumps({'protocol':'agent_token_v1'}))
            with self.assertRaisesRegex(ValueError,'Unsupported full-model'):full_model_identity(p)

    def test_all_training_configs_and_prompt_match_new_targets(self):
        for name in ('minicpmo_lora','minicpmo_full_sft','minicpmo_full_grpo'):
            config=load_config(Path('train/configs')/(name+'.json'))
            self.assertEqual(config['agent_output_protocol'],PROTOCOL)
            s=source_sample();s['frontend_context']={'system_prompt':'Delegate with agent_call.'}
            prompt=system_prompt({'sample':s},config)
            self.assertIn('task_start',prompt);self.assertNotIn('agent_call',prompt)
            self.assertNotIn('private reference evidence',prompt)
            self.assertNotIn('search for caps',prompt)


if __name__=='__main__':unittest.main()
