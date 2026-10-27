"""原子委托token、全参权重与推理解析闭环的CPU测试。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM

from benchmark.data.task_text import TaskTextStream
from train.src.domain.agent import PROTOCOL,RECORD_FORMAT,SEPARATOR,training_sample,training_record,validate_prepared
from train.src.domain.targets import target_chunks
from train.src.infrastructure.tokens import register_task_token,resize_vocabulary,validate_task_token
from train.src.infrastructure.checkpoints import finalize_export,full_model_identity
from train.src.infrastructure.config import load_config
from train.tests.helpers import ByteTokenizer


def tokenizer():
    raw=Tokenizer(WordLevel({'[UNK]':0,'I':1,'check':2,'search':3,'caps':4},unk_token='[UNK]'))
    raw.pre_tokenizer=Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]')


def source_sample():
    return dict(mode='agent',sample_type='positive',available_tools=['product_search'],responses=[dict(
        role='assistant',response='I will check.',window_start='00:01',window_end='00:03',
        tool_calls=[dict(type='function',function=dict(name='agent_call',arguments=json.dumps(dict(
            instruction='search for caps',observed_context='private reference evidence',user_goal='find caps'))))])])


class TaskStartTests(unittest.TestCase):
    def test_registration_reads_vocabulary_a_constant_number_of_times(self):
        class Counted:
            def __init__(self):self.inner=tokenizer();self.calls=0
            def __getattr__(self,key):return getattr(self.inner,key)
            def __len__(self):return len(self.inner)
            def get_vocab(self):self.calls+=1;return self.inner.get_vocab()
        t=Counted();register_task_token(t)
        self.assertLessEqual(t.calls,4)

    def test_atomic_append_mean_rows_gradients_and_export_roundtrip(self):
        torch.manual_seed(3);t=tokenizer();old=t.get_vocab().copy()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';out=root/'export'
            model=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(t),hidden_size=8,intermediate_size=16,
                num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,tie_word_embeddings=False))
            model.save_pretrained(base);t.save_pretrained(base)
            base_tokenizer=(base/'tokenizer.json').read_bytes()
            before_in=model.get_input_embeddings().weight.detach().clone();before_out=model.get_output_embeddings().weight.detach().clone()
            token=register_task_token(t);self.assertEqual(token,len(old));self.assertEqual(t.encode(SEPARATOR),[token])
            self.assertTrue(all(t.get_vocab()[k]==v for k,v in old.items()))
            resize_vocabulary(model,t)
            torch.testing.assert_close(model.get_input_embeddings().weight[:-1],before_in)
            torch.testing.assert_close(model.get_output_embeddings().weight[:-1],before_out)
            torch.testing.assert_close(model.get_input_embeddings().weight[-1],before_in.mean(0))
            torch.testing.assert_close(model.get_output_embeddings().weight[-1],before_out.mean(0))
            optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
            model(input_ids=torch.tensor([[1,token,3,4]]),labels=torch.tensor([[-100,token,3,4]])).loss.backward()
            self.assertGreater(model.get_input_embeddings().weight.grad[token].abs().sum(),0)
            self.assertGreater(model.get_output_embeddings().weight.grad[token].abs().sum(),0)
            optimizer.step();trained=model.get_input_embeddings().weight.detach().clone()
            register_task_token(t);resize_vocabulary(model,t)
            torch.testing.assert_close(model.get_input_embeddings().weight,trained)
            model.save_pretrained(out);t.save_pretrained(out)
            finalize_export(out,root/'checkpoint',dict(model_path=str(base),agent_output_protocol=PROTOCOL),t)
            meta=full_model_identity(out)
            self.assertEqual(meta['separator_token_ids'],[token]);self.assertTrue(meta['separator_is_special'])
            loaded=Qwen2ForCausalLM.from_pretrained(out)
            torch.testing.assert_close(loaded.get_input_embeddings().weight,model.get_input_embeddings().weight)
            torch.testing.assert_close(loaded.get_output_embeddings().weight,model.get_output_embeddings().weight)
            self.assertEqual((base/'tokenizer.json').read_bytes(),base_tokenizer)

    def test_agent_target_roundtrip_preserves_atomic_separator_across_chunks(self):
        s=source_sample();before=deepcopy(s);t=ByteTokenizer()
        row=dict(sample=s,reference=s['responses'][0],response_index=0)
        chunks=target_chunks(row,t,12)
        self.assertEqual(s,before);self.assertEqual(sum(ids.count(300) for ids,_ in chunks),1)
        parser=TaskTextStream(t);event=None
        for clock,(ids,_) in enumerate(chunks,1):event=parser.feed(ids,clock,ended=clock==len(chunks))
        self.assertEqual(event['response_text'],'I will check.');self.assertEqual(event['agent_instruction'],'search for caps')
        self.assertEqual(event['tool_calls'][0]['function']['name'],'task_start')
        adapted=training_record(row);self.assertEqual(adapted['reference'],adapted['sample']['responses'][0])
        self.assertEqual(training_sample(training_sample(s)),training_sample(s))

    def test_duplicate_empty_and_truncated_atomic_calls_remain_invalid(self):
        t=ByteTokenizer()
        for text,ended,status in [('reply'+SEPARATOR+SEPARATOR+'task',True,'duplicate_separator'),
                                  ('reply'+SEPARATOR,True,'empty_field'),('reply'+SEPARATOR+'task',False,'truncated')]:
            parser=TaskTextStream(t);result=parser.feed(t.encode(text),1,ended=ended)
            if not ended:result=parser.finish(complete=False)
            self.assertEqual(result['task_text_status'],status);self.assertFalse(result['tool_calls'])

    def test_old_protocol_and_non_atomic_tokenizers_are_rejected(self):
        config=dict(agent_output_protocol=PROTOCOL)
        with self.assertRaisesRegex(ValueError,'prepare again'):validate_prepared({'protocol':'task_text_sft_v1','config':config},config)
        validate_prepared(dict(protocol=PROTOCOL,record_format=RECORD_FORMAT,config=config),config)
        with self.assertRaisesRegex(ValueError,'registered special token'):validate_task_token(tokenizer())
        cfg=load_config(Path('train/configs/minicpmo_full_sft.json'))
        self.assertEqual(cfg['training_method'],'full_sft');self.assertNotIn('lora_rank',cfg)
        grpo=load_config(Path('train/configs/minicpmo_full_grpo.json'))
        self.assertEqual(grpo['training_method'],'full_grpo');self.assertNotIn('lora_rank',grpo)
        with tempfile.TemporaryDirectory() as directory:
            for method in ('lora_sft',):
                p=Path(directory)/'config.json';p.write_text(json.dumps(dict(cfg,training_method=method)))
                with self.assertRaisesRegex(ValueError,'Unsupported training_method'):load_config(p)


if __name__=='__main__':unittest.main()
