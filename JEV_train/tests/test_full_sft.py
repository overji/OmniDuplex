"""全参训练的尾批平均及完整分类器导出/重载；ZeRO合并以CPU state dict替身验证。"""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM

from JEV_train.src.config import PROTOCOL
from train.src.infrastructure.tokens import register_task_token,resize_vocabulary


class FullSFTTests(unittest.TestCase):
    def test_accumulation_tail_matches_true_batch_average(self):
        from JEV_train.src.training import optimize
        initial=torch.tensor(1.5)
        class Engine:
            def __init__(self):
                self.weight=torch.nn.Parameter(initial.clone());self.optimizer=torch.optim.SGD([self.weight],lr=.1)
                self.boundary=False;self.norm=None
            def __call__(self,row):return (self.weight-row['target'])**2
            def set_gradient_accumulation_boundary(self,value):self.boundary=value
            def backward(self,loss):(loss/4).backward()
            def step(self):
                if self.boundary:
                    self.norm=self.weight.grad.abs().item();self.optimizer.step();self.optimizer.zero_grad()
            def get_global_grad_norm(self):return self.norm
        rows=[dict(sample_index=i,target=float(i+1)) for i in range(5)]
        order=list(rows);random.Random(42).shuffle(order)
        expected=torch.nn.Parameter(initial.clone());optimizer=torch.optim.SGD([expected],lr=.1)
        for start in range(0,5,4):
            optimizer.zero_grad();batch=order[start:start+4]
            sum((expected-r['target'])**2 for r in batch).div(len(batch)).backward();optimizer.step()
        engine=Engine()
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);optimize(rows,engine,dict(seed=42,epochs=1,gradient_accumulation_steps=4,rank=0),output)
            history=json.loads((output/'metrics.json').read_text())
            self.assertEqual([h['logical_batch'] for h in history],[4,1])
        torch.testing.assert_close(engine.weight,expected)

    def test_full_gate_export_retains_token_rows_head_and_native_weights(self):
        from JEV_train.src.checkpoints import export_checkpoint,validate_gate_export
        from JEV_train.src.gate import load
        from JEV_train.src.config import digest
        class Mini(torch.nn.Module):
            def __init__(self,llm):super().__init__();self.llm=llm;self.config=llm.config
            @property
            def device(self):return self.llm.device
            def save_pretrained(self,path,**kwargs):self.llm.save_pretrained(path,**kwargs)
            def prepare_processor(self,tokenizer):self.processor=SimpleNamespace(tokenizer=tokenizer)
            def as_duplex(self,**kwargs):return SimpleNamespace(tokenizer=self.processor.tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';checkpoint=root/'checkpoint';output=root/'export';checkpoint.mkdir()
            t=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0,'<|listen|>':1,'<|speak|>':2,'</unit>':3,'text':4},unk_token='[UNK]')),unk_token='[UNK]')
            original=Mini(Qwen2ForCausalLM(Qwen2Config(vocab_size=5,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1)))
            original.save_pretrained(base);t.save_pretrained(base)
            trained=copy.deepcopy(original);register_task_token(t)
            t.add_special_tokens({'additional_special_tokens':['<|silent|>','<|response|>']},replace_additional_special_tokens=False)
            resize_vocabulary(trained,t)
            with torch.no_grad():trained.llm.get_input_embeddings().weight[-3:].fill_(.37)
            head=torch.randn(2,8)
            state={**{'model.'+k:v.detach().clone() for k,v in trained.state_dict().items()},'head.weight':head}
            config=dict(protocol=PROTOCOL,training_method='full_sft',jev_stage='classifier',classifier_model=str(base),cache_dir=str(root/'cache'))
            (checkpoint/'training_config.json').write_text(json.dumps(config));t.save_pretrained(checkpoint/'tokenizer')
            with patch('train.src.infrastructure.checkpoints.read_zero_state',return_value=state),patch('transformers.AutoModel.from_pretrained',return_value=copy.deepcopy(original)):
                export_checkpoint(checkpoint,output)
            meta=validate_gate_export(output,config)
            self.assertEqual(meta['task_start_id'],5);self.assertEqual(meta['token_ids'],[6,7])
            reloaded=Mini(Qwen2ForCausalLM.from_pretrained(output))
            with patch('transformers.AutoModel.from_pretrained',return_value=reloaded):
                loaded=load(dict(config,classifier_checkpoint=str(output),classifier_device='cpu',dtype='float32',attention_backend='eager'))
            torch.testing.assert_close(loaded.model.llm.get_input_embeddings().weight[-3:],torch.full((3,8),.37))
            torch.testing.assert_close(loaded.head.weight,head)
            self.assertEqual(digest(loaded.tokenizer.get_vocab()),meta['tokenizer_sha256'])
            self.assertTrue(all(not p.requires_grad for p in loaded.model.parameters()))


if __name__=='__main__':unittest.main()
