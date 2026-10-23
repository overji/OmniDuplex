from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from peft import get_peft_model,LoraConfig

from JEV_train.src.gate import CachedDecoder,encode,loss,GateSession
from JEV_train.src.rows import DecisionEmbedding
from JEV_train.src.config import SILENT,RESPONSE
from JEV_train.src.media import MediaUnit


class Tokenizer:
    def encode(self,text,**kw):
        result=[]
        while text:
            token=next((t for t in (SILENT,RESPONSE) if text.startswith(t)),None)
            if token:result.append(self.convert_tokens_to_ids(token));text=text[len(token):]
            else:result.append(20+ord(text[0])%200);text=text[1:]
        return result
    def convert_tokens_to_ids(self,t):return {SILENT:254,RESPONSE:255,'</unit>':2}[t]


def state():
    torch.set_num_threads(2);torch.manual_seed(2)
    llm=Qwen2ForCausalLM(Qwen2Config(vocab_size=256,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1))
    llm.requires_grad_(False)
    llm=get_peft_model(llm,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    embedding=DecisionEmbedding(llm.get_input_embeddings(),[254,255]);llm.set_input_embeddings(embedding)
    tokenizer=Tokenizer();duplex=NS(_reset_streaming_state=lambda:None)
    def prefill(**kwargs):
        ids=tokenizer.encode(' '.join(kwargs['text_list'])) if 'text_list' in kwargs else [3]
        duplex.decoder.feed(duplex.decoder.embed_token(ids));return {'success':True}
    duplex.streaming_prefill=prefill
    return NS(model=NS(llm=llm,device=torch.device('cpu'),init_streaming_processor=lambda:None),
              tokenizer=tokenizer,duplex=duplex,embedding=embedding,ids=[254,255],head=torch.nn.Linear(16,2,bias=False))


class GateTests(unittest.TestCase):
    def test_two_row_loss_trains_lora_and_rows_without_touching_base(self):
        s=state();row=dict(sample_index=1,sample=dict(mode='pr',question='Tell me',question_time='00:00'),
            labels=[0,1,0],train_labels=[0,1,0],events=[dict(second=2,target='A reply')])
        config=dict(max_sequence_tokens=4096)
        with patch('JEV_train.src.media.units',return_value=[MediaUnit(i,frame=object()) for i in (1,2,3)]):
            with torch.no_grad():item=encode(s,row,config)
        before=s.embedding.base.weight.detach().clone();value=loss(s,item);value.backward()
        self.assertGreater(s.embedding.rows.grad.abs().sum(),0)
        self.assertGreater(s.head.weight.grad.abs().sum(),0)
        self.assertGreater(sum(p.grad.abs().sum() for n,p in s.model.llm.named_parameters() if 'lora_' in n and p.grad is not None),0)
        self.assertIsNone(s.embedding.base.weight.grad);torch.testing.assert_close(before,s.embedding.base.weight)
    def test_cached_decoder_matches_full_backbone_and_gate_uses_current_second(self):
        s=state();s.model.llm.eval()
        with torch.inference_mode():
            decoder=CachedDecoder(s,100)
            first=s.embedding(torch.tensor([4,5]));second=s.embedding(torch.tensor([6,7]))
            decoder.feed(first);decoder.feed(second)
            expected=s.model.llm.get_base_model().model(inputs_embeds=torch.cat([first,second]).unsqueeze(0),use_cache=False).last_hidden_state[:,-1]
            torch.testing.assert_close(decoder.hidden,expected)
            gate=GateSession(s,dict(max_sequence_tokens=4096,threshold=.5))
            gate.start(dict(mode='pr',question='Tell me',question_time='00:02'))
            decision,p=gate.decide(MediaUnit(1,frame=object()));self.assertFalse(decision)
            gate.decide(MediaUnit(2,frame=object()))
            gate.decide(MediaUnit(3,frame=object()));self.assertTrue(gate.sent)
            self.assertGreaterEqual(p,0);self.assertLessEqual(p,1)


if __name__=='__main__':unittest.main()
