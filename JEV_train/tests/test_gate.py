from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from JEV_train.src.gate import CachedDecoder,encode,loss,GateSession
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


class Mini(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llm=Qwen2ForCausalLM(Qwen2Config(vocab_size=256,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1))
        self.vpm=torch.nn.Linear(3,16);self.apm=torch.nn.Linear(3,16)
        self.resampler=torch.nn.Linear(16,16);self.audio_projection_layer=torch.nn.Linear(16,16)
    @property
    def device(self):return next(self.parameters()).device
    def init_streaming_processor(self):self.audio_past_key_values=None


class Duplex:
    def __init__(self,model,tokenizer):self.model=model;self.tokenizer=tokenizer;self.decoder=NS()
    def _reset_streaming_state(self):pass
    @torch.no_grad()
    def streaming_prefill(self,text_list=None,**kwargs):
        if text_list:self.decoder.feed(self.decoder.embed_token(self.tokenizer.encode(' '.join(text_list))))
        else:
            self.decoder.feed(self.model.resampler(self.model.vpm(torch.ones(1,3))))
            self.decoder.feed(self.model.audio_projection_layer(self.model.apm(torch.ones(1,3))))
        return {'success':True}


def state():
    torch.set_num_threads(2);torch.manual_seed(2)
    model=Mini();model.llm.get_output_embeddings().requires_grad_(False)
    tokenizer=Tokenizer();duplex=Duplex(model,tokenizer)
    return NS(model=model,tokenizer=tokenizer,duplex=duplex,embedding=model.llm.get_input_embeddings(),
              ids=[254,255],head=torch.nn.Linear(16,2,bias=False))


class GateTests(unittest.TestCase):
    def test_bfloat16_head_accepts_full_backbone_hidden_states(self):
        from JEV_train.src.graphs import GateGraph
        s=state();s.head.to(torch.bfloat16)
        row=dict(sample_index=1,sample=dict(mode='pr',question='Tell me',question_time='00:00'),
                 labels=[0,1],train_labels=[0,1],events=[])
        with patch('JEV_train.src.media.units',return_value=[MediaUnit(i,frame=object()) for i in (1,2)]):
            value=GateGraph(s,dict(max_sequence_tokens=4096))(row)
        self.assertEqual(value.dtype,torch.float32);value.backward()
        self.assertGreater(s.head.weight.grad.abs().sum(),0)

    def test_full_classifier_recomputes_media_and_trains_backbone_encoders_and_head(self):
        s=state();row=dict(sample_index=1,sample=dict(mode='pr',question='Tell me',question_time='00:00'),
            labels=[0,1,0],train_labels=[0,1,0],events=[dict(second=2,target='A reply')])
        config=dict(max_sequence_tokens=4096);decoder=s.duplex.decoder
        with patch('JEV_train.src.media.units',side_effect=lambda *a:[MediaUnit(i,frame=object(),audio=object()) for i in (1,2,3)]):
            item=encode(s,row,config);loss(s,item).backward()
            self.assertIs(s.duplex.decoder,decoder)
            for module in ('llm','vpm','apm','resampler','audio_projection_layer'):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(s.model,module).parameters()),module)
            self.assertGreater(s.head.weight.grad.abs().sum(),0)
            self.assertGreater(s.embedding.weight.grad[254:256].abs().sum(),0)
            self.assertIsNone(s.model.llm.lm_head.weight.grad)
            # 一次更新后再次前向重新编码，没有detach后复用旧embedding。
            torch.optim.AdamW([p for p in s.model.parameters() if p.requires_grad],lr=.01).step()
            newer=encode(s,row,config)
            self.assertFalse(torch.equal(item['embeds'].detach(),newer['embeds'].detach()))

    def test_cached_decoder_matches_full_backbone_and_gate_uses_current_second(self):
        s=state();s.model.eval()
        with torch.inference_mode():
            decoder=CachedDecoder(s,100)
            first=s.embedding(torch.tensor([4,5]));second=s.embedding(torch.tensor([6,7]))
            decoder.feed(first);decoder.feed(second)
            expected=s.model.llm.model(inputs_embeds=torch.cat([first,second]).unsqueeze(0),use_cache=False).last_hidden_state[:,-1]
            torch.testing.assert_close(decoder.hidden,expected)
            gate=GateSession(s,dict(max_sequence_tokens=4096,threshold=.5))
            gate.start(dict(mode='pr',question='Tell me',question_time='00:02'))
            decision,p=gate.decide(MediaUnit(1,frame=object()));self.assertFalse(decision)
            gate.decide(MediaUnit(2,frame=object()));gate.decide(MediaUnit(3,frame=object()))
            self.assertTrue(gate.sent);self.assertGreaterEqual(p,0);self.assertLessEqual(p,1)


if __name__=='__main__':unittest.main()
