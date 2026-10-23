"""CPU native Thinker cache checks; media encoders are deliberately mocked."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

import torch
from transformers import Qwen2_5OmniThinkerConfig,Qwen2_5OmniThinkerForConditionalGeneration

from JEV_train.src.backend import Qwen25Session,merge_packets
from JEV_train.src.media import MediaUnit


class Tokenizer:
    specials={'<|im_start|>':1,'<|im_end|>':2,'<|IMAGE|>':10,'<|VIDEO|>':11,'<|AUDIO|>':12,
              '<|audio_bos|>':13,'<|audio_eos|>':14,'<|vision_bos|>':15,'<|vision_eos|>':16}
    def get_vocab(self):return self.specials
    def convert_tokens_to_ids(self,t):return self.specials[t]
    def encode(self,text,**kw):
        ids=[]
        while text:
            found=next((s for s in self.specials if text.startswith(s)),None)
            if found:ids.append(self.specials[found]);text=text[len(found):]
            else:ids.append(30+ord(text[0])%150);text=text[1:]
        return ids
    def decode(self,ids,**kw):return 'reply'


class Processor:
    def __init__(self):self.tokenizer=Tokenizer()
    def apply_chat_template(self,*args,**kw):return '<|im_start|>system\nBrief.<|im_end|>\n'
    def __call__(self,text,**kw):
        if 'audio' in kw:text=text.replace('<|AUDIO|>','<|AUDIO|><|AUDIO|>')
        packet={'input_ids':torch.tensor([self.tokenizer.encode(text)])}
        if 'images' in kw:
            packet.update(pixel_values=torch.ones(1,4),image_grid_thw=torch.tensor([[1,1,1]]))
        if 'audio' in kw:
            packet.update(input_features=torch.ones(1,4,8),feature_attention_mask=torch.ones(1,8,dtype=torch.long))
        return packet


def session():
    torch.manual_seed(1);torch.set_num_threads(2)
    config=Qwen2_5OmniThinkerConfig(
        text_config=dict(vocab_size=256,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096,
            rope_scaling={'rope_type':'default','mrope_section':[1,1,2]}),
        audio_config=dict(d_model=16,encoder_layers=1,encoder_attention_heads=2,encoder_ffn_dim=32,
            num_mel_bins=4,output_dim=32,max_source_positions=100,n_window=10),
        vision_config=dict(depth=1,hidden_size=16,intermediate_size=32,num_heads=2,out_hidden_size=32,
            patch_size=2,spatial_merge_size=1,temporal_patch_size=1,fullatt_block_indexes=[0]),
        audio_token_index=12,image_token_index=10,video_token_index=11,audio_start_token_id=13,
        audio_end_token_id=14,vision_start_token_id=15)
    model=Qwen2_5OmniThinkerForConditionalGeneration(config).eval();model.requires_grad_(False)
    # These mocks let the actual multimodal placeholder, MRoPE, attention and KV code execute on CPU.
    model.get_image_features=Mock(side_effect=lambda pixels,grid:torch.ones(len(grid),32)*.2)
    model.get_audio_features=Mock(side_effect=lambda features,**kw:torch.ones(len(features)*2,32)*.3)
    cfg=dict(max_sequence_tokens=4096,max_pixels=4096,max_new_tokens=2,cache_atol=1e-5,cache_rtol=1e-5)
    return Qwen25Session(model,Processor(),cfg)


class QwenCacheTests(unittest.TestCase):
    def test_backend_lora_trains_only_text_layers_and_has_real_gradients(self):
        from JEV_train.src.backend import load
        from JEV_train.src.config import load as load_config,ROOT
        config=load_config(ROOT/'JEV_train/configs/default.json')
        config.update(mode='separate_sft',backend_device='cpu',dtype='float32',lora_rank=2,lora_alpha=4)
        tiny=session()
        with patch('transformers.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained',return_value=tiny.model), \
             patch('transformers.Qwen2_5OmniProcessor.from_pretrained',return_value=tiny.processor):
            s=load(config,training=True)
        trainable=[n for n,p in s.model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable);self.assertTrue(all('.model.layers.' in n and 'lora_' in n for n in trainable))
        s.start_session(dict(mode='pr',question='Tell me',question_time='00:00'),execute=False)
        s.append_media(MediaUnit(1,frame=object(),audio=object()),execute=False)
        s.begin_reply(execute=False);index=len(s.packets);s.end_reply('Response')
        batch=merge_packets(s.packets);labels=torch.full_like(batch['input_ids'],-100)
        offset=sum(p['input_ids'].shape[1] for p in s.packets[:index])
        n=s.packets[index]['input_ids'].shape[1];labels[:,offset:offset+n]=batch['input_ids'][:,offset:offset+n]
        loss=s.model(**batch,labels=labels,use_cache=False).loss;loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(sum(p.grad.abs().sum() for p in s.model.parameters() if p.grad is not None),0)
    def test_actual_native_kv_matches_causal_full_forward_with_media(self):
        s=session();sample=dict(mode='pr',question='Tell me',question_time='00:00')
        with torch.inference_mode():
            s.start_session(sample)
            s.append_media(MediaUnit(1,frame=object(),audio=object()))
            self.assertTrue(s.checked);self.assertEqual(s.past.get_seq_length(),s.length)
            before=s.length;s.append_media(MediaUnit(2,frame=object(),audio=None))
            self.assertGreater(s.length,before)
            result=s.generate_response()
            self.assertFalse(result['complete'])
            s.append_media(MediaUnit(3,audio=object()))
            self.assertEqual(s.last_second,3)
            self.assertTrue(all(not p.requires_grad for p in s.model.parameters()))
            with self.assertRaises(ValueError):s.append_media(MediaUnit(3))
    def test_verification_failure_blocks_generation(self):
        s=session();s.start_session(dict(mode='pr',question='Tell me',question_time='00:00'),execute=False)
        with self.assertRaisesRegex(RuntimeError,'Unverified'):s.generate_response()
    def test_mismatched_cache_is_rejected_before_generation(self):
        s=session();forward=s.model.forward
        def corrupt(*args,**kwargs):
            result=forward(*args,**kwargs)
            if kwargs.get('use_cache') is False:result.logits=result.logits+1
            return result
        s.model.forward=corrupt
        with torch.inference_mode():
            s.start_session(dict(mode='pr',question='Tell me',question_time='00:00'))
            with self.assertRaises(AssertionError):s.append_media(MediaUnit(1,frame=object()))
            with self.assertRaisesRegex(RuntimeError,'Unverified'):s.generate_response()
    def test_training_packets_have_no_cache_or_generation(self):
        s=session()
        s.start_session(dict(mode='pr',question='Tell me',question_time='00:00'),execute=False)
        s.append_media(MediaUnit(1,frame=object()),execute=False)
        s.begin_reply(execute=False);index=len(s.packets);s.end_reply('Response')
        self.assertIsNone(s.past)
        self.assertEqual(s.packets[index]['input_ids'][0,-1].item(),s.eos)
        batch=merge_packets(s.packets)
        self.assertEqual(batch['input_ids'].shape[1],s.length)
        with torch.no_grad():out=s.model(**batch,use_cache=False)
        self.assertEqual(out.logits.shape[1],s.length)


if __name__=='__main__':unittest.main()
