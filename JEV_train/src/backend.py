"""Qwen2.5 Thinker incremental media packets with explicit MRoPE positions."""
import json
import time
from pathlib import Path
from typing import Protocol

from benchmark.data.prompts import user_request
from train.src.agent import plain_prompt
from benchmark.data.timestamps import seconds
from .config import PROTOCOL, digest,base_identity

BACKEND_PROMPT = ('Observe timestamped media and answer the user when the input message ends. '
    'Give a grounded natural-language reply. If external delegation is necessary, append exactly '
    '<|task_start|> followed by one concise, self-contained plain-text backend instruction. '
    'Output reply<|task_start|>instruction on one line, with the separator exactly once. '
    'Do not output JSON or a function envelope. Use only available capabilities and observed facts. '
    'Otherwise give only the reply. Do not claim unexecuted tools succeeded. No tool results will arrive.')


class Backend(Protocol):
    def start_session(self, sample): ...
    def append_media(self, unit): ...
    def generate_response(self): ...
    def close_session(self): ...


def parse_reply(text, complete=True):
    from benchmark.data.task_text import event
    result = event(text, 0, complete=complete)
    if '<|agent|>' in text:
        result.update(tool_calls=[], agent_attempted=True, agent_parse_status='malformed',
                      task_text_status='legacy_format', task_text_format_valid=False)
    keep = ('response_text', 'tool_calls', 'agent_attempted', 'agent_parse_status', 'agent_instruction',
            'task_text_status', 'task_text_format_valid', 'task_text_generation_complete', 'text')
    result = {key: result[key] for key in keep}
    result.update(complete=complete, error=None if result['agent_parse_status'] in ('absent','structured')
                  else 'invalid_or_incomplete_task_text')
    return result


def merge_packets(packets):
    import torch
    result={}
    for key in set().union(*(p.keys() for p in packets)):
        values=[p[key] for p in packets if key in p]
        if key in ('attention_mask','cache_position'):continue
        if key in ('input_ids','position_ids'): result[key]=torch.cat(values,dim=-1)
        else: result[key]=torch.cat(values,dim=0)
    result['attention_mask']=torch.ones_like(result['input_ids'])
    return result


def load(config,training=False):
    """加载原始Thinker或完整SFT导出，不支持PEFT；训练时媒体和文本模型一起更新。"""
    import torch
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
    source = config.get('backend_checkpoint') or config['backend_model']
    if config.get('backend_checkpoint'):
        meta=json.loads((Path(source)/'jev_backend.json').read_text())
        if meta.get('protocol')!=PROTOCOL or meta.get('base')!=config['backend_model'] or meta.get('base_identity')!=base_identity(config['backend_model']):
            raise ValueError('Backend full-SFT/base identity mismatch')
    model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(source,
        torch_dtype=getattr(torch,config['dtype']),attn_implementation=config['attention_backend'],local_files_only=True)
    model.to(config['backend_device']);model.requires_grad_(training);model.eval()
    processor=Qwen2_5OmniProcessor.from_pretrained(source,local_files_only=True,
                                                use_fast=config.get('processor_use_fast',False))
    if config.get('backend_checkpoint'):
        if meta['tokenizer']!=digest(processor.tokenizer.get_vocab()) or meta['files']!=base_identity(source):
            raise ValueError('Backend tokenizer/model identity mismatch')
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache=False
    return Qwen25Session(model,processor,config)


class Qwen25Session:
    def __init__(self,model,processor,config):
        self.model=model;self.processor=processor;self.config=config
        self.base=model
        self.tokenizer=processor.tokenizer
        for token in ('<|im_start|>','<|im_end|>'):
            if token not in self.tokenizer.get_vocab():raise ValueError('Unexpected Qwen template')
        self.eos=self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        self.close_session()
    def close_session(self):
        self.past=None;self.length=0;self.offset=0;self.logits=None;self.packets=[]
        self.checked=False;self.last_second=0;self.question_sent=False;self.closed=True
    def text_packet(self,text):
        import torch
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        return {'input_ids':torch.tensor([ids],dtype=torch.long,device=self.model.device)}
    def position(self,packet):
        import torch
        packet={k:v.to(self.model.device) for k,v in packet.items() if k!='attention_mask'}
        for key in ('input_features','pixel_values','pixel_values_videos'):
            if key in packet:packet[key]=packet[key].to(self.model.dtype)
        features=packet.get('feature_attention_mask')
        local,_=self.base.get_rope_index(input_ids=packet['input_ids'],image_grid_thw=packet.get('image_grid_thw'),
            video_grid_thw=packet.get('video_grid_thw'),attention_mask=torch.ones_like(packet['input_ids']),
            use_audio_in_video=False,audio_seqlens=features.sum(-1) if features is not None else None)
        packet['position_ids']=local+self.offset
        self.offset=int(packet['position_ids'].max())+1
        return packet
    def append_packet(self,packet,execute=True):
        import torch
        packet=self.position(packet)
        n=packet['input_ids'].shape[1]
        if self.length+n>min(self.config['max_sequence_tokens'],self.base.config.text_config.max_position_embeddings):
            raise ValueError('Backend context overflow; no truncation')
        if execute:
            result=self.model(**packet,past_key_values=self.past,
                attention_mask=torch.ones((1,self.length+n),dtype=torch.long,device=self.model.device),
                cache_position=torch.arange(self.length,self.length+n,device=self.model.device),use_cache=True,return_dict=True)
            self.past=result.past_key_values;self.logits=result.logits[:,-1]
        self.length+=n
        self.packets.append(packet)
        return packet
    def append_text(self,text,execute=True):return self.append_packet(self.text_packet(text),execute)
    def start_session(self,sample,execute=True):
        self.close_session();self.sample=sample;self.closed=False
        self.append_text(self.processor.apply_chat_template([
            {'role':'system','content':plain_prompt(BACKEND_PROMPT,sample)}],
            tokenize=False,add_generation_prompt=False)+'<|im_start|>user\n',execute)
    def media_packet(self,unit):
        text=f'\nVideo time [{unit.second-1},{unit.second}) seconds:\n'
        kwargs=dict(return_tensors='pt',add_special_tokens=False)
        if unit.frame is not None:
            text+='<|vision_bos|><|IMAGE|><|vision_eos|>'
            kwargs.update(images=[unit.frame],images_kwargs={'max_pixels':self.config['max_pixels']})
        if unit.audio is not None:
            text+='<|audio_bos|><|AUDIO|><|audio_eos|>'
            kwargs.update(audio=[unit.audio],sampling_rate=16000)
        return dict(self.processor(text=text,**kwargs))
    def append_media(self,unit,execute=True):
        if self.closed or unit.second!=self.last_second+1:raise ValueError('Media must arrive exactly once in order')
        if not self.question_sent and unit.second-1>=seconds(self.sample['question_time']):
            self.append_text(user_request(self.sample)+'\n',execute);self.question_sent=True
        packet=self.append_packet(self.media_packet(unit),execute);self.last_second=unit.second
        if execute and not self.checked:self.verify_cache()
        # Only the first actual prefix is retained for cache comparison, not a second full-media cache.
        if execute and self.checked:self.packets.clear()
        return packet
    def verify_cache(self):
        import torch
        if len(self.packets)<2 or self.past is None:raise RuntimeError('Cache verification needs a multimodal prefix')
        batch=merge_packets(self.packets)
        if batch['input_ids'].shape[1]!=self.length or self.past.get_seq_length()!=self.length:
            raise RuntimeError('KV length/media token count mismatch')
        with torch.no_grad():full=self.model(**batch,use_cache=False,return_dict=True).logits[:,-1]
        torch.testing.assert_close(full.float(),self.logits.float(),atol=self.config['cache_atol'],rtol=self.config['cache_rtol'])
        self.checked=True
    def begin_reply(self,execute=True):
        if not self.question_sent:raise RuntimeError('Cannot answer before question input')
        return self.append_text('<|im_end|>\n<|im_start|>assistant\n',execute)
    def end_reply(self,text,execute=False):
        packet=self.append_text(text+'<|im_end|>',execute)
        self.append_text('\n<|im_start|>user\n',execute)
        return packet
    def generate_response(self):
        import torch
        if not self.checked:raise RuntimeError('Unverified incremental cache; generation prohibited')
        started=time.monotonic();self.begin_reply();ids=[];complete=False;first_token=None
        for _ in range(self.config['max_new_tokens']):
            token=int(self.logits.argmax(-1).item())
            if token!=self.eos and first_token is None:first_token=time.monotonic()-started
            self.append_packet({'input_ids':torch.tensor([[token]],device=self.model.device)})
            if token==self.eos:complete=True;break
            ids.append(token)
        if not complete:self.append_text('<|im_end|>')
        self.append_text('\n<|im_start|>user\n');self.packets.clear()
        result=parse_reply(self.tokenizer.decode(ids,skip_special_tokens=False),complete)
        result['first_token_latency_seconds']=first_token
        return result


BACKENDS={'qwen25_omni':load}
