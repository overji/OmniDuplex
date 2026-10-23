"""Qwen2.5 Thinker incremental media packets with explicit MRoPE positions."""
import json
import time
from pathlib import Path
from typing import Protocol

from benchmark.data.agent_schema import validate_task_start
from benchmark.data.prompts import build_system_prompt, user_request
from benchmark.data.timestamps import seconds
from .config import AGENT, PROTOCOL, digest,base_identity

BACKEND_PROMPT = ('Observe timestamped media and answer the user when the input message ends. '
    'Give a grounded natural-language reply. If external delegation is necessary, append exactly '
    '<|agent|>{"name":"task_start","arguments":{"instruction":"A concise backend task"}}. '
    'Otherwise give only the reply. Do not claim unexecuted tools succeeded. No tool results will arrive.')


class Backend(Protocol):
    def start_session(self, sample): ...
    def append_media(self, unit): ...
    def generate_response(self): ...
    def close_session(self): ...


def parse_reply(text, complete=True):
    response, marker, payload = text.partition(AGENT)
    calls=[];error=None
    if marker:
        try:
            call=json.loads(payload)
            if not complete or AGENT in payload or set(call) != {'name','arguments'} or not validate_task_start(call):
                raise ValueError('Invalid/incomplete delegation')
            calls=[{'type':'function','function':call}]
        except (ValueError,TypeError): error='invalid_or_incomplete_agent_json'
    return dict(response_text=response.strip(),tool_calls=calls,agent_attempted=bool(marker),
                agent_parse_status='malformed' if error else 'structured' if calls else 'absent',
                error=error,text=text,complete=complete)


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
    import torch
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
    model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(config['backend_model'],
        torch_dtype=getattr(torch,config['dtype']),attn_implementation=config['attention_backend'],local_files_only=True)
    model.to(config['backend_device']);model.requires_grad_(False);model.eval()
    processor=Qwen2_5OmniProcessor.from_pretrained(config['backend_model'],local_files_only=True,
                                                use_fast=config.get('processor_use_fast',False))
    if config.get('backend_adapter'):
        if training:raise ValueError('Use original backend weights for a new SFT run')
        from peft import PeftModel
        adapter=Path(config['backend_adapter']);meta=json.loads((adapter/'jev_backend.json').read_text())
        if meta['protocol']!=PROTOCOL or meta['base']!=config['backend_model'] or meta['tokenizer']!=digest(processor.tokenizer.get_vocab()) or meta['base_identity']!=base_identity(config['backend_model']):
            raise ValueError('Backend adapter identity mismatch')
        model=PeftModel.from_pretrained(model,str(adapter))
    if training:
        from peft import LoraConfig,get_peft_model
        names='|'.join(config['target_modules'])
        model=get_peft_model(model,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],target_modules=f'model[.]layers[.][0-9]+[.](self_attn|mlp)[.]({names})',
            bias='none',task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return Qwen25Session(model,processor,config)


class Qwen25Session:
    def __init__(self,model,processor,config):
        self.model=model;self.processor=processor;self.config=config
        self.base=model.get_base_model() if hasattr(model,'peft_config') else model
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
            {'role':'system','content':build_system_prompt(BACKEND_PROMPT,sample)}],
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
