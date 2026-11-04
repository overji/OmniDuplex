"""Local Gemma4 with timestamped prefix frames and original audio; no media files cached."""
import json
import math
import subprocess
from pathlib import Path
from data_quality.core import SYSTEM, CHECKS, frame_times, validate_judgment


class GemmaAudit:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        if config['expand_kv_for_sdpa']:
            import transformers.integrations.sdpa_attention as sdpa
            sdpa.use_gqa_in_sdpa=lambda attention_mask,key: False
        torch.set_num_threads(config['cpu_threads'])
        torch.cuda.set_per_process_memory_fraction(config['memory_fraction'])
        self.processor=AutoProcessor.from_pretrained(config['model_path'],local_files_only=True)
        self.model=AutoModelForMultimodalLM.from_pretrained(config['model_path'],local_files_only=True,
            dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        self.config=config
        self.sr=self.processor.feature_extractor.sampling_rate
        self.chunk=int(self.processor.audio_seq_length*self.processor.audio_ms_per_token/1000*self.sr)
        self.video=None;self.audio=None;self.container=None

    def media(self, path, cutoff, start):
        import av
        import numpy as np
        from PIL import Image
        if self.video!=str(path):
            if self.container is not None:self.container.close()
            self.container=av.open(str(path));self.video=str(path)
            self.container.streams.video[0].codec_context.thread_count=self.config['cpu_threads']
            if self.container.streams.audio:
                raw=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar',str(self.sr),
                    '-threads',str(self.config['cpu_threads']),'-f','f32le','pipe:1'],capture_output=True,check=True,timeout=120).stdout
                self.audio=np.frombuffer(raw,dtype=np.float32).copy()
            else:self.audio=np.empty(0,dtype=np.float32)
        stream=self.container.streams.video[0];frames=[]
        for t in frame_times(cutoff,start,self.config['frame_count'],self.config['dense_seconds']):
            self.container.seek(max(0,int(t/float(stream.time_base))),stream=stream,backward=True)
            previous=None
            for frame in self.container.decode(stream):
                pts=float(frame.pts*stream.time_base) if frame.pts is not None else None
                if pts is None:continue
                if pts>t:break
                previous=(pts,frame.to_image().copy())
            if previous is not None and previous[0]<=cutoff:
                frames.append(previous)
        if not frames:raise ValueError('No causal video frame decoded')
        audio=self.audio[:int(cutoff*self.sr)]
        return frames,[audio[i:i+self.chunk] for i in range(0,len(audio),self.chunk)]

    def review(self, sample, ri, path, cutoff):
        import torch
        from benchmark.data.prompts import describe_tool
        ref=sample['responses'][ri]
        from benchmark.data.timestamps import seconds
        frames,audio=self.media(path,cutoff,seconds(ref['window_start']))
        content=[]
        for t,img in frames:content.extend([{'type':'text','text':f'Video frame at {t:.3f} seconds:'},{'type':'image','image':img}])
        for i,chunk in enumerate(audio):
            begin=i*self.chunk/self.sr
            content.extend([{'type':'text','text':f'Original audio [{begin:.3f}, {begin+len(chunk)/self.sr:.3f}) seconds:'},{'type':'audio','audio':chunk}])
        payload={k:sample.get(k) for k in ['question','question_time','task','ability','mode','sample_type','audio_dependency','vision_dependency']}
        payload.update(response_index=ri,candidate_response=ref,prefix_end_seconds=cutoff,
            available_capabilities=[dict(name=t,description=describe_tool(t)) for t in (sample.get('available_tools') or [])])
        from data_generation.src.domain.catalog import ABILITIES
        ability=ABILITIES.get(sample.get('ability'))
        payload['task_definition']=ability.description if ability else sample.get('task')
        content.append({'type':'text','text':json.dumps(payload,ensure_ascii=False)})
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':content}]
        attempts=[]
        for attempt in range(self.config['attempts']):
            text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=self.processor(text=text,images=[img for _,img in frames],audio=audio or None,
                sampling_rate=self.sr,images_kwargs={'max_soft_tokens':self.config['image_soft_tokens']},return_tensors='pt')
            length=inputs['input_ids'].shape[-1]
            if length>self.config['max_input_tokens']:raise ValueError('Audit prefix exceeds configured context budget')
            inputs=inputs.to(self.model.device,dtype=self.model.dtype)
            with torch.inference_mode():
                output=self.model.generate(**inputs,do_sample=False,max_new_tokens=self.config['max_new_tokens']*(attempt+1))
            tokens=output[0,length:];raw=self.processor.decode(tokens,skip_special_tokens=True).strip()
            eos=self.model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            try:
                if not len(tokens) or int(tokens[-1]) not in eos:raise ValueError('Generation truncated before EOS')
                begin=raw.find('{');value,_=json.JSONDecoder().raw_decode(raw[begin:])
                judgment=validate_judgment(value,cutoff)
                attempts.append(dict(raw=raw,input_tokens=length,output_tokens=len(tokens)))
                return dict(status='complete',**judgment,attempts=attempts,
                    coverage=dict(prefix_end=cutoff,frame_times=[t for t,_ in frames],audio_seconds=sum(len(a) for a in audio)/self.sr,
                                  sparse_video=True,full_prefix_audio=bool(audio)))
            except (ValueError,KeyError,TypeError) as exc:
                attempts.append(dict(raw=raw,error=str(exc),input_tokens=length,output_tokens=len(tokens)))
                messages.extend([{'role':'assistant','content':raw},{'role':'user','content':f'Fix the JSON schema only using the same evidence. Error: {exc}. Return JSON, all required check keys, concise reasons.'}])
            finally:
                del inputs,output,tokens
                torch.cuda.empty_cache()
        return dict(status='failed',error='Invalid or truncated model JSON',attempts=attempts)
