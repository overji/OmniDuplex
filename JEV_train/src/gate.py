"""MiniCPM全参二分类器：可微媒体前缀与独立二分类头，不使用LoRA或冻结特征缓存。"""
import math
from pathlib import Path
from types import SimpleNamespace

from .config import SILENT, RESPONSE, digest

GATE_PROMPT = ('Decide whether a new response to the user is warranted after the current second. '
               'Choose <|silent|> or <|response|>. Consider what has already been answered. '
               'Never answer the question yourself.')


def prompt(sample):
    from train.src.agent import plain_prompt
    return '<|im_start|>system\n' + plain_prompt(GATE_PROMPT, sample) + '<|im_end|>'


def core(llm):
    return llm.model


def load(config, training=False):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from train.src.infrastructure.tokens import register_task_token, resize_vocabulary, validate_task_token
    source=Path(config.get('classifier_checkpoint') or config['classifier_model']).resolve()
    meta=None
    if config.get('classifier_checkpoint'):
        from .checkpoints import validate_gate_export
        meta=validate_gate_export(source,config)
    alias=Path(config['cache_dir'])/('gate_base_'+digest(str(source))[:12])
    alias.parent.mkdir(parents=True,exist_ok=True)
    if not alias.exists():alias.symlink_to(source,target_is_directory=True)
    if alias.resolve()!=source:raise ValueError('Classifier alias changed')
    model=AutoModel.from_pretrained(str(alias),trust_remote_code=True,torch_dtype=getattr(torch,config['dtype']),
                                   attn_implementation=config['attention_backend'],init_tts=False).to(config['classifier_device'])
    model.init_tts=lambda **kwargs:None
    tokenizer=AutoTokenizer.from_pretrained(str(alias),trust_remote_code=True,local_files_only=True)
    base_vocab=model.llm.get_input_embeddings().weight.shape[0]
    task_id=register_task_token(tokenizer)
    if meta is None:
        if any(t in tokenizer.get_vocab() for t in (SILENT,RESPONSE)):
            raise ValueError('Use a base without decision tokens, or supply a full classifier_checkpoint')
        tokenizer.add_special_tokens({'additional_special_tokens':[SILENT,RESPONSE]},replace_additional_special_tokens=False)
    ids=[tokenizer.convert_tokens_to_ids(t) for t in (SILENT,RESPONSE)]
    if any(tokenizer.encode(t,add_special_tokens=False)!=[i] or i not in tokenizer.all_special_ids for t,i in zip((SILENT,RESPONSE),ids)):
        raise ValueError('Gate tokens must be registered atomic special tokens')
    if len(set([task_id,*ids]))!=3:raise ValueError('Gate and delegation token IDs overlap')
    resize_vocabulary(model,tokenizer)
    head_init=model.llm.get_output_embeddings().weight[[tokenizer.convert_tokens_to_ids('<|listen|>'),tokenizer.convert_tokens_to_ids('<|speak|>')]].detach().float().clone()
    head=torch.nn.Linear(head_init.shape[1],2,bias=False,device=model.device,dtype=torch.float32)
    with torch.no_grad():head.weight.copy_(head_init)
    if meta is not None:
        from safetensors.torch import load_file
        tensors=load_file(str(source/'jev_gate_head.safetensors'))
        with torch.no_grad():head.weight.copy_(tensors['weight'])
        if meta['token_ids']!=ids or meta['task_start_id']!=validate_task_token(tokenizer) or meta['tokenizer_sha256']!=digest(tokenizer.get_vocab()):
            raise ValueError('Classifier token identity mismatch')
    if training:
        from train.src.models.loading import enable_full_training
        enable_full_training(model)
        # 分类输出只使用二分类头；原生成lm_head不参与该任务，不伪称它收到分类梯度。
        model.llm.get_output_embeddings().requires_grad_(False)
    else:model.requires_grad_(False)
    model.eval();head.train(training);head.requires_grad_(training)
    model.prepare_processor(tokenizer=tokenizer)
    duplex=model.as_duplex(generate_audio=False,sliding_window_mode='off')
    return SimpleNamespace(model=model,duplex=duplex,tokenizer=tokenizer,head=head,
        embedding=model.llm.get_input_embeddings(),ids=ids,base=str(Path(config['classifier_model']).resolve()),base_vocab=base_vocab)


def prefill(duplex, unit, differentiable=False):
    kwargs=dict(frame_list=[unit.frame] if unit.frame is not None else [],max_slice_nums=1,batch_vision_feed=False)
    if unit.audio is not None:kwargs['audio_waveform']=unit.audio
    if unit.frame is None and unit.audio is None:kwargs['text_list']=[f'Video time {unit.second}s; no media.']
    return feed(duplex,differentiable,**kwargs)


def feed(duplex,differentiable=False,**kwargs):
    method=duplex.streaming_prefill
    if differentiable and hasattr(method,'__wrapped__'):
        result=method.__wrapped__(duplex,**kwargs)
    else:result=method(**kwargs)
    if not result['success']:raise RuntimeError(result.get('reason','MiniCPM prefill failed'))
    return result


def reset(state, decoder, sample):
    state.duplex.decoder=decoder
    state.duplex._reset_streaming_state()
    state.model.init_streaming_processor()
    decoder.feed(decoder.embed_token(state.tokenizer.encode(prompt(sample),add_special_tokens=False)))


def encode(state,row,config):
    """每次loss重新计算媒体特征并保留梯度；历史回复仅在其触发秒的决策之后输入。"""
    import torch
    from train.src.models.streaming import GraphRecorder
    from benchmark.data.prompts import user_request
    from benchmark.data.timestamps import seconds
    from .media import units
    decoder=state.duplex.decoder
    recorder=GraphRecorder(state.model.llm.get_input_embeddings(),config['max_sequence_tokens'])
    positions=[];sent=False;events={e['second']:e for e in row['events']}
    try:
        reset(state,recorder,row['sample'])
        for unit in units(row,config):
            if not sent and unit.second-1>=seconds(row['sample']['question_time']):
                feed(state.duplex,True,text_list=[user_request(row['sample'])]);sent=True
            prefill(state.duplex,unit,differentiable=True)
            positions.append(recorder.length-1)
            value=int(row['labels'][unit.second-1]==1)
            recorder.feed(recorder.embed_token(state.ids[value]))
            recorder.feed(recorder.embed_token(state.tokenizer.convert_tokens_to_ids('</unit>')))
            if unit.second in events:
                recorder.feed(recorder.embed_token(state.tokenizer.encode('Previous reply: '+events[unit.second]['target'],add_special_tokens=False)))
        if len(positions)!=len(row['train_labels']):raise ValueError('Incomplete classifier media trajectory')
        return dict(embeds=torch.cat(recorder.parts),positions=positions,labels=row['train_labels'],sample_index=row['sample_index'])
    finally:
        state.duplex.decoder=decoder
        state.model.audio_past_key_values=None


def loss(state,encoded):
    import torch
    output=core(state.model.llm)(inputs_embeds=encoded['embeds'].unsqueeze(0),use_cache=False,return_dict=True)
    hidden=output.last_hidden_state[0,encoded['positions']]
    logits=state.head(hidden.to(state.head.weight.dtype)).float()
    return torch.nn.functional.cross_entropy(logits,torch.tensor(encoded['labels'],device=hidden.device))


class CachedDecoder:
    def __init__(self,state,limit): self.state=state;self.limit=limit;self.past=None;self.length=0;self.hidden=None
    def embed_token(self,ids):
        import torch
        return self.state.embedding(torch.as_tensor(ids,device=self.state.model.device).reshape(-1))
    def register_unit_start(self): pass
    def feed(self,embeds,return_logits=False):
        import torch
        embeds=embeds.reshape(1,-1,embeds.shape[-1]);self.length+=embeds.shape[1]
        if self.length>self.limit:raise ValueError('Classifier context overflow; no truncation')
        out=core(self.state.model.llm)(inputs_embeds=embeds,past_key_values=self.past,
            attention_mask=torch.ones((1,self.length),device=embeds.device,dtype=torch.long),use_cache=True,return_dict=True)
        self.past=out.past_key_values;self.hidden=out.last_hidden_state[:,-1]
        return None,None


class GateSession:
    def __init__(self,state,config):self.state=state;self.config=config
    def start(self,sample):
        self.sample=sample;self.sent=False
        self.decoder=CachedDecoder(self.state,self.config['max_sequence_tokens']);reset(self.state,self.decoder,sample)
    def decide(self,unit):
        from benchmark.data.timestamps import seconds
        from benchmark.data.prompts import user_request
        if not self.sent and unit.second-1>=seconds(self.sample['question_time']):
            self.state.duplex.streaming_prefill(text_list=[user_request(self.sample)]);self.sent=True
        prefill(self.state.duplex,unit)
        p=self.state.head(self.decoder.hidden.to(self.state.head.weight.dtype)).float().softmax(-1)[0,1].item()
        if not math.isfinite(p):raise RuntimeError('Classifier produced a nonfinite probability')
        decision=self.sent and p>=self.config['threshold']
        self.decoder.feed(self.decoder.embed_token(self.state.ids[int(decision)]))
        self.decoder.feed(self.decoder.embed_token(self.state.tokenizer.convert_tokens_to_ids('</unit>')))
        return bool(decision),p
    def feedback(self,text):
        self.decoder.feed(self.decoder.embed_token(self.state.tokenizer.encode('Previous reply: '+text,add_special_tokens=False)))
