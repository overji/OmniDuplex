"""MiniCPM media prefill with a two-row token projection, not free-form decoding."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .config import PROTOCOL, SILENT, RESPONSE, digest, write,base_identity

GATE_PROMPT = ('Decide whether a new response to the user is warranted after the current second. '
               'Choose <|silent|> or <|response|>. Consider what has already been answered. '
               'Never answer the question yourself.')


def prompt(sample):
    from train.src.agent import plain_prompt
    return '<|im_start|>system\n' + plain_prompt(GATE_PROMPT, sample) + '<|im_end|>'


def core(llm):
    return (llm.get_base_model() if hasattr(llm, 'peft_config') else llm).model


def load(config, training=False):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model
    from .rows import mean_row
    from .rows import DecisionEmbedding
    source = Path(config['classifier_model']).resolve()
    alias = Path(config['cache_dir']) / ('gate_base_' + digest(str(source))[:12])
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists(): alias.symlink_to(source, target_is_directory=True)
    if alias.resolve() != source: raise ValueError('Classifier alias changed')
    model = AutoModel.from_pretrained(str(alias), trust_remote_code=True, torch_dtype=getattr(torch, config['dtype']),
        attn_implementation=config['attention_backend'], init_tts=False).to(config['classifier_device'])
    model.requires_grad_(False); model.eval(); model.init_tts = lambda **kwargs: None
    adapter = Path(config['classifier_adapter']) if config.get('classifier_adapter') else None
    if training and adapter: raise ValueError('Use the original base for a new classifier training run')
    tokenizer = AutoTokenizer.from_pretrained(str(adapter / 'tokenizer' if adapter else alias), trust_remote_code=True)
    old_vocab = model.llm.get_input_embeddings().weight.shape[0]
    if adapter:
        meta = json.loads((adapter / 'jev_gate.json').read_text())
        if meta['protocol'] != PROTOCOL or meta['base'] != str(source) or meta['base_vocab'] != old_vocab or meta['base_identity']!=base_identity(source):
            raise ValueError('Classifier adapter/base identity mismatch')
    else:
        if any(t in tokenizer.get_vocab() for t in (SILENT, RESPONSE)):
            raise ValueError('Use a base tokenizer without gate tokens')
        tokenizer.add_special_tokens({'additional_special_tokens':[SILENT, RESPONSE]}, replace_additional_special_tokens=False)
    ids = [tokenizer.convert_tokens_to_ids(t) for t in (SILENT, RESPONSE)]
    if ids != [old_vocab, old_vocab+1] or any(tokenizer.encode(t, add_special_tokens=False) != [i] for t,i in zip((SILENT,RESPONSE), ids)):
        raise ValueError('Gate tokens must be atomic appended vocabulary entries')
    average = mean_row(model.llm.get_input_embeddings().weight)
    head_init = model.llm.get_output_embeddings().weight[[tokenizer.convert_tokens_to_ids('<|listen|>'),tokenizer.convert_tokens_to_ids('<|speak|>')]].detach().float().clone()
    model.llm.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.vocab_size = len(tokenizer)
    with torch.no_grad(): model.llm.get_input_embeddings().weight[ids] = average.expand(2,-1).to(model.llm.get_input_embeddings().weight.dtype)
    model.llm.requires_grad_(False)
    if training:
        model.llm = get_peft_model(model.llm, LoraConfig(r=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'], target_modules=config['target_modules'], bias='none', task_type='CAUSAL_LM'))
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    elif adapter:
        model.llm = PeftModel.from_pretrained(model.llm, str(adapter))
    embedding = DecisionEmbedding(model.llm.get_input_embeddings(), ids)
    model.llm.set_input_embeddings(embedding)
    head = torch.nn.Linear(head_init.shape[1], 2, bias=False, device=model.device, dtype=torch.float32)
    with torch.no_grad(): head.weight.copy_(head_init)
    if adapter:
        from safetensors.torch import load_file
        tensors = load_file(str(adapter / 'gate_rows.safetensors'))
        with torch.no_grad(): embedding.rows.copy_(tensors['input_rows']); head.weight.copy_(tensors['output_rows'])
        if meta['tokenizer_sha256'] != digest(tokenizer.get_vocab()): raise ValueError('Gate tokenizer changed')
    model.prepare_processor(tokenizer=tokenizer)
    duplex = model.as_duplex(generate_audio=False, sliding_window_mode='off')
    return SimpleNamespace(model=model, duplex=duplex, tokenizer=tokenizer, head=head, embedding=embedding,
                           ids=ids, base=str(source), base_vocab=old_vocab)


def save(state, path):
    from safetensors.torch import save_file
    path.mkdir(parents=True)
    state.model.llm.save_pretrained(path, save_embedding_layers=False)
    state.tokenizer.save_pretrained(path / 'tokenizer')
    save_file({'input_rows':state.embedding.rows.detach().cpu().contiguous(),
               'output_rows':state.head.weight.detach().cpu().contiguous()}, str(path/'gate_rows.safetensors'))
    write(path/'jev_gate.json',dict(protocol=PROTOCOL,base=state.base,base_identity=base_identity(state.base),base_vocab=state.base_vocab,
                                  token_ids=state.ids,tokenizer_sha256=digest(state.tokenizer.get_vocab())))


def prefill(duplex, unit):
    kwargs = dict(frame_list=[unit.frame] if unit.frame is not None else [], max_slice_nums=1, batch_vision_feed=False)
    if unit.audio is not None: kwargs['audio_waveform'] = unit.audio
    if unit.frame is None and unit.audio is None: kwargs['text_list'] = [f'Video time {unit.second}s; no media.']
    result = duplex.streaming_prefill(**kwargs)
    if not result['success']: raise RuntimeError(result.get('reason', 'MiniCPM prefill failed'))


def reset(state, decoder, sample):
    state.duplex.decoder = decoder
    state.duplex._reset_streaming_state()
    state.model.init_streaming_processor()
    decoder.feed(decoder.embed_token(state.tokenizer.encode(prompt(sample), add_special_tokens=False)))


def encode(state, row, config):
    import torch
    from train.src.model import Recorder
    from benchmark.data.prompts import user_request
    from benchmark.data.timestamps import seconds
    from .media import units
    recorder = Recorder(state.model.llm.get_input_embeddings())
    reset(state, recorder, row['sample'])
    prompt_ids=state.tokenizer.encode(prompt(row['sample']),add_special_tokens=False)
    token_positions=[(p,state.ids.index(token)) for p,token in enumerate(prompt_ids) if token in state.ids]
    positions = []
    sent = False; length = sum(len(p) for p in recorder.parts)
    events = {e['second']: e for e in row['events']}
    for unit in units(row, config):
        if not sent and unit.second-1 >= seconds(row['sample']['question_time']):
            state.duplex.streaming_prefill(text_list=[user_request(row['sample'])]); sent=True
        prefill(state.duplex, unit)
        length = sum(len(p) for p in recorder.parts)
        positions.append(length-1)
        value = int(row['labels'][unit.second-1] == 1)
        token_positions.append((length, value))
        recorder.feed(recorder.embed_token(state.ids[value]))
        recorder.feed(recorder.embed_token(state.tokenizer.convert_tokens_to_ids('</unit>')))
        if unit.second in events:
            recorder.feed(recorder.embed_token(state.tokenizer.encode('Previous reply: '+events[unit.second]['target'], add_special_tokens=False)))
        if sum(len(p) for p in recorder.parts) > config['max_sequence_tokens']:
            raise ValueError('Classifier sequence exceeds max_sequence_tokens; no truncation')
    return dict(embeds=recorder.take(), positions=positions, token_positions=token_positions,
                labels=row['train_labels'], sample_index=row['sample_index'])


def loss(state, encoded):
    import torch
    x=encoded['embeds'].to(state.model.device)
    positions=torch.tensor([p for p,_ in encoded['token_positions']],device=x.device)
    ids=torch.tensor([state.ids[v] for _,v in encoded['token_positions']],device=x.device)
    x=x.index_copy(0,positions,state.embedding(ids))
    output=core(state.model.llm)(inputs_embeds=x.unsqueeze(0),use_cache=False,return_dict=True)
    hidden=output.last_hidden_state[0,encoded['positions']]
    return torch.nn.functional.cross_entropy(state.head(hidden.float()),torch.tensor(encoded['labels'],device=x.device))


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
        p=self.state.head(self.decoder.hidden.float()).softmax(-1)[0,1].item()
        if not math.isfinite(p):raise RuntimeError('Classifier produced a nonfinite probability')
        decision=self.sent and p>=self.config['threshold']
        self.decoder.feed(self.decoder.embed_token(self.state.ids[int(decision)]))
        self.decoder.feed(self.decoder.embed_token(self.state.tokenizer.convert_tokens_to_ids('</unit>')))
        return bool(decision),p
    def feedback(self,text):
        self.decoder.feed(self.decoder.embed_token(self.state.tokenizer.encode('Previous reply: '+text,add_special_tokens=False)))
