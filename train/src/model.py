"""Freeze native encoders and record exactly the media embeddings sent to the LLM."""
import bisect
import math
from pathlib import Path


def load_model(config, device):
    from train.src.agent import require_protocol
    require_protocol(config)
    import torch
    from transformers import AutoModel
    path = Path(config['model_path']).resolve()
    # Remote-code loading treats dots in local directory names as module separators.
    alias = model_alias(config)
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        alias.symlink_to(path, target_is_directory=True)
    if alias.resolve() != path:
        raise ValueError('Model alias points to a different checkpoint')
    model = AutoModel.from_pretrained(str(alias), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation=config['attention_backend'], init_tts=False, low_cpu_mem_usage=True).eval().to(device)
    model.requires_grad_(False)
    model.init_tts = lambda **kwargs: None
    from transformers import AutoTokenizer
    from train.src.checkpoints import tokenizer_metadata, full_model_identity
    tokenizer = AutoTokenizer.from_pretrained(str(alias), trust_remote_code=True, local_files_only=True)
    tokenizer_metadata(tokenizer)
    if (path / 'full_training_model.json').is_file():
        full_model_identity(path)
    model.prepare_processor(tokenizer=tokenizer)
    duplex = model.as_duplex(generate_audio=False, sliding_window_mode='off')
    return model, duplex


def model_alias(config):
    if config.get('training_method', 'lora_sft') == 'lora_sft':
        return Path(config['media_cache']) / 'MiniCPM_o_4_5'
    import hashlib
    suffix = hashlib.sha256(str(Path(config['model_path']).resolve()).encode()).hexdigest()[:12]
    return Path(config['media_cache']) / 'training_models' / ('minicpmo_' + suffix)


class Recorder:
    """Native prefill uses feed for both tokens and media; do not run inference/KV cache."""
    def __init__(self, embedding):
        self.embedding = embedding
        self.parts = []

    def embed_token(self, ids):
        import torch
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.embedding.weight.device).reshape(-1)
        return self.embedding(ids)

    def feed(self, embeds, return_logits=False):
        self.parts.append(embeds.detach().reshape(-1, embeds.shape[-1]).cpu())
        return (None, None)

    def register_unit_start(self):
        pass

    def take(self):
        import torch
        result = torch.cat(self.parts)
        self.parts = []
        return result


def encode_record(record, model, duplex, config):
    import numpy as np
    import torch
    from PIL import Image
    from benchmark.data.media import media_cache
    from benchmark.data.prompts import user_request
    from benchmark.data.timestamps import seconds
    from train.src.data import target_chunks

    from train.src.agent import training_record
    record = training_record(record)
    sample = record['sample']
    tokenizer = duplex.tokenizer
    target = target_chunks(record, tokenizer, config['max_speak_tokens_per_chunk'],
                           protocol=config['agent_output_protocol'])
    end = record['onset_seconds'] + len(target) - 1
    cache, metadata, audio = media_cache(Path(config['videos_dir']) / sample['video'], Path(config['media_cache']))
    if end > math.floor(metadata['duration']):
        raise ValueError(f"Target exceeds video duration: sample {record['sample_index']}")
    from train.src.agent import system_prompt
    prompt = system_prompt(sample, config)
    prompt = '<|im_start|>system\n' + prompt + '<|im_end|>'
    recorder = Recorder(model.llm.get_input_embeddings())
    duplex.decoder = recorder
    duplex._reset_streaming_state()
    model.init_streaming_processor()
    embeddings, labels, groups = [], [], []

    def add(embeds, ids=None, kinds=None, token_ids=None):
        embeds = embeds.detach().cpu()
        embeddings.append(embeds)
        labels.extend(ids if ids is not None else [-100] * len(embeds))
        groups.extend(kinds if kinds is not None else [0] * len(embeds))
        if len(labels) > config['max_sequence_tokens']:
            raise ValueError('Sequence exceeds max_sequence_tokens; no silent truncation')

    with torch.no_grad():
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        add(recorder.embed_token(prompt_ids), token_ids=prompt_ids)
        qtime = seconds(sample['question_time'])
        sent = False
        for second in range(1, end + 1):
            if not sent and second - 1 >= qtime:
                duplex.streaming_prefill(text_list=[user_request(sample)])
                text_ids = [duplex.unit_token_id] + tokenizer.encode(user_request(sample), add_special_tokens=False)
                add(recorder.take(), token_ids=text_ids); sent = True
            fi = bisect.bisect_left(metadata['frame_times'], second) - 1
            frames = []
            if fi >= 0:
                with Image.open(cache / metadata['frames'][fi]) as image:
                    frames = [image.convert('RGB')]
            chunk = audio[(second - 1) * 16000:second * 16000]
            if len(chunk) < 16000:
                chunk = np.pad(chunk, (0, 16000 - len(chunk)))
            # Match benchmark's audio input, including zero audio for a silent video.
            result = duplex.streaming_prefill(audio_waveform=chunk, frame_list=frames,
                                              max_slice_nums=1, batch_vision_feed=False)
            if not result['success']:
                raise ValueError(result.get('reason', 'Native prefill failed'))
            add(recorder.take())
            if second >= record['onset_seconds']:
                ids, kinds = target[second - record['onset_seconds']]
            else:
                ids = [tokenizer.convert_tokens_to_ids('<|listen|>')]
                # Silence labels are opt-in because annotations may not be exhaustive.
                safe_silence = sent and second < seconds(record['reference']['window_start'])
                kinds = [1 if config['supervise_silence'] and safe_silence else 0]
            add(recorder.embed_token(ids), [i if g else -100 for i, g in zip(ids, kinds)], kinds)
            add(recorder.embed_token(tokenizer.convert_tokens_to_ids('</unit>')))
    return {'inputs_embeds': torch.cat(embeddings), 'labels': torch.tensor(labels),
            'groups': torch.tensor(groups), 'sample_index': record['sample_index'],
            'seconds': end, 'target_chunks': len(target)}


def add_lora(model, config):
    from peft import LoraConfig, get_peft_model
    model.llm = get_peft_model(model.llm, LoraConfig(r=config['lora_rank'],
        lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
        target_modules=config['target_modules'], bias='none', task_type='CAUSAL_LM'))
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.llm.config.use_cache = False
    return model.llm


def loss_for_record(llm, encoded, config, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    x = encoded['inputs_embeds'].to(device)
    x = x.unsqueeze(0)
    labels = encoded['labels'].to(device)
    groups = encoded['groups'].to(device)
    # Avoid materializing [all video tokens, 151748] logits. Only target positions need logits.
    base = llm.get_base_model()
    hidden = base.model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state[0]
    positions = torch.where(labels[1:] != -100)[0]
    if not len(positions):
        raise ValueError('No supervised token')
    losses = []
    weights = (0, config['decision_weight'], config['content_weight'], config['call_weight'])
    for group in (1, 2, 3):
        selected = positions[groups[positions + 1] == group]
        if not len(selected) or not weights[group]:
            continue
        total = hidden.sum() * 0
        for start in range(0, len(selected), config['loss_projection_chunk']):
            ix = selected[start:start + config['loss_projection_chunk']]
            def project_loss(h, y):
                return F.cross_entropy(base.lm_head(h).float(), y, reduction='sum')
            if torch.is_grad_enabled():
                total = total + checkpoint(project_loss, hidden[ix], labels[ix + 1], use_reentrant=False)
            else:
                total = total + project_loss(hidden[ix], labels[ix + 1])
        losses.append(weights[group] * total / len(selected))
    if not losses:
        raise ValueError('All loss weights are zero')
    return sum(losses)
