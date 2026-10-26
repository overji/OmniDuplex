"""Native streaming inputs and differentiable replay, without editing model code."""
import bisect
import copy
import math
from contextlib import contextmanager
from pathlib import Path

import torch

from benchmark.data.timestamps import seconds
from benchmark.data.prompts import user_request
from benchmark.data.media import media_cache
from benchmark.data.task_text import TaskTextStream
from train.src.data import target_chunks
from train.src.grpo import action_log_probs


class SequenceTooLong(ValueError):
    pass


def system_prompt(record, config):
    from train.src.agent import system_prompt as render
    return render(record['sample'], config)


class Clip:
    """Only raw/preprocessed inputs are cached; encoder outputs are never cached."""
    def __init__(self, record, config, horizon):
        import numpy as np
        from PIL import Image
        cache, metadata, audio = media_cache(Path(config['videos_dir']) / record['sample']['video'], Path(config['media_cache']))
        self.horizon = min(horizon, math.floor(metadata['duration']))
        self.chunks, self.processed = [], {}
        self.question_time = seconds(record['sample']['question_time'])
        for second in range(1, self.horizon + 1):
            fi = bisect.bisect_left(metadata['frame_times'], second) - 1
            frames = []
            if fi >= 0:
                with Image.open(cache / metadata['frames'][fi]) as image:
                    frames = [image.convert('RGB')]
            chunk = audio[(second - 1) * 16000:second * 16000]
            if len(chunk) < 16000:
                chunk = np.pad(chunk, (0, 16000 - len(chunk)))
            self.chunks.append(dict(audio_waveform=chunk, frame_list=frames,
                                    max_slice_nums=1, batch_vision_feed=False))

    @contextmanager
    def preprocessing_cache(self, processor):
        """Cache per-call CPU BatchFeatures; .to() mutates them, so return copies."""
        names = ('process_image', 'process_audio_streaming')
        originals = {name: getattr(processor, name) for name in names}
        counters = {name: 0 for name in names}
        def wrap(name):
            def call(*args, **kwargs):
                key = (name, counters[name]); counters[name] += 1
                if key not in self.processed:
                    feature = copy.deepcopy(originals[name](*args, **kwargs))
                    snapshot = (copy.deepcopy(processor.get_streaming_snapshot())
                                if name == 'process_audio_streaming' else None)
                    self.processed[key] = (feature, snapshot)
                feature, snapshot = self.processed[key]
                if snapshot is not None:
                    processor.restore_streaming_snapshot(copy.deepcopy(snapshot))
                return copy.deepcopy(feature)
            return call
        try:
            for name in names:
                setattr(processor, name, wrap(name))
            yield
        finally:
            for name, fn in originals.items():
                setattr(processor, name, fn)


class GraphRecorder:
    def __init__(self, embedding, limit):
        self.embedding, self.limit = embedding, limit
        self.parts, self.length = [], 0

    def embed_token(self, ids):
        ids = torch.as_tensor(ids, device=self.embedding.weight.device, dtype=torch.long).reshape(-1)
        return self.embedding(ids)

    def feed(self, embeds, return_logits=False):
        embeds = embeds.reshape(-1, embeds.shape[-1])
        self.parts.append(embeds)
        self.length += len(embeds)
        if self.length > self.limit:
            raise SequenceTooLong('Sequence exceeds max_sequence_tokens; excluded without truncation')
        return None, None

    def register_unit_start(self):
        pass


def replay(model, duplex, record, clip, config, trajectory=None):
    """Rebuild every media and token embedding with gradients attached."""
    decoder = duplex.decoder
    recorder = GraphRecorder(model.llm.get_input_embeddings(), config['max_sequence_tokens'])
    duplex.decoder = recorder
    labels, groups, positions, token_ids, turns = [], [], [], [], []
    def align():
        labels.extend([-100] * (recorder.length - len(labels)))
        groups.extend([0] * (recorder.length - len(groups)))
    def append(ids, kinds=None, actions=None):
        start = recorder.length
        if trajectory is not None and actions is not None:
            # Native generation feeds each sampled/forced output token separately.
            for token in ids:
                recorder.feed(recorder.embed_token(token))
        else:
            recorder.feed(recorder.embed_token(ids))
        align()
        if kinds:
            for offset, (token, group) in enumerate(zip(ids, kinds)):
                if group:
                    labels[start + offset], groups[start + offset] = token, group
        for action in actions or []:
            # Hidden state at p-1 predicts the token placed at p.
            positions.append(start + action['offset'] - 1)
            token_ids.append(ids[action['offset']])
            turns.append(action['turn_open'])
    try:
        duplex._reset_streaming_state()
        model.init_streaming_processor()
        prompt = '<|im_start|>system\n' + system_prompt(record, config) + '<|im_end|>'
        append(duplex.tokenizer.encode(prompt, add_special_tokens=False))
        targets = None if trajectory is not None else target_chunks(record, duplex.tokenizer,
            config['max_speak_tokens_per_chunk'], protocol=config['agent_output_protocol'])
        sent = False
        with clip.preprocessing_cache(duplex.processor):
            for i, inputs in enumerate(clip.chunks):
                second = i + 1
                if not sent and i >= clip.question_time:
                    result = type(duplex).streaming_prefill.__wrapped__(duplex, text_list=[user_request(record['sample'])])
                    if not result['success']:
                        raise RuntimeError(result)
                    sent = True
                result = type(duplex).streaming_prefill.__wrapped__(duplex, **inputs)
                if not result['success']:
                    raise RuntimeError(result)
                align()
                if trajectory is not None:
                    chunk = trajectory['chunks'][i]
                    append(chunk['ids'], actions=chunk['actions'])
                else:
                    if second >= record['onset_seconds']:
                        ids, kinds = targets[second - record['onset_seconds']]
                    else:
                        ids = [duplex.listen_token_id]
                        safe = sent and second < seconds(record['reference']['window_start'])
                        kinds = [1 if config['supervise_silence'] and safe else 0]
                    append(ids, kinds)
                    append([duplex.tokenizer.convert_tokens_to_ids('</unit>')])
        encoded = dict(inputs_embeds=torch.cat(recorder.parts), segments=recorder.parts, labels=torch.tensor(labels),
            groups=torch.tensor(groups), positions=positions, token_ids=token_ids, turns=turns)
        return encoded
    finally:
        # Release stateful graph references, including audio history, after each trajectory.
        duplex.decoder = decoder
        model.audio_past_key_values = None


@torch.no_grad()
def rollout(model, duplex, record, clip, config, seed):
    """Native prefill/generate with an equivalent, instrumented native action law."""
    from transformers import set_seed
    set_seed(seed)
    model.eval()
    if config.get('batch_prompt_prefill', True):
        # Native prepare feeds the protected system prefix one token at a time.
        # With ZeRO hooks this dominates short rollouts. Sliding windows are off;
        # prefill exactly the same causal tokens in one decoder call instead.
        duplex.clear_break_event(); duplex.clear_session_stop()
        duplex._reset_streaming_state(); duplex.decoder.reset()
        model.init_streaming_processor()
        prefix = duplex.tokenizer.encode('<|im_start|>system\n' + system_prompt(record, config), add_special_tokens=False)
        suffix = duplex.tokenizer.encode('<|im_end|>', add_special_tokens=False)
        duplex.decoder.feed(duplex.decoder.embed_tokens(prefix + suffix))
        duplex.decoder.register_system_prompt()
    else:
        duplex.prepare(prefix_system_prompt=system_prompt(record, config))
    duplex.force_listen_count = math.ceil(clip.question_time)
    native_decode = duplex.decoder.decode
    actions = []
    def decode(logits, **kwargs):
        law = action_log_probs(logits, duplex.decoder.forbidden_token_ids,
            duplex.chunk_eos_token_id, duplex.listen_token_id, duplex.tts_bos_token_id,
            not duplex.current_turn_ended)[0]
        token = torch.multinomial(law.exp(), 1)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(token, src=0)
        actions.append(dict(offset=len(duplex.total_ids) - before, turn_open=not duplex.current_turn_ended,
                            old_log_prob=law[token.item()].item()))
        return token
    duplex.decoder.decode = decode
    chunks, events = [], []
    parser, sent = TaskTextStream(duplex.tokenizer), False
    try:
        with clip.preprocessing_cache(duplex.processor):
            for i, inputs in enumerate(clip.chunks):
                if not sent and i >= clip.question_time:
                    result = duplex.streaming_prefill(text_list=[user_request(record['sample'])])
                    if not result['success']:
                        raise RuntimeError(result)
                    sent = True
                result = duplex.streaming_prefill(**inputs)
                if not result['success']:
                    raise RuntimeError(result)
                before = len(duplex.total_ids)
                actions = []
                result = duplex.streaming_generate(max_new_speak_tokens_per_chunk=config['max_speak_tokens_per_chunk'],
                    decode_mode='sampling', temperature=1., top_k=0, top_p=1.,
                    text_repetition_penalty=1., listen_prob_scale=1.)
                ids = list(duplex.total_ids[before:])
                chunks.append(dict(ids=ids, actions=actions))
                if duplex.decoder.get_cache_length() > config['max_sequence_tokens']:
                    raise SequenceTooLong('Rollout exceeds token limit; discard whole group')
                if sent:
                    event = parser.feed(ids, i + 1, ended=result['end_of_turn'] or result['is_listen'], interrupted=result.get('is_interrupt', False))
                    if event:
                        events.append(event)
        pending = parser.finish(complete=False)
        if pending:
            events.append(pending)
        return dict(chunks=chunks, events=events, seed=seed)
    finally:
        duplex.decoder.decode = native_decode
        duplex.decoder.reset()
        duplex.total_hidden = []
        model.audio_past_key_values = None


class TrainGraph(torch.nn.Module):
    """DeepSpeed sees the complete model and all trainable media computations."""
    def __init__(self, model, duplex, config):
        super().__init__()
        self.model, self.duplex, self.config = model, duplex, config
        model.requires_grad_(True)
        if hasattr(model, 'tts') and model.tts is not None:
            model.tts.requires_grad_(False)
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.llm.config.use_cache = False
        # Keep native streaming APM caches differentiable. Its checkpointing is
        # intentionally disabled: cache mutation cannot be recomputed safely.
        if hasattr(model.apm, 'gradient_checkpointing_disable'):
            model.apm.gradient_checkpointing_disable()
        if hasattr(model.vpm, 'gradient_checkpointing_enable'):
            model.vpm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    def forward(self, record, clip, trajectory=None):
        from torch.utils.checkpoint import checkpoint
        from train.src.grpo import policy_loss
        config, llm = self.config, self.model.llm
        encoded = replay(self.model, self.duplex, record, clip, config, trajectory)
        self.last_stats = dict(sequence_tokens=len(encoded['inputs_embeds']),
                               supervised_tokens=int((encoded['labels'] != -100).sum()))
        if trajectory is None:
            hidden = llm.model(inputs_embeds=encoded['inputs_embeds'][None], use_cache=False,
                               return_dict=True).last_hidden_state[0]
        else:
            # BF16 full-sequence recomputation can differ materially from native
            # chunked KV decoding. Preserve native feed boundaries and positions
            # for on-policy ratios; retain a differentiable KV cache for backward.
            past, offset, states = None, 0, []
            for segment in encoded['segments']:
                pos = torch.arange(offset, offset + len(segment), device=segment.device)[None]
                out = llm.model(inputs_embeds=segment[None], position_ids=pos,
                    past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                states.append(out.last_hidden_state[0]); offset += len(segment)
            hidden = torch.cat(states)
        if trajectory is None:
            labels, groups = encoded['labels'].to(hidden.device), encoded['groups'].to(hidden.device)
            loss = hidden.sum() * 0
            active = False
            for group, key in ((1, 'decision_weight'), (2, 'content_weight'), (3, 'call_weight')):
                positions = torch.where((labels[1:] != -100) & (groups[1:] == group))[0]
                if not len(positions) or not config[key]:
                    continue
                active = True
                for ix in positions.split(config['loss_projection_chunk']):
                    def project(h, targets):
                        return torch.nn.functional.cross_entropy(llm.lm_head(h).float(), targets, reduction='sum')
                    loss = loss + config[key] * checkpoint(project, hidden[ix], labels[ix + 1], use_reentrant=False) / len(positions)
            if not active:
                raise ValueError('No supervised tokens with nonzero loss weight')
            return loss
        positions = torch.tensor(encoded['positions'], device=hidden.device)
        ids = torch.tensor(encoded['token_ids'], device=hidden.device)
        turns = torch.tensor(encoded['turns'], device=hidden.device)
        if not len(positions):
            raise ValueError('No sampled actions in trajectory')
        log_probs = []
        for start in range(0, len(positions), config['loss_projection_chunk']):
            end = start + config['loss_projection_chunk']
            def project(h, target, turn_open):
                law = action_log_probs(llm.lm_head(h), self.duplex.decoder.forbidden_token_ids,
                    self.duplex.chunk_eos_token_id, self.duplex.listen_token_id,
                    self.duplex.tts_bos_token_id, turn_open)
                return law.gather(-1, target[:, None]).squeeze(-1)
            log_probs.append(checkpoint(project, hidden[positions[start:end]], ids[start:end], turns[start:end], use_reentrant=False))
        current = torch.cat(log_probs)
        old = torch.tensor([a['old_log_prob'] for c in trajectory['chunks'] for a in c['actions']], device=hidden.device)
        error = (current.detach() - old).abs().max()
        if 'advantage' not in trajectory:
            return current.detach(), error
        loss, ratio = policy_loss(current, old, trajectory['advantage'], config.get('clip_epsilon', .2))
        return loss, dict(replay_max_logprob_error=error.item(), ratio_mean=ratio.mean().item(),
                          actions=len(current))
