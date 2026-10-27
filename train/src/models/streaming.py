"""MiniCPM原生流式适配：逐秒媒体先输入，再录制SFT标签或采样GRPO动作。"""

import math
import torch
from benchmark.data.prompts import user_request
from benchmark.data.task_text import TaskTextStream
from train.src.domain.trajectory import compile_trajectory, second_target, schedule_stats, TrajectoryError
from train.src.domain.grpo import action_log_probs


class SequenceTooLong(TrajectoryError):
    pass


def system_prompt(record, config):
    from train.src.domain.agent import system_prompt as render
    return render(record['sample'], config)


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
            raise SequenceTooLong('sequence_exceeds_max_sequence_tokens')
        return None, None

    def register_unit_start(self):
        pass


def replay(model, duplex, record, clip, config, trajectory=None):
    """Rebuild every media and token embedding with gradients attached.
    
    SFT按固定调度构图；传入trajectory时按原采样动作重放GRPO。返回embedding/标签/动作位置，并在finally释放流式图引用。"""
    schedule = None
    if trajectory is None:
        schedule = compile_trajectory(record, duplex.tokenizer, config,
                                      duration=getattr(clip, 'duration', None))
        if len(clip.chunks) != schedule['horizon']:
            raise TrajectoryError('clip_does_not_cover_complete_trajectory')
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
                    ids, kinds = second_target(schedule, second, duplex.tokenizer, config, sent)
                    append(ids, kinds)
                    append([duplex.tokenizer.convert_tokens_to_ids('</unit>')])
        encoded = dict(inputs_embeds=torch.cat(recorder.parts), segments=recorder.parts, labels=torch.tensor(labels),
            groups=torch.tensor(groups), positions=positions, token_ids=token_ids, turns=turns)
        if schedule is not None:
            encoded.update(schedule_stats(schedule))
        return encoded
    finally:
        # Release stateful graph references, including audio history, after each trajectory.
        duplex.decoder = decoder
        model.audio_past_key_values = None


@torch.no_grad()
def rollout(model, duplex, record, clip, config, seed):
    """Native prefill/generate with an equivalent, instrumented native action law.
    
    在no_grad下调用原生逐秒prefill/generate并记录采样概率；不注入参考回复，不执行工具。"""
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
