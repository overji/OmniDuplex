"""连续多回复时间线。窗口只用于离线标签调度，未来回复不提前进入模型上下文。"""

import math

from benchmark.data.timestamps import seconds
from train.src.domain.agent import training_record
from train.src.domain.targets import target_chunks
from train.src.domain.records import response_record


class TrajectoryError(ValueError):
    def __init__(self, reason, response_index=None):
        self.reason = reason
        self.response_index = response_index
        super().__init__(reason + (f':response_index={response_index}' if response_index is not None else ''))


def compile_trajectory(record, tokenizer, config, duration=None):
    """返回按秒索引的chunks、全部targets和终点horizon。只在原窗口内顺延；冲突/越界抛TrajectoryError，由上层整条排除QA。duration为完整视频秒数。"""
    if 'targets' not in record:
        raise ValueError('Single-response records are unsupported; run prepare again')
    record = training_record(record)
    sample = record['sample']
    ordered = sorted(record['targets'], key=lambda t: (seconds(t['reference']['window_start']), t['response_index']))
    chunks, targets, windows = {}, [], []
    end = 0
    for target in ordered:
        ri, ref = target['response_index'], target['reference']
        # Recheck temporal evidence even if prepared metadata was edited.
        causal, error = response_record(sample, record['sample_index'], ri, ref, config)
        if error:
            raise TrajectoryError(error, ri)
        # 一个原生时间单元只容纳一个回复chunk，下一回复至少等到下一秒。
        onset = max(causal['onset_seconds'], end + 1)
        if onset > seconds(ref['window_end']):
            raise TrajectoryError('response_schedule_conflict', ri)
        if config.get('sft_horizon') != 'full_video' and onset > config['max_prefix_seconds']:
            raise TrajectoryError('prefix_too_long', ri)
        try:
            body = target_chunks({'sample': sample, **target}, tokenizer,
                                 config['max_speak_tokens_per_chunk'], protocol=config['agent_output_protocol'])
        except ValueError as exc:
            raise TrajectoryError(f'invalid_target:{exc}', ri) from exc
        end = onset + len(body) - 1
        if duration is not None and end > math.floor(duration):
            raise TrajectoryError('target_exceeds_video_duration', ri)
        chunks.update((onset + offset, chunk) for offset, chunk in enumerate(body))
        targets.append({'response_index': ri, 'onset_seconds': onset,
                        'end_seconds': end, 'chunks': len(body)})
        windows.append((seconds(ref['window_start']), seconds(ref['window_end'])))
    # Keep the optional tail beyond every reference window and response chunk.
    # Idle positions inside those windows are supervised by second_target too.
    tail_start = math.floor(max(end, *(stop for _, stop in windows))) + 1
    horizon = end
    tail = config.get('post_response_silence_seconds', 0)
    if tail:
        stop = min(tail_start + tail - 1, config['max_prefix_seconds'])
        if duration is not None:
            stop = min(stop, math.floor(duration))
        horizon = max(end, stop)
    if config.get('sft_horizon') == 'full_video':
        # None requests full media duration from Clip before duration is known.
        horizon = math.floor(duration) if duration is not None else None
    return {'chunks': chunks, 'targets': targets, 'windows': windows,
            'horizon': horizon, 'response_end_seconds': end, 'tail_start_seconds': tail_start,
            'tail_silence_seconds': max(0, horizon - tail_start + 1) if horizon is not None else 0,
            'target_chunks': len(chunks), 'response_count': len(targets)}


def second_target(schedule, second, tokenizer, config, question_sent):
    """参考回复chunk优先；问题输入后其余所有空闲秒监督listen，包括窗口内。"""
    if second in schedule['chunks']:
        return schedule['chunks'][second]
    supervised = bool(config['supervise_silence'] and question_sent)
    return [tokenizer.convert_tokens_to_ids('<|listen|>')], [int(supervised)]


def supervision_metadata(schedule, second, ids, kinds, tokenizer):
    """Fine roles: before/after silence, first/later onset, continue/end/marker.

    Event IDs follow scheduled chronological order, not source response indices.
    Idle and masked positions have event -1; every response chunk shares its ID.
    """
    event = next((i for i, t in enumerate(schedule['targets'])
                  if t['onset_seconds'] <= second <= t['end_seconds']), -1)
    roles, events = [], []
    speak = tokenizer.convert_tokens_to_ids('<|speak|>')
    end = tokenizer.convert_tokens_to_ids('<|turn_eos|>')
    for offset, (token, kind) in enumerate(zip(ids, kinds)):
        role = 0
        if kind == 1:
            if event < 0:
                role = 1 if second < schedule['targets'][0]['onset_seconds'] else 2
            elif token == speak:
                fresh = second == schedule['targets'][event]['onset_seconds'] and offset == 0
                role = (3 if event == 0 else 4) if fresh else 5
            else:
                role = 6 if token == end else 7
        roles.append(role)
        events.append(event if kind else -1)
    return roles, events


def schedule_stats(schedule):
    return {k: schedule[k] for k in ('targets', 'response_count', 'target_chunks', 'tail_silence_seconds')}
