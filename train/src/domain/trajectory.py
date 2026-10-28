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
        if onset > config['max_prefix_seconds']:
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
    # The closed response windows are ambiguous silence positions. Start tail
    # supervision strictly after every window and after all response chunks.
    tail_start = math.floor(max(end, *(stop for _, stop in windows))) + 1
    horizon = end
    tail = config.get('post_response_silence_seconds', 0)
    if tail:
        stop = min(tail_start + tail - 1, config['max_prefix_seconds'])
        if duration is not None:
            stop = min(stop, math.floor(duration))
        horizon = max(end, stop)
    return {'chunks': chunks, 'targets': targets, 'windows': windows,
            'horizon': horizon, 'response_end_seconds': end, 'tail_start_seconds': tail_start,
            'tail_silence_seconds': max(0, horizon - tail_start + 1),
            'target_chunks': len(chunks), 'response_count': len(targets)}


def second_target(schedule, second, tokenizer, config, question_sent):
    """先取该秒已调度的回复；空闲时仅在问题已输入且处于所有闭区间窗口之外监督listen。"""
    if second in schedule['chunks']:
        return schedule['chunks'][second]
    safe = question_sent and not any(start <= second <= end for start, end in schedule['windows'])
    supervised = bool(config['supervise_silence'] and safe)
    return [tokenizer.convert_tokens_to_ids('<|listen|>')], [int(supervised)]


def schedule_stats(schedule):
    return {k: schedule[k] for k in ('targets', 'response_count', 'target_chunks', 'tail_silence_seconds')}
