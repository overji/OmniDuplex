"""因果时间规则与QA记录构造。每条训练记录包含全部可评分response，不拆散视频。"""

import math
from pathlib import Path
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import reference_windows
from train.src.domain.agent import training_sample


def response_record(sample, index, ri, ref, config):
    """返回单条目标的最早合法整数发声秒及reference；失败返回(None, 原因)，不修改标注窗口。"""
    question = seconds(sample['question_time'])
    evidence_time = max([seconds(e.get('available_time', e.get('time', 0)))
                         for e in ref.get('evidence', [])] or [0])
    confirmed = (ref.get('trigger') or {}).get('confirmed_time')
    # 问题在[t-1,t)输入前插入，故输出秒必须至少为ceil(question)+1。
    earliest = max(seconds(ref['window_start']), math.ceil(question) + 1, evidence_time,
                   seconds(confirmed) if confirmed is not None else 0)
    preferred = seconds(ref.get('trigger_time', ref['window_start']))
    onset = math.ceil(max(earliest, preferred))
    if onset > seconds(ref['window_end']):
        return None, 'no_causal_integer_onset'
    if config.get('sft_horizon') != 'full_video' and onset > config['max_prefix_seconds']:
        return None, 'prefix_too_long'
    if not (Path(config['videos_dir']) / sample['video']).is_file():
        return None, 'missing_video'
    return {'sample_index': index, 'response_index': ri, 'onset_seconds': onset,
            'sample': sample, 'reference': ref}, None


def first_record(sample, index, config):
    """First-response coverage statistic; training uses qa_record instead."""
    sample = training_sample(sample)
    refs = reference_windows(sample)
    if not refs:
        return None, 'no_scored_window'
    ri, ref = min(refs, key=lambda x: seconds(x[1]['window_start']))
    return response_record(sample, index, ri, ref, config)


def qa_record(sample, index, config):
    """将同一QA全部可评分回复装入targets；任一回复因果条件不合法时拒绝整条QA。"""
    sample = training_sample(sample)
    kind = sample.get('sample_type', sample.get('generation_context', {}).get('sample_type'))
    if sample.get('mode') == 'agent' and kind in ('negative', 'missing_tool', 'no_context'):
        return None, 'agent_negative'
    refs = sorted(reference_windows(sample), key=lambda x: (seconds(x[1]['window_start']), x[0]))
    if not refs:
        return None, 'no_scored_window'
    targets = []
    for ri, ref in refs:
        row, reason = response_record(sample, index, ri, ref, config)
        if reason:
            return None, f'{reason}:response_index={ri}'
        targets.append({k: row[k] for k in ('response_index', 'onset_seconds', 'reference')})
    return {'sample_index': index, 'sample': sample, 'targets': targets}, None
