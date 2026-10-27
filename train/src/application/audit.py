"""CPU审计用例：真实tokenizer调度与视频时长校验，不加载模型权重或媒体编码。"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from benchmark.data.timestamps import seconds
from benchmark.data.prompts import user_request
from benchmark.evaluation.timing import reference_windows
from train.src.domain.agent import system_prompt, RECORD_FORMAT
from train.src.domain.records import qa_record
from train.src.domain.trajectory import compile_trajectory, second_target, TrajectoryError, schedule_stats


def audit(data, output, config, metadata=None):
    """只加载真实tokenizer，审计整个QA的回复调度与视频末尾；文本token下界不包含媒体，不能据此保证GPU显存。"""
    from transformers import AutoTokenizer
    from train.src.infrastructure.config import model_alias
    from train.src.infrastructure.checkpoints import tokenizer_metadata
    from train.src.infrastructure.tokens import register_task_token
    alias = model_alias(config)
    path = alias if alias.exists() and alias.resolve() == Path(config['model_path']).resolve() else Path(config['model_path'])
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    register_task_token(tokenizer)
    metadata = metadata or Path(config['videos_dir']) / 'video_metadata.json'
    durations = {}
    if metadata.is_file():
        rows = json.loads(metadata.read_text())['videos']
        durations = {r['file']: float(r['duration_seconds']) for r in rows}
    data_bytes = data.read_bytes()
    samples = json.loads(data_bytes)
    accepted, excluded = [], []
    for index, sample in enumerate(samples):
        record, reason = qa_record(sample, index, config)
        if reason:
            excluded.append(dict(sample_index=index, reason=reason,
                                 response_count=len(reference_windows(sample))))
            continue
        video = sample['video']
        if video not in durations:
            durations[video] = float(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(Path(config['videos_dir']) / video)], text=True))
        try:
            schedule = compile_trajectory(record, tokenizer, config, duration=durations[video])
            # Only a lower bound: media and native media wrapper tokens are not encoded here.
            text_tokens = len(tokenizer.encode('<|im_start|>system\n' + system_prompt(sample, config) + '<|im_end|>',
                                               add_special_tokens=False))
            text_tokens += 1 + len(tokenizer.encode(user_request(sample), add_special_tokens=False))
            for second in range(1, schedule['horizon'] + 1):
                ids, _ = second_target(schedule, second, tokenizer, config,
                                       second - 1 >= seconds(sample['question_time']))
                text_tokens += len(ids) + 1
            if text_tokens > config['max_sequence_tokens']:
                raise TrajectoryError('text_only_tokens_exceed_sequence_limit')
            assert schedule['response_count'] == len(reference_windows(sample))
            accepted.append(dict(sample_index=index, mode=sample['mode'], seconds=schedule['horizon'],
                                 text_only_tokens_lower_bound=text_tokens, **schedule_stats(schedule)))
        except TrajectoryError as exc:
            excluded.append(dict(sample_index=index, reason=exc.reason, response_index=exc.response_index,
                                 response_count=len(reference_windows(sample))))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), record_format=RECORD_FORMAT,
                  source=str(data.resolve()), source_sha256=hashlib.sha256(data_bytes).hexdigest(),
                  tokenizer=tokenizer_metadata(tokenizer),
                  metadata=str(metadata), metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest() if metadata.is_file() else None,
                  config=config, source_qa=len(samples), source_responses=sum(len(reference_windows(s)) for s in samples),
                  accepted_qa=len(accepted), accepted_responses=sum(r['response_count'] for r in accepted),
                  excluded_qa=len(excluded), excluded_responses=sum(r['response_count'] for r in excluded),
                  exclusion_reasons=dict(Counter(r['reason'] for r in excluded)),
                  accepted_by_mode=dict(Counter(r['mode'] for r in accepted)),
                  full_media_sequence_lengths_checked=False, gpu_training_performed=False,
                  scope='All scored responses or exclude the entire QA. CPU tokenizer/scheduling and video-duration checks only; media token lengths and GPU memory remain unverified.',
                  accepted=accepted, excluded=excluded)
    assert result['accepted_responses'] + result['excluded_responses'] == result['source_responses']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('source_qa', 'source_responses', 'accepted_qa',
        'accepted_responses', 'excluded_qa', 'excluded_responses', 'exclusion_reasons',
        'full_media_sequence_lengths_checked')}, ensure_ascii=False, indent=2))
