"""Prefix-only pilot records. Original videos are split before QA selection."""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import reference_windows
from train.src.agent import training_sample, training_record, require_protocol, PROTOCOL

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    config = json.loads(Path(path).read_text())
    method = config.setdefault('training_method', 'lora_sft')
    if method not in ('lora_sft', 'full_sft', 'full_grpo'):
        raise ValueError('Unsupported training_method')
    require_protocol(config)
    paths = ['model_path', 'videos_dir', 'media_cache', 'system_prompt', 'agent_prompt']
    if config.get('data'):
        paths.append('data')
    for key in paths:
        p = Path(config[key]).expanduser()
        config[key] = str(p if p.is_absolute() else ROOT / p)
    if config.get('deepspeed_config'):
        p = Path(config['deepspeed_config'])
        config['deepspeed_config'] = str(p if p.is_absolute() else ROOT / p)
    for key in ('max_prefix_seconds', 'max_sequence_tokens', 'lora_rank', 'lora_alpha',
                'gradient_accumulation_steps', 'epochs', 'loss_projection_chunk'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['max_speak_tokens_per_chunk'] < 4:
        raise ValueError('Chunk budget must fit speak, text, turn and chunk terminators')
    if method == 'full_grpo':
        if type(config.get('group_size', 4)) is not int or config.get('group_size', 4) < 2 or config.get('kl_beta', 0) != 0:
            raise ValueError('GRPO requires group_size >= 2 and kl_beta=0 in this implementation')
        if config.get('temperature', 1.) != 1.:
            raise ValueError('Native two-stage GRPO replay currently requires temperature=1')
        if not 0 < config.get('clip_epsilon', .2) < 1 or config.get('replay_logprob_tolerance', .15) <= 0:
            raise ValueError('Invalid GRPO clip or replay tolerance')
    if method != 'lora_sft' and not config.get('deepspeed_config'):
        raise ValueError('Full training requires deepspeed_config')
    return config


def first_record(sample, index, config):
    sample = training_sample(sample)
    refs = reference_windows(sample)
    if not refs:
        return None, 'no_scored_window'
    ri, ref = min(refs, key=lambda x: seconds(x[1]['window_start']))
    question = seconds(sample['question_time'])
    evidence_time = max([seconds(e.get('available_time', e.get('time', 0)))
                         for e in ref.get('evidence', [])] or [0])
    confirmed = (ref.get('trigger') or {}).get('confirmed_time')
    earliest = max(seconds(ref['window_start']), math.ceil(question) + 1, evidence_time,
                   seconds(confirmed) if confirmed is not None else 0)
    preferred = seconds(ref.get('trigger_time', ref['window_start']))
    onset = math.ceil(max(earliest, preferred))
    if onset > seconds(ref['window_end']):
        return None, 'no_causal_integer_onset'
    if onset > config['max_prefix_seconds']:
        return None, 'prefix_too_long'
    if not (Path(config['videos_dir']) / sample['video']).is_file():
        return None, 'missing_video'
    return {'sample_index': index, 'response_index': ri, 'onset_seconds': onset,
            'sample': sample, 'reference': ref}, None


def prepare(data, output, config, max_samples, validation_fraction):
    if not 0 < validation_fraction < 1 or max_samples < 2:
        raise ValueError('Need >=2 samples and validation fraction between 0 and 1')
    if output.exists():
        raise ValueError('Use a new prepared-data directory')
    samples = json.loads(data.read_text())
    # All QA on the same UUID video stay together. Source aliases must be deduplicated upstream.
    videos = sorted({s['video'] for s in samples})
    random.Random(config['seed']).shuffle(videos)
    nval = max(1, round(len(videos) * validation_fraction))
    validation = set(videos[:nval])
    buckets = {'train': defaultdict(list), 'validation': defaultdict(list)}
    skipped = Counter()
    for i, sample in enumerate(samples):
        record, reason = first_record(sample, i, config)
        if record and config.get('training_method') == 'full_grpo':
            refs = reference_windows(sample)
            if any(seconds(r['window_end']) > config['max_prefix_seconds'] for _, r in refs):
                record, reason = None, 'window_exceeds_rollout_horizon'
            elif any(seconds(r['window_end']) < math.ceil(seconds(sample['question_time'])) + 1 for _, r in refs):
                record, reason = None, 'noncausal_window'
        if reason:
            skipped[reason] += 1
            continue
        split = 'validation' if sample['video'] in validation else 'train'
        buckets[split][sample['mode']].append(record)
    selected = {}
    for split, groups in buckets.items():
        chosen = []
        rng = random.Random(config['seed'])
        for group in groups.values():
            rng.shuffle(group)
        limit = max_samples if split == 'train' else max(1, round(max_samples * validation_fraction))
        while len(chosen) < limit and any(groups.values()):
            for mode in sorted(groups):
                if groups[mode] and len(chosen) < limit:
                    chosen.append(groups[mode].pop())
        selected[split] = chosen
    if not all(selected.values()):
        raise ValueError('No eligible train/validation records; increase prefix limit or use more videos')
    output.mkdir(parents=True)
    for split, records in selected.items():
        (output / f'{split}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    manifest = {'protocol': PROTOCOL, 'source': str(data.resolve()), 'source_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
        'config': config, 'counts': {s: len(r) for s, r in selected.items()}, 'skipped': dict(skipped),
        'videos': {s: sorted({r['sample']['video'] for r in rows}) for s, rows in selected.items()},
        'scope': ('All scored windows within full rollout horizon. No tool results.'
                  if config.get('training_method') == 'full_grpo' else
                  'One earliest scored response per QA; full prefix from time zero. No tool results.')}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ('counts', 'skipped', 'scope')}, indent=2))


def target_chunks(record, tokenizer, budget, *, protocol=PROTOCOL):
    """Teacher-forced text with native speak/turn/chunk markers, split across seconds."""
    require_protocol({'agent_output_protocol': protocol})
    record = training_record(record)
    reference, sample = record['reference'], record['sample']
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    if sample['mode'] == 'agent':
        calls = reference.get('tool_calls') or []
        kind = sample.get('sample_type', sample.get('generation_context', {}).get('sample_type'))
        if kind in ('missing_tool', 'no_context') and calls:
            raise ValueError('Negative Agent reference must not contain a tool call')
        calls = json.loads(json.dumps(calls))
        for call in calls:
            args = call['function'].get('arguments')
            if isinstance(args, str):
                call['function']['arguments'] = json.loads(args)
        from benchmark.data.agent_token import TOKEN
        parts = [(encode(reference['response']), 2)]
        if calls:
            token_id = tokenizer.convert_tokens_to_ids(TOKEN)
            if encode(TOKEN) != [token_id] or TOKEN not in tokenizer.get_vocab():
                raise ValueError('Missing atomic Agent token')
            parts.extend([([token_id], 1),
                          (encode(json.dumps(calls[0]['function'], ensure_ascii=False)), 3)])
    else:
        parts = [(encode(reference['response']), 2)]
    body = [(token, group) for tokens, group in parts for token in tokens]
    chunks = []
    while body:
        take, body = body[:budget - 3], body[budget - 3:]
        ids = [tokenizer.convert_tokens_to_ids('<|speak|>')] + [x[0] for x in take]
        groups = [1] + [x[1] for x in take]
        if not body:
            ids.append(tokenizer.convert_tokens_to_ids('<|turn_eos|>')); groups.append(1)
        # Native runtime inserts chunk_eos on budget exhaustion. Keep it as context, not a target.
        ids.append(tokenizer.convert_tokens_to_ids('<|chunk_eos|>')); groups.append(0)
        chunks.append((ids, groups))
    if not chunks:
        raise ValueError('Empty training target')
    return chunks
