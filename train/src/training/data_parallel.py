"""Single-pass full SFT using DeepSpeed's native ZeRO-2 data parallel engine.

Different QA lengths never issue forward parameter all-gathers. Invalid/padding
slots contribute a zero gradient, and valid losses are normalized by valid QA.
"""
import time
from contextlib import nullcontext

import torch

from train.src.domain.trajectory import compile_trajectory, TrajectoryError
from train.src.infrastructure.distributed import all_true
from train.src.infrastructure.media import Clip
from train.src.infrastructure.offload import saved_on_cpu



def padding_qa(records, tokenizer, config):
    """Choose a genuinely schedulable short QA; never synthesize missing labels."""
    for record in sorted(records, key=lambda r: max(t['onset_seconds'] for t in r['targets'])):
        try:
            horizon = compile_trajectory(record, tokenizer, config)['horizon']
            clip = Clip(record, config, horizon)
            compile_trajectory(record, tokenizer, config, duration=clip.duration)
            return record
        except TrajectoryError:
            continue
    raise ValueError('No complete QA available for padding forward')

def step(engine, record, config, device, world_size):
    context = (saved_on_cpu(config.get('sft_activation_cpu_budget_gib', 128),
                            keep_gib=config.get('sft_activation_gpu_budget_gib', 0),
                            reserve_gib=config.get('sft_activation_gpu_reserve_gib', 12))
               if config.get('sft_activation_offload', False) else nullcontext())
    result = {}
    loss, clip, eligible = None, None, record is not None
    started = time.perf_counter()
    with context as stats:
        if eligible:
            try:
                horizon = compile_trajectory(record, engine.module.duplex.tokenizer, config)['horizon']
                clip = Clip(record, config, horizon)
                compile_trajectory(record, engine.module.duplex.tokenizer, config, duration=clip.duration)
                loss = engine(record, clip)
            except TrajectoryError as exc:
                eligible = False
                result.update(excluded=str(exc), response_index=exc.response_index)
        valid = torch.tensor(int(eligible), device=device)
        torch.distributed.all_reduce(valid)
        valid_count = int(valid.item())
        # Empty tail slots must still advance DeepSpeed's accumulation boundary.
        # Use the normal graph with zero loss on every rank, just like rank padding.
        if not eligible:
            # Every rank still participates in the same backward/update collectives.
            # A parameter-sum dummy changes ZeRO-2 bucket order. Reuse the normal
            # streaming graph, then zero its loss so gradient hooks stay aligned.
            fallback = engine.module.padding_record
            horizon = compile_trajectory(fallback, engine.module.duplex.tokenizer, config)['horizon']
            clip = Clip(fallback, config, horizon)
            compile_trajectory(fallback, engine.module.duplex.tokenizer, config, duration=clip.duration)
            loss = engine(fallback, clip) * 0
            engine.module.last_stats.update(targets=[], response_count=0, target_chunks=0,
                                            supervised_tokens=0, tail_silence_seconds=0)
        if not all_true(bool(torch.isfinite(loss).item()), device):
            raise RuntimeError('Nonfinite data-parallel SFT loss')
        local_value = loss.item()
        result['forward_wall_seconds'] = time.perf_counter() - started
        backward_started = time.perf_counter()
        # ZeRO averages across world_size; padding/invalid QA must not dilute valid QA.
        engine.backward(loss * world_size / max(valid_count, 1))
        result['backward_wall_seconds'] = time.perf_counter() - backward_started
        if stats is not None:
            result.update(activation_cpu_saved_gib_max=stats['saved_bytes'] / 2**30,
                          activation_gpu_kept_gib=stats['kept_bytes'] / 2**30,
                          parameter_offload_cache_hits=stats['parameter_cache_hits'])
    total = torch.tensor(local_value if eligible else 0., device=device, dtype=torch.float64)
    torch.distributed.all_reduce(total)
    result.update(local_loss=local_value, local_valid=eligible, valid_qa=valid_count,
                  global_micro_batch_size=world_size,
                  global_batch_size=world_size * config.get('gradient_accumulation_steps', 1),
                  sample_index=record['sample_index'] if record else -1)
    return total.item() / max(valid_count, 1), result
