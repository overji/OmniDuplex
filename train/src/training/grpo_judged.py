"""Reference-judged GRPO: collect valid QA groups, then update at QA boundaries."""
import json
import time
from contextlib import nullcontext

import torch

from benchmark.data.files import append_jsonl, write_json
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import reference_windows
from train.src.domain.batching import epoch_slots
from train.src.domain.grpo import advantages
from train.src.domain.rewards import judged_reward
from train.src.domain.trajectory import TrajectoryError
from train.src.infrastructure.checkpoints import save, prune_checkpoints
from train.src.infrastructure.distributed import all_true
from train.src.infrastructure.judge import Judge
from train.src.infrastructure.media import Clip
from train.src.infrastructure.offload import saved_on_cpu
from train.src.models.streaming import rollout


def broadcast(packet, rank):
    values = [packet if rank == 0 else None]
    torch.distributed.broadcast_object_list(values, src=0)
    return values[0]


def update_batch(engine, groups, config, device):
    """Average over actual QA groups and candidates, including partial tail batches.

    DeepSpeed divides backward losses by GAS. Undo that once, then divide by
    the actual number of groups. Zero-loss graph slots complete a partial GAS
    boundary without repeating a rewarded QA or dropping valid tail records.
    """
    gas, count = config['gradient_accumulation_steps'], len(groups)
    if not 0 < count <= gas or engine.micro_steps % gas:
        raise ValueError('GRPO update requires a clean accumulation boundary and 1..GAS groups')
    loss_mean, errors, ratios = 0., [], []
    stats = dict(valid_qa=count, global_batch_size=count, configured_qa_batch_size=gas,
                 candidate_trajectories=sum(len(g['trajectories']) for g in groups), padding_slots=gas-count)
    for slot in range(gas):
        padding = slot >= count
        group = groups[0] if padding else groups[slot]
        trajectories = group['trajectories'][:1] if padding else group['trajectories']
        for trajectory, advantage in zip(trajectories, [0.] if padding else group['advantages']):
            trajectory = dict(trajectory, advantage=float(advantage))
            context = (saved_on_cpu(config.get('grpo_activation_cpu_budget_gib', 128))
                       if config.get('grpo_activation_offload', False) else nullcontext())
            with context as offload:
                loss, info = engine(group['record'], group['clip'], trajectory)
                okay = bool(torch.isfinite(loss).item()) and info['replay_max_logprob_error'] <= config['replay_logprob_tolerance']
                if not all_true(okay, device):
                    raise RuntimeError('Nonfinite GRPO loss or gradient-enabled replay mismatch')
                scale = 0. if padding else gas / (count * len(trajectories))
                engine.backward(loss * scale)
            if offload is not None:
                stats['activation_cpu_saved_gib_max'] = max(stats.get('activation_cpu_saved_gib_max', 0.), offload['saved_bytes'] / 2**30)
            if not padding:
                loss_mean += loss.item() / (count * len(trajectories))
                errors.append(float(info['replay_max_logprob_error']))
                ratios.append(float(info['ratio_mean']))
        engine.step()
    if engine.micro_steps % gas:
        raise RuntimeError('GRPO update ended inside an accumulation boundary')
    stats.update(loss=loss_mean, ratio_mean=sum(ratios)/len(ratios), replay_max_logprob_error=max(errors))
    get_norm = getattr(engine, 'get_global_grad_norm', None)
    norm = get_norm() if get_norm else None
    if norm is not None:
        stats['gradient_norm'] = float(norm)
    return stats


def run(engine, model, duplex, records, args, config, device, rank, world,
        identity, tracking, started, epoch_start=0, cursor=0):
    """No service is launched here. Rank zero requires an already-running judge."""
    judge, packet = None, None
    if rank == 0:
        try:
            judge = Judge(config['judge'])
            judge.ready()
            packet = {'ok': True}
        except Exception as exc:
            packet = {'ok': False, 'error_type': type(exc).__name__}
    packet = broadcast(packet, rank)
    if not packet['ok']:
        raise RuntimeError('Judge is unavailable: ' + packet['error_type'])
    metrics = args.output / f'metrics.rank{rank}.jsonl'
    paths = args.output / f'trajectories.rank{rank}.jsonl'
    gas = config['gradient_accumulation_steps']
    limit = 1 if args.profile else args.max_steps
    stop = bool(limit and engine.global_steps >= limit)
    exclusions = zero_groups = updates = qa_total = scored_groups = sampled_total = silent_total = 0
    batch_started = time.perf_counter()
    last_epoch, last_cursor, final_loss = epoch_start, cursor, None
    torch.cuda.reset_peak_memory_stats(device)

    def flush(pending, epoch, next_cursor):
        nonlocal updates, qa_total, final_loss, stop, last_epoch, last_cursor, batch_started
        began = time.perf_counter()
        result = update_batch(engine, pending, config, device)
        rewards = [r for group in pending for r in group['rewards']]
        result['rewards'] = [r['total'] for r in rewards]
        for key in ('timing', 'content', 'call', 'silence'):
            result['reward_' + key] = sum(r[key] for r in rewards) / len(rewards)
        result.update(zero_variance_groups=zero_groups, excluded_groups=exclusions,
                      all_silent_fraction=silent_total/max(1, sampled_total),
                      zero_variance_group_rate=zero_groups/max(1, scored_groups),
                      epoch=epoch, slot=next_cursor-1, step=engine.global_steps, optimizer_boundary=True,
                      backward_wall_seconds=time.perf_counter()-began,
                      step_wall_seconds=time.perf_counter()-batch_started, elapsed_seconds=time.time()-started,
                      learning_rate=config['learning_rate'],
                      sampled_action_tokens=sum(len(c['actions']) for g in pending for t in g['trajectories'] for c in t['chunks']))
        stats = broadcast(judge.snapshot() if rank == 0 else None, rank)
        result.update({'judge_' + k: v for k, v in stats.items()})
        result['judge_failure_rate'] = stats['failures'] / max(stats.get('score_requests', stats['requests']), 1)
        result['judge_cache_hit_rate'] = stats['cache_hits'] / max(stats['cache_hits'] + stats['requests'], 1)
        append_jsonl(metrics, result)
        if rank == 0:
            print(json.dumps(result), flush=True)
            tracking.log_train(result)
        updates += 1; qa_total += len(pending); final_loss = result['loss']
        last_epoch, last_cursor = epoch, next_cursor
        if config.get('save_every_steps') and engine.global_steps % config['save_every_steps'] == 0:
            save(engine, args.output / f'checkpoint-{engine.global_steps}', epoch, next_cursor, identity, config, duplex, rank)
            prune_checkpoints(args.output, config.get('save_total_limit', 0), rank)
        requested = broadcast((args.output / 'STOP').exists() if rank == 0 else None, rank)
        stop = bool(requested or (limit and engine.global_steps >= limit))
        batch_started = time.perf_counter()

    def log_skip(epoch, slot, **extra):
        row = dict(epoch=epoch, slot=slot, step=engine.global_steps, skipped=True,
                   zero_variance_groups=zero_groups, excluded_groups=exclusions,
                   zero_variance_group_rate=zero_groups/max(1, scored_groups),
                   all_silent_fraction=silent_total/max(1, sampled_total), **extra)
        append_jsonl(metrics, row)
        if rank == 0:
            stats = judge.snapshot()
            row.update({'judge_' + k: v for k, v in stats.items()})
            row['judge_failure_rate'] = stats['failures']/max(stats.get('score_requests', stats['requests']), 1)
            tracking.log_train(row)

    for epoch in range(epoch_start, config['epochs']):
        if stop:
            break
        # No repeated tail QA and no physical-world-size multiplier: ranks share QA.
        order = epoch_slots(len(records), config['seed'], epoch, 1, 0, 1)
        pending = []
        for slot in range(cursor if epoch == epoch_start else 0, len(order)):
            record = records[order[slot]]
            engine.eval()
            trajectories, failure = [], None
            try:
                clip = Clip(record, config, config['max_prefix_seconds'])
                if any(seconds(ref['window_end']) > clip.horizon for _, ref in reference_windows(record['sample'])):
                    raise TrajectoryError('reference_window_exceeds_video_duration')
                for candidate in range(config['group_size']):
                    seed = config['seed'] + epoch*10000000 + slot*config['group_size'] + candidate
                    trajectories.append(rollout(model, duplex, record, clip, config, seed))
                # Check every candidate before spending judge calls or building gradients.
                with torch.no_grad():
                    for trajectory in trajectories:
                        _, error = engine(record, clip, trajectory)
                        if not all_true(error.item() <= config['replay_logprob_tolerance'], device):
                            raise RuntimeError('Rollout/replay probability mismatch')
            except TrajectoryError as exc:
                failure = str(exc)
            if not all_true(failure is None, device):
                exclusions += 1
                log_skip(epoch, slot, excluded=failure or 'peer_invalid_trajectory')
                continue
            sampled_total += len(trajectories)
            silent_total += sum(not t['events'] and not any(c['new_response'] for c in t['chunks']) for t in trajectories)
            packet = None
            if rank == 0:
                try:
                    answers = judge.score_group(record['sample'], trajectories)
                    rewards = [judged_reward(record['sample'], t, a, config) for t, a in zip(trajectories, answers)]
                    packet = {'ok': True, 'rewards': rewards}
                except Exception as exc:
                    packet = {'ok': False, 'error_type': type(exc).__name__}
            packet = broadcast(packet, rank)
            if not packet['ok']:
                exclusions += 1
                log_skip(epoch, slot, judge_failed=True, error_type=packet['error_type'])
                continue
            rewards = packet['rewards']
            scored_groups += 1
            adv, zero = advantages([r['total'] for r in rewards])
            for candidate, (trajectory, score) in enumerate(zip(trajectories, rewards)):
                append_jsonl(paths, dict(epoch=epoch, slot=slot, sample_index=record['sample_index'],
                                         candidate=candidate, **trajectory, reward=score))
            if zero:
                zero_groups += 1
                log_skip(epoch, slot, zero_variance=True, rewards=[r['total'] for r in rewards])
                continue
            pending.append(dict(record=record, clip=clip, trajectories=trajectories,
                                rewards=rewards, advantages=adv.tolist()))
            if len(pending) == gas:
                flush(pending, epoch, slot+1)
                pending = []
                if stop:
                    break
        if pending and not stop:
            flush(pending, epoch, len(order))
        if not stop:
            last_epoch, last_cursor = epoch, len(order)
        cursor = 0
    checkpoint_saved = not args.profile or config.get('save_profile_checkpoint', True)
    if checkpoint_saved:
        save(engine, args.output / 'checkpoint-final', last_epoch, last_cursor, identity, config, duplex, rank)
    report = dict(optimizer_steps=engine.global_steps, updates_this_run=updates, valid_qa_this_run=qa_total,
                  excluded=exclusions, zero_variance_groups=zero_groups, last_loss=final_loss,
                  checkpoint_saved=checkpoint_saved, world_size=world, training_method='full_grpo',
                  reward_backend='llm', elapsed_seconds=time.time()-started,
                  training_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
    write_json(args.output / f'memory_report.rank{rank}.json', report)
    if rank == 0:
        tracking.finish(report)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
