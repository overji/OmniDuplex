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
from train.src.training.grpo_candidates import candidate_ids, gather_candidates, JudgePipeline


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
                loss, info = None, None
                try:
                    loss, info = engine(group['record'], group['clip'], trajectory)
                except TrajectoryError:
                    pass  # Every rank must reach the same validity collective.
                okay = (loss is not None and bool(torch.isfinite(loss).item())
                        and info['replay_max_logprob_error'] <= config['replay_logprob_tolerance'])
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
    ratio_mean, max_error = sum(ratios)/len(ratios), max(errors)
    if config.get('grpo_candidate_parallel', False):
        # ZeRO averages gradients over ranks. Log the same global objective,
        # not just rank zero's candidate; use MAX for the replay safety check.
        world = torch.distributed.get_world_size()
        packed = torch.tensor([loss_mean, ratio_mean], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(packed)
        loss_mean, ratio_mean = (packed / world).tolist()
        maximum = torch.tensor(max_error, device=device)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        max_error = maximum.item()
        stats['candidate_trajectories'] *= world
    stats.update(loss=loss_mean, ratio_mean=ratio_mean, replay_max_logprob_error=max_error)
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
    candidates = candidate_ids(config, rank, world)
    parallel = config.get('grpo_candidate_parallel', False)
    pipeline = JudgePipeline(judge, config, rank)
    depth = config.get('grpo_judge_pipeline_depth', 1)
    limit = 1 if args.profile else args.max_steps
    stop = bool(limit and engine.global_steps >= limit)
    exclusions = zero_groups = updates = qa_total = scored_groups = sampled_total = silent_total = 0
    batch_started = time.perf_counter()
    last_epoch, last_cursor, final_loss = epoch_start, cursor, None
    torch.cuda.reset_peak_memory_stats(device)

    def flush(pending, epoch, next_cursor):
        nonlocal updates, qa_total, final_loss, stop, last_epoch, last_cursor, batch_started
        if pipeline.queue:
            raise RuntimeError('Cannot update the policy with pending judge/rollout groups')
        began = time.perf_counter()
        result = update_batch(engine, pending, config, device)
        for key in ('rollout_wall_seconds', 'precheck_wall_seconds', 'gather_wall_seconds', 'judge_wait_wall_seconds'):
            value = torch.tensor(sum(g.get(key, 0.) for g in pending), device=device, dtype=torch.float64)
            if parallel:
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
            result[key] = value.item()
        rewards = [r for group in pending for r in group['rewards']]
        result['rewards'] = [r['total'] for r in rewards]
        for key in ('timing', 'content', 'call', 'silence', 'tp', 'fp', 'fn', 'tn',
                    'reference_windows', 'negative_seconds', 'unparsed_response_starts', 'invalid_calls',
                    'tp_contribution', 'fp_contribution', 'fn_contribution', 'tn_contribution', 'invalid_call_contribution'):
            if not all(key in r for r in rewards):
                continue
            result['reward_' + key] = sum(r[key] for r in rewards) / len(rewards)
        result.update(zero_variance_groups=zero_groups, excluded_groups=exclusions,
                      all_silent_fraction=silent_total/max(1, sampled_total),
                      zero_variance_group_rate=zero_groups/max(1, scored_groups),
                      epoch=epoch, slot=next_cursor-1, step=engine.global_steps, optimizer_boundary=True,
                      backward_wall_seconds=time.perf_counter()-began,
                      step_wall_seconds=time.perf_counter()-batch_started, elapsed_seconds=time.time()-started,
                      learning_rate=config['learning_rate'],
                      sampled_action_tokens=sum(len(c['actions']) for g in pending for t in g['trajectories'] for c in t['chunks']))
        if parallel:
            tokens = torch.tensor(result['sampled_action_tokens'], device=device, dtype=torch.long)
            torch.distributed.all_reduce(tokens)
            result['sampled_action_tokens'] = tokens.item()
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
        evaluation = config.get('grpo_evaluation')
        if not args.profile and evaluation and engine.global_steps % evaluation['every_steps'] == 0:
            from train.src.training.grpo_evaluation import evaluate
            evaluate(engine, model, duplex, args, config, rank, world)
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

    def accept(pending):
        nonlocal exclusions, sampled_total, silent_total, scored_groups, zero_groups
        waiting = time.perf_counter()
        group, packet = pipeline.pop()
        packet = broadcast(packet, rank)
        group['judge_wait_wall_seconds'] = time.perf_counter() - waiting
        if not packet['ok']:
            exclusions += 1
            log_skip(group['epoch'], group['slot'], judge_failed=True, error_type=packet['error_type'])
            return group['slot']
        rewards = packet['rewards']
        sampled_total += packet['action_candidates']
        silent_total += packet['silent']
        scored_groups += 1
        adv, zero = advantages([r['total'] for r in rewards])
        for candidate, trajectory in zip(candidates, group['trajectories']):
            append_jsonl(paths, dict(epoch=group['epoch'], slot=group['slot'],
                sample_index=group['record']['sample_index'], candidate=candidate,
                **trajectory, reward=rewards[candidate]))
        if zero:
            zero_groups += 1
            log_skip(group['epoch'], group['slot'], zero_variance=True, rewards=[r['total'] for r in rewards])
        else:
            group.update(rewards=rewards, advantages=[float(adv[i]) for i in candidates])
            pending.append(group)
        return group['slot']

    try:
        if not args.profile and config.get('grpo_evaluation'):
            if engine.global_steps:
                raise ValueError('Fixed GRPO before/after evaluation currently requires a fresh run')
            from train.src.training.grpo_evaluation import evaluate
            evaluate(engine, model, duplex, args, config, rank, world)
        for epoch in range(epoch_start, config['epochs']):
            if stop:
                break
            # All candidates in an optimizer batch are sampled by the SAME policy.
            # Queue capacity accounts for both scored and still-pending QA groups.
            order = epoch_slots(len(records), config['seed'], epoch, 1, 0, 1)
            pending = []
            for slot in range(cursor if epoch == epoch_start else 0, len(order)):
                record = records[order[slot]]
                engine.eval()
                trajectories, failure = [], None
                sampling_started = time.perf_counter()
                precheck_seconds = 0.
                try:
                    clip = Clip(record, config, None if config.get('grpo_horizon') == 'full_video' else config['max_prefix_seconds'])
                    if any(seconds(ref['window_end']) > clip.horizon for _, ref in reference_windows(record['sample'])):
                        raise TrajectoryError('reference_window_exceeds_video_duration')
                    for candidate in candidates:
                        seed = config['seed'] + epoch*10000000 + slot*config['group_size'] + candidate
                        trajectories.append(rollout(model, duplex, record, clip, config, seed))
                    rollout_seconds = time.perf_counter() - sampling_started
                except TrajectoryError as exc:
                    failure = str(exc)
                # Independent candidates may fail on only one rank. Synchronize
                # eligibility BEFORE any replay checks, keeping collective order.
                if not all_true(failure is None, device):
                    exclusions += 1
                    log_skip(epoch, slot, excluded=failure or 'peer_invalid_trajectory')
                    continue
                if rank == 0:
                    print(json.dumps(dict(stage='rollout_complete', epoch=epoch, slot=slot,
                        sample_index=record['sample_index'], candidate_count=config['group_size'],
                        rank0_rollout_seconds=rollout_seconds)), flush=True)
                precheck_started = time.perf_counter()
                if config.get('grpo_precheck_replay', True):
                    valid_replay = True
                    try:
                        with torch.no_grad():
                            for trajectory in trajectories:
                                _, error = engine(record, clip, trajectory)
                                valid_replay &= bool(torch.isfinite(error).item()) and error.item() <= config['replay_logprob_tolerance']
                    except TrajectoryError:
                        valid_replay = False
                    if not all_true(valid_replay, device):
                        raise RuntimeError('Rollout/replay probability mismatch')
                precheck_seconds = time.perf_counter() - precheck_started
                if rank == 0:
                    print(json.dumps(dict(stage='replay_precheck_complete', epoch=epoch, slot=slot,
                        sample_index=record['sample_index'], enabled=config.get('grpo_precheck_replay', True),
                        rank0_precheck_seconds=precheck_seconds)), flush=True)
                gathering = time.perf_counter()
                views = gather_candidates(trajectories, parallel, rank, world)
                pipeline.submit(dict(record=record, clip=clip, trajectories=trajectories,
                                     epoch=epoch, slot=slot, rollout_wall_seconds=rollout_seconds,
                                     precheck_wall_seconds=precheck_seconds,
                                     gather_wall_seconds=time.perf_counter()-gathering), views)
                while pipeline.queue and (len(pipeline.queue) >= depth or
                                           len(pending) + len(pipeline.queue) >= gas):
                    consumed = accept(pending)
                    if len(pending) == gas:
                        flush(pending, epoch, consumed+1)
                        pending = []
                        if stop:
                            break
                if stop:
                    break
            # Drain before the partial final update; never carry an old-policy
            # trajectory over an optimizer step, checkpoint, or epoch boundary.
            while pipeline.queue:
                accept(pending)
            if pending and not stop:
                flush(pending, epoch, len(order))
            if not stop:
                last_epoch, last_cursor = epoch, len(order)
            cursor = 0
    finally:
        pipeline.close()
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
