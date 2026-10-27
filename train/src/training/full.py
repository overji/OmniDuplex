"""全参SFT/GRPO训练编排。各rank处理相同QA以保持ZeRO collectives顺序，不能视作独立样本DP。"""

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
import torch
from benchmark.data.files import append_jsonl, write_json
from train.src.infrastructure.media import Clip
from train.src.models.graph import TrainGraph
from train.src.models.streaming import rollout
from train.src.domain.trajectory import compile_trajectory, TrajectoryError
from train.src.domain.grpo import advantages
from train.src.domain.rewards import reward
from train.src.infrastructure.distributed import all_true, restore_rng, gradient_norms, initialize_distributed
from train.src.infrastructure.checkpoints import fingerprints, save


def run(args, config):
    """组织ZeRO全参SFT/GRPO。各rank处理同一逻辑QA，保持条件媒体模块的collective顺序；只在完整优化器边界保存/停止。"""
    from train.src.domain.agent import validate_prepared
    validate_prepared(json.loads((args.prepared / 'manifest.json').read_text()), config)
    import deepspeed
    from transformers import set_seed
    from train.src.models.loading import load_model
    device, rank, world = initialize_distributed(args, config)
    config = dict(config)
    if args.profile:
        config['gradient_accumulation_steps'] = 1
    exists = torch.tensor(int(args.output.exists()) if rank == 0 else 0, device=device)
    torch.distributed.broadcast(exists, src=0)
    if exists.item():
        raise ValueError('Use a fresh output directory, including when resuming')
    if rank == 0:
        args.output.mkdir(parents=True)
    torch.distributed.barrier()
    torch.set_num_threads(config.get('cpu_threads', 4))
    set_seed(config['seed'])
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    for key in ('seed', 'max_prefix_seconds', 'videos_dir'):
        if manifest['config'][key] != config[key]:
            raise ValueError(f'Preparation config mismatch: {key}')
    grpo = config['training_method'] == 'full_grpo'
    if grpo and manifest['config'].get('training_method') != 'full_grpo':
        raise ValueError('GRPO requires its own prepare manifest with complete-window filtering')
    records = json.loads((args.prepared / 'train.json').read_text())
    validation = json.loads((args.prepared / 'validation.json').read_text())
    if args.profile:
        records = ([r for r in records if r['sample_index'] == args.sample_index]
                   if args.sample_index is not None else sorted(records, key=lambda r: max(t['onset_seconds'] for t in r['targets']))[:1])
        validation = []
    if not records:
        raise ValueError('Empty training set')
    identity = fingerprints(args.prepared, config)
    identity.update(world_size=world, selected_samples=[r['sample_index'] for r in records], profile=args.profile)
    if rank == 0:
        write_json(args.output / 'run_config.json', dict(config=config, identity=identity,
            world_size=world, profile=args.profile, sample_index=args.sample_index,
            max_steps=args.max_steps, resume=str(args.resume) if args.resume else None))
    started = time.time()
    model, duplex = load_model(config, device)
    graph = TrainGraph(model, duplex, config)
    from train.src.infrastructure.zero import register_external_parameters, wrap_audio_position_table
    register_external_parameters(graph, model)
    dsconfig = json.loads(Path(config['deepspeed_config']).read_text())
    dsconfig['gradient_accumulation_steps'] = config['gradient_accumulation_steps']
    dsconfig['train_micro_batch_size_per_gpu'] = 1
    dsconfig['gradient_clipping'] = config['max_grad_norm']
    dsconfig['optimizer']['params'].update(lr=config['learning_rate'], weight_decay=config['weight_decay'])
    engine, optimizer, _, _ = deepspeed.initialize(model=graph,
        model_parameters=[p for p in graph.parameters() if p.requires_grad], config=dsconfig)
    wrap_audio_position_table(model)
    # Native rollout uses cached inference; SFT explicitly disables the LLM cache.
    model.llm.config.use_cache = grpo
    epoch_start = cursor = 0
    if args.resume:
        _, state = engine.load_checkpoint(str(args.resume), tag='state')
        if state is None or state['identity'] != identity or state['config'] != config:
            raise ValueError('Resume configuration or prepared data identity mismatch')
        epoch_start, cursor = state['epoch'], state['cursor']
        restore_rng(torch.load(args.resume / f'rng_rank{rank}.pt', weights_only=False))
    metrics = args.output / f'metrics.rank{rank}.jsonl'
    trajectories_path = args.output / f'trajectories.rank{rank}.jsonl'
    torch.cuda.reset_peak_memory_stats(device)
    count, zero_groups, exclusions, final_loss = 0, 0, 0, None
    last_epoch, last_cursor = epoch_start, cursor
    limit = 1 if args.profile else args.max_steps
    gas = config['gradient_accumulation_steps']
    stop = bool(limit and engine.global_steps >= limit)
    for epoch in range(epoch_start, config['epochs']):
        if stop:
            break
        order = list(range(len(records)))
        random.Random(config['seed'] + epoch).shuffle(order)
        # Native media paths have conditional module calls. ZeRO collectives must
        # follow the same order on all ranks: shard states while replaying the same
        # logical sample/trajectory on each rank (not independent rank rollouts).
        quantum = gas
        padded = math.ceil(len(order) / quantum) * quantum
        order = (order * math.ceil(padded / len(order)))[:padded]
        local = order
        for slot in range(cursor if epoch == epoch_start else 0, len(local)):
            step_started = time.perf_counter()
            record = records[local[slot]]
            model.eval() if grpo else model.train()
            result, eligible = {}, True
            try:
                horizon = config['max_prefix_seconds'] if grpo else compile_trajectory(
                    record, duplex.tokenizer, config)['horizon']
            except TrajectoryError as exc:
                exclusions += 1
                append_jsonl(metrics, dict(epoch=epoch, slot=slot, sample_index=record['sample_index'],
                                          skipped=True, excluded=str(exc), response_index=exc.response_index))
                last_epoch, last_cursor = epoch, slot + 1
                if args.profile:
                    raise
                continue
            clip = Clip(record, config, horizon)
            if clip.horizon < horizon and not grpo:
                eligible = False
                result['excluded'] = 'target_exceeds_video_duration'
            if grpo:
                from benchmark.evaluation.timing import reference_windows
                from benchmark.data.timestamps import seconds
                eligible = all(seconds(r['window_end']) <= clip.horizon for _, r in reference_windows(record['sample']))
                if not eligible:
                    result['excluded'] = 'reference_window_exceeds_video_duration'
            try:
                if eligible and grpo:
                    trajectories, rewards = [], []
                    for candidate in range(config['group_size']):
                        seed = config['seed'] + epoch * 10000000 + slot * config['group_size'] + candidate
                        trajectory = rollout(model, duplex, record, clip, config, seed)
                        score = reward(record['sample'], trajectory['events'], config)
                        trajectory['reward'] = score
                        if rank == 0:
                            print(json.dumps(dict(sample_index=record['sample_index'], candidate=candidate,
                                rollout_reward=score, seconds=clip.horizon)), flush=True)
                        trajectories.append(trajectory); rewards.append(score['total'])
                        append_jsonl(trajectories_path, dict(epoch=epoch, slot=slot, sample_index=record['sample_index'], **trajectory))
                    adv, zero = advantages(rewards)
                    zero_groups += int(zero)
                    eligible = not zero
                    result = dict(rewards=rewards, reward_std=torch.tensor(rewards).std(unbiased=False).item(),
                                  zero_variance=zero, advantages=adv.tolist())
                    # Even zero-reward groups must pass causal token/probability replay checks.
                    max_error = 0.
                    with torch.no_grad():
                        for trajectory in trajectories:
                            _, error = engine(record, clip, trajectory)
                            max_error = max(max_error, error.item())
                            if error > config.get('replay_logprob_tolerance', .15):
                                raise RuntimeError(f'Rollout/replay distribution mismatch: {error.item():.4f}')
                    result['replay_max_logprob_error'] = max_error
                elif eligible:
                    # Validate length before building a backward graph on any rank.
                    with torch.no_grad():
                        loss = engine(record, clip)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite SFT loss')
            except TrajectoryError as exc:
                eligible = False
                result['excluded'] = str(exc)
                result['response_index'] = exc.response_index
            if not all_true(eligible, device):
                exclusions += int(not result.get('zero_variance', False))
                # No backward on any rank. Do not advance accumulation microsteps.
                append_jsonl(metrics, dict(epoch=epoch, slot=slot, sample_index=record['sample_index'], skipped=True, **result))
                last_epoch, last_cursor = epoch, slot + 1
                continue
            # 采样、无梯度预检均成功后才进行可微回放；失败不推进累积步数。
            if grpo:
                value = 0.
                ratios, errors = [], []
                for trajectory, advantage in zip(trajectories, adv.tolist()):
                    trajectory['advantage'] = advantage
                    from train.src.infrastructure.offload import saved_on_cpu
                    context = (saved_on_cpu(config.get('grpo_activation_cpu_budget_gib', 128))
                               if config.get('grpo_activation_offload', False) else nullcontext())
                    with context as offload_stats:
                        loss, info = engine(record, clip, trajectory)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite GRPO loss')
                        replay_error = info['replay_max_logprob_error']
                        replay_ok = replay_error <= config.get('replay_logprob_tolerance', .15)
                        if not all_true(replay_ok, device):
                            raise RuntimeError(
                                'Gradient-enabled rollout/replay distribution mismatch: '
                                f'{replay_error.item():.4f}'
                            )
                        value += loss.item() / len(trajectories)
                        engine.backward(loss / len(trajectories))
                    if offload_stats is not None:
                        result['activation_cpu_saved_gib_max'] = max(result.get('activation_cpu_saved_gib_max', 0.), offload_stats['saved_bytes'] / 2**30)
                        result['parameter_offload_cache_hits'] = result.get('parameter_offload_cache_hits', 0) + offload_stats['parameter_cache_hits']
                    ratios.append(info['ratio_mean']); errors.append(info['replay_max_logprob_error'])
                result.update(ratio_mean=sum(ratios) / len(ratios), replay_max_logprob_error=max(errors),
                    sampled_action_tokens=sum(len(c['actions']) for t in trajectories for c in t['chunks']))
            else:
                from train.src.infrastructure.offload import saved_on_cpu
                context = (saved_on_cpu(config.get('sft_activation_cpu_budget_gib', 128))
                           if config.get('sft_activation_offload', False) else nullcontext())
                # Only relocate saved tensors; preserve the complete trajectory and loss.
                with context as offload_stats:
                    loss = engine(record, clip)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite SFT loss')
                    value = loss.item()
                    engine.backward(loss)
                if offload_stats is not None:
                    result['activation_cpu_saved_gib_max'] = offload_stats['saved_bytes'] / 2**30
                    result['parameter_offload_cache_hits'] = offload_stats['parameter_cache_hits']
            boundary = engine.is_gradient_accumulation_boundary()
            # DeepSpeed uses BF16 partitioned grads; collect norms before step/reset.
            if boundary and args.profile:
                result['module_gradient_norms'] = gradient_norms(model)
            engine.step()
            count += 1; final_loss = value
            result.update(epoch=epoch, slot=slot, step=engine.global_steps, loss=value,
                sample_index=record['sample_index'], optimizer_boundary=boundary,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                elapsed_seconds=time.time() - started, step_wall_seconds=time.perf_counter() - step_started,
                **graph.last_stats)
            if grpo:
                result['actions_per_second'] = result['sampled_action_tokens'] / result['step_wall_seconds']
            append_jsonl(metrics, result)
            if rank == 0:
                print(json.dumps(result), flush=True)
            last_epoch, last_cursor = epoch, slot + 1
            if boundary and config.get('save_every_steps', 0) and engine.global_steps % config['save_every_steps'] == 0:
                save(engine, args.output / f'checkpoint-{engine.global_steps}', epoch, slot + 1, identity, config, duplex, rank)
            if boundary and limit and engine.global_steps >= limit:
                stop = True
                break
        cursor = 0
        if not grpo and validation and not stop:
            model.eval()
            values, skipped_validation = [], []
            local_val = validation
            with torch.no_grad():
                for record in local_val:
                    try:
                        end = compile_trajectory(record, duplex.tokenizer, config)['horizon']
                        clip = Clip(record, config, end)
                        if clip.horizon != end:
                            raise TrajectoryError('target_exceeds_video_duration')
                        values.append(engine(record, clip).item())
                    except TrajectoryError as exc:
                        skipped_validation.append(dict(sample_index=record['sample_index'], reason=str(exc)))
            append_jsonl(metrics, dict(epoch=epoch, validation_loss=sum(values) / len(values) if values else None,
                                      validation_scored=len(values), validation_excluded=skipped_validation))
    # A skipped group can leave a partial accumulation; never silently save it as
    # an optimizer checkpoint. Persist only complete optimizer states.
    if engine.micro_steps % gas:
        raise RuntimeError('Incomplete accumulation after exclusions; reduce accumulation or use more records. No final checkpoint saved.')
    checkpoint_saved = not args.profile or config.get('save_profile_checkpoint', True)
    if checkpoint_saved:
        save(engine, args.output / 'checkpoint-final', last_epoch, last_cursor, identity, config, duplex, rank)
    report = dict(optimizer_steps=engine.global_steps, microbatches=count, zero_variance_groups=zero_groups,
        excluded=exclusions, last_loss=final_loss, elapsed_seconds=time.time() - started,
        training_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        training_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        world_size=world, training_method=config['training_method'], checkpoint_saved=checkpoint_saved,
        trainable_parameters=sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel() for p in model.parameters() if p.requires_grad))
    write_json(args.output / f'memory_report.rank{rank}.json', report)
    if rank == 0:
        print(json.dumps(report, indent=2), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
