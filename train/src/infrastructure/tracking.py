"""Optional rank-zero W&B telemetry; credentials stay in the environment."""
import json


class Tracking:
    def __init__(self, run=None):
        self.run = run

    def log_train(self, record):
        if self.run is None:
            return
        values = {'train/step': record['step']}
        for key in ('loss', 'epoch', 'slot', 'sequence_tokens', 'supervised_tokens',
                    'response_count', 'tail_silence_seconds', 'step_wall_seconds', 'elapsed_seconds',
                    'peak_allocated_gib', 'activation_cpu_saved_gib_max',
                    'learning_rate', 'valid_qa', 'global_batch_size', 'forward_wall_seconds',
                    'backward_wall_seconds', 'qa_per_second', 'activation_gpu_kept_gib', 'reward_std', 'ratio_mean', 'replay_max_logprob_error',
                    'configured_qa_batch_size', 'candidate_trajectories', 'padding_slots', 'gradient_norm',
                    'reward_timing', 'reward_content', 'reward_call', 'reward_silence',
                    'all_silent_fraction', 'zero_variance_groups', 'zero_variance_group_rate', 'excluded_groups',
                    'judge_requests', 'judge_score_requests', 'judge_cache_hits', 'judge_failures', 'judge_retries',
                    'judge_failure_rate', 'judge_cache_hit_rate'):
            value = record.get(key)
            if isinstance(value, (int, float)):
                values['train/' + key] = value
        if record.get('rewards'):
            values['train/reward_mean'] = sum(record['rewards']) / len(record['rewards'])
        self.run.log(values)

    def log_validation(self, step, loss, scored, excluded):
        if self.run is not None:
            values = {'train/step': step, 'validation/scored': scored,
                      'validation/excluded': excluded}
            if loss is not None:
                values['validation/loss'] = loss
            self.run.log(values)

    def finish(self, report):
        if self.run is not None:
            self.run.summary.update(report)
            self.run.finish()


def start_tracking(config, output, rank, world_size):
    """Do not import the SDK on other ranks or without explicit configuration."""
    settings = config.get('wandb')
    if rank != 0 or not settings or settings.get('mode') == 'disabled':
        return Tracking()
    import wandb
    # Upload numeric hyperparameters, never QA, generated text, weights or arbitrary config keys.
    keys = ('training_method', 'agent_output_protocol', 'learning_rate', 'epochs',
            'gradient_accumulation_steps', 'max_prefix_seconds', 'max_sequence_tokens',
            'max_speak_tokens_per_chunk', 'supervise_silence', 'post_response_silence_seconds', 'seed', 'attention_backend',
            'decision_weight', 'content_weight', 'call_weight', 'sft_activation_offload',
            'sft_activation_cpu_budget_gib', 'sft_activation_gpu_budget_gib', 'parallel_mode',
            'reward_backend', 'group_size', 'clip_epsilon', 'kl_beta', 'temperature')
    hyperparameters = {k: config[k] for k in keys if k in config}
    hyperparameters['world_size'] = world_size
    hyperparameters['effective_batch_size'] = config.get('gradient_accumulation_steps', 1) * (world_size if config.get('parallel_mode') == 'data_parallel' else 1)
    run = wandb.init(project=settings['project'], entity=settings.get('entity'),
                     name=settings.get('name'), mode=settings.get('mode', 'online'),
                     dir=str(output), config=hyperparameters, save_code=False)
    if run is None:
        raise RuntimeError('W&B did not create a run')
    run.define_metric('train/step')
    run.define_metric('train/*', step_metric='train/step')
    run.define_metric('validation/*', step_metric='train/step')
    run.log({'train/step': 0, 'system/world_size': world_size})
    (output / 'wandb_run.json').write_text(json.dumps(
        {'id': run.id, 'url': run.url, 'project': run.project, 'entity': run.entity}, indent=2) + '\n')
    print('W&B run: ' + str(run.url), flush=True)
    return Tracking(run)
