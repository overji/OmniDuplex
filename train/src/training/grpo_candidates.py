"""Candidate placement and CPU-only judge overlap for on-policy GRPO.

Ranks share a QA but sample different seeds under ZeRO-2. Only compact reward
telemetry crosses ranks; media tensors and action histories stay on their owner.
A bounded queue must be drained before any optimizer update changes the policy.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import torch
from train.src.domain.rewards import judged_reward


def candidate_ids(config, rank, world):
    if config.get('grpo_candidate_parallel', False):
        if world != config['group_size']:
            raise ValueError('Candidate parallel requires world_size == group_size')
        return [rank]
    return list(range(config['group_size']))


def reward_view(trajectory):
    """Exclude token histories/embeddings; keep every field used by reward backends."""
    return dict(events=trajectory['events'], chunks=[{k: c[k] for k in
                ('new_response', 'idle_wait') if k in c} for c in trajectory['chunks']])


def gather_candidates(trajectories, parallel, rank, world):
    views = [reward_view(t) for t in trajectories]
    if not parallel:
        return views if rank == 0 else None
    collected = [None] * world if rank == 0 else None
    torch.distributed.gather_object(views[0], collected, dst=0)
    return collected


def score_candidates(judge, sample, trajectories, config):
    try:
        answers = judge.score_group(sample, trajectories)
        rewards = [judged_reward(sample, t, a, config) for t, a in zip(trajectories, answers)]
        silent = sum(not t['events'] and not any(c['new_response'] for c in t['chunks'])
                     for t in trajectories)
        return dict(ok=True, rewards=rewards, silent=silent,
                    action_candidates=len(trajectories))
    except Exception as exc:
        return dict(ok=False, error_type=type(exc).__name__)


class JudgePipeline:
    """One judge group in flight while the GPU samples the next; no CUDA in threads."""
    def __init__(self, judge, config, rank):
        self.judge, self.config, self.rank = judge, config, rank
        self.queue = deque()
        self.pool = ThreadPoolExecutor(max_workers=1) if rank == 0 else None

    def submit(self, group, views):
        future = (self.pool.submit(score_candidates, self.judge, group['record']['sample'],
                                   views, self.config) if self.pool else None)
        self.queue.append((group, future))

    def pop(self):
        group, future = self.queue.popleft()
        return group, future.result() if future else None

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
