"""Select distinct suitable abilities, then independently sample their modes."""
import random
from .catalog import ABILITIES, split_task
from .config import PIPELINE_ABILITY_COUNT_WEIGHTS, validate_agent_positive_probability


def ability_counts(task_counts):
    counts = dict.fromkeys(ABILITIES, 0)
    for task, count in task_counts.items():
        ability, _ = split_task(task)
        counts[ability] += count
    return counts


def select_tasks(ranked, counts, *, seed, pr_probability, agent_probability, agent_positive_probability):
    validate_agent_positive_probability(agent_positive_probability)
    rng = random.Random(seed)
    decisions = []
    totals = ability_counts(counts)
    sampled_count = rng.choices(range(1, 9), weights=PIPELINE_ABILITY_COUNT_WEIGHTS, k=1)[0] if ranked else 0
    remaining = list(ranked)
    abilities = []
    draws = []
    for _ in range(min(sampled_count, len(remaining))):
        candidates = [dict(item, count=totals[item['task']]) for item in remaining]
        weights = [1 / (item["rank"] * (item["count"] + 1)) for item in candidates]
        ability = rng.choices(candidates, weights=weights, k=1)[0]['task']
        abilities.append(ability)
        draws.append({'ability': ability, 'candidates': candidates, 'weights': weights})
        remaining = [item for item in remaining if item['task'] != ability]

    for ability in abilities:
        mode = 'pr' if rng.random() < pr_probability else 'rtd'
        decisions.append({'task': f'{ability}_{mode}', 'branch': None})
        if rng.random() < agent_probability:
            branch = 'positive'
            decisions.append({'task': f'{ability}_agent', 'branch': branch})
    return {'tasks': decisions, 'agent_selected': any(item['branch'] is not None for item in decisions),
            'sampled_ability_count': sampled_count, 'abilities': abilities, 'ability_draws': draws}
