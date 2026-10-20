from collections import Counter
import unittest
from unittest.mock import patch

from data_generation.src.domain.catalog import ABILITIES, split_task
from data_generation.src.domain.selection import ability_counts, select_tasks


class SelectionTests(unittest.TestCase):
    def choose(self, seed, ranked=None, counts=None, **overrides):
        options = dict(pr_probability=0.5, agent_probability=0.75, agent_positive_probability=1.0)
        options.update(overrides)
        if ranked is None:
            ranked = [{'task': name, 'rank': index} for index, name in enumerate(ABILITIES, 1)]
        return select_tasks(ranked, counts or {}, seed=seed, **options)

    def test_count_distribution_distinct_abilities_and_repeatability(self):
        histogram = Counter()
        for seed in range(20000):
            result = self.choose(seed)
            size = len(result['abilities'])
            histogram[size] += 1
            self.assertEqual(size, result['sampled_ability_count'])
            self.assertEqual(size, len(set(result['abilities'])))
            self.assertLessEqual(size, 8)
            self.assertEqual(len(result['tasks']), len({item['task'] for item in result['tasks']}))
        self.assertEqual(set(histogram), set(range(1, 9)))
        self.assertAlmostEqual(sum(size * count for size, count in histogram.items()) / 20000, 1.5, delta=0.04)
        for size, probability in enumerate((.68, .22, .07, .01, .005, .005, .005, .005), 1):
            self.assertAlmostEqual(histogram[size] / 20000, probability, delta=.015)
        self.assertEqual(self.choose(42), self.choose(42))

    def test_insufficient_candidates_and_empty_router(self):
        ranked = [{'task': 'object_perception', 'rank': 1}]
        for seed in range(20):
            result = self.choose(seed, ranked=ranked)
            self.assertEqual(result['abilities'], ['object_perception'])
            self.assertEqual(len(result['tasks']), 1 + int(result['agent_selected']))
        result = self.choose(42, ranked=[])
        self.assertEqual(result['abilities'], [])
        self.assertEqual(result['tasks'], [])
        self.assertEqual(result['sampled_ability_count'], 0)

    def test_modes_aggregate_into_ability_counts_and_lower_counts_get_more_weight(self):
        counts = {'object_perception_pr': 2, 'object_perception_rtd': 3, 'object_perception_agent': 4}
        totals = ability_counts(counts)
        self.assertEqual(totals['object_perception'], 9)
        self.assertEqual(totals['action_perception'], 0)
        ranked = [{'task': 'object_perception', 'rank': 1}, {'task': 'action_perception', 'rank': 2}]
        with patch('data_generation.src.domain.selection.PIPELINE_ABILITY_COUNT_WEIGHTS',
                   (0, 1, 0, 0, 0, 0, 0, 0)):
            result = self.choose(42, ranked=ranked, counts=counts)
        self.assertEqual(result['ability_draws'][0]['weights'], [.1, .5])
        self.assertEqual(result['ability_draws'][1]['candidates'][0]['task'],
                         next(name for name in totals if name in {'object_perception', 'action_perception'}
                              and name != result['abilities'][0]))

    def test_per_ability_mode_boundaries_and_default_proportions(self):
        positive = self.choose(42, pr_probability=1, agent_probability=1, agent_positive_probability=1)
        for index, ability in enumerate(positive['abilities']):
            self.assertEqual(positive['tasks'][2*index], {'task': ability+'_pr', 'branch': None})
            self.assertEqual(positive['tasks'][2*index+1], {'task': ability+'_agent', 'branch': 'positive'})
        rtd = self.choose(42, pr_probability=0, agent_probability=0)
        self.assertTrue(all(item['task'].endswith('_rtd') for item in rtd['tasks']))
        with self.assertRaisesRegex(ValueError, 'disabled'):
            self.choose(42, agent_probability=1, agent_positive_probability=0)
        tally = Counter()
        mixed_agent_selection = False
        for seed in range(5000):
            result = self.choose(seed)
            tally['abilities'] += len(result['abilities'])
            agents = 0
            for item in result['tasks']:
                _, mode = split_task(item['task'])
                tally[mode] += 1
                if mode == 'agent':
                    agents += 1
                    tally[item['branch']] += 1
            mixed_agent_selection |= 0 < agents < len(result['abilities'])
        self.assertTrue(mixed_agent_selection)
        self.assertAlmostEqual(tally['pr'] / tally['abilities'], .5, delta=.02)
        self.assertAlmostEqual(tally['agent'] / tally['abilities'], .75, delta=.02)
        for branch, probability in [('positive', 1), ('no_context', 0), ('missing_tool', 0)]:
            self.assertAlmostEqual(tally[branch] / tally['agent'], probability, delta=.02)

    def test_adaptive_counts_improve_balance_without_removing_rank_preference(self):
        adaptive = Counter()
        static = Counter()
        for seed in range(3000):
            for result, counts in [(self.choose(seed, counts=adaptive, agent_probability=0), adaptive),
                                   (self.choose(seed, agent_probability=0), static)]:
                counts.update(item['task'] for item in result['tasks'])
        adaptive_abilities = ability_counts(adaptive)
        static_abilities = ability_counts(static)
        def ratio(counts):
            return max(counts.values()) / min(counts.values())
        self.assertLess(ratio(adaptive_abilities), ratio(static_abilities))
        names = list(ABILITIES)
        self.assertGreater(adaptive_abilities[names[0]], adaptive_abilities[names[-1]])
