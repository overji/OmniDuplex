import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark.data.files import read_jsonl
from benchmark.data.types import SampleResult
from benchmark.runtime.runner import run, select_samples


class FakeAdapter:
    def __init__(self, *, fail_index=None, partial=False, strict=False):
        self.fail_index = fail_index
        self.partial = partial
        self.STRICT_CONFIG = strict
        self.SELECT_BEFORE_RESUME = strict
        self.loaded = 0
        self.calls = []
        self.contexts = []

    def run_config(self, args, base):
        return {'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'prompt': base}

    def load(self, args):
        self.loaded += 1
        return None

    def run_sample(self, args, state, context):
        self.calls.append(context.index)
        self.contexts.append(context)
        context.build_prompt('Native protocol')
        with context.open_raw('.json') as raw:
            json.dump({'sample_index': context.index}, raw)
        if context.index == self.fail_index:
            raise RuntimeError('Interrupted inference')
        return SampleResult([{'onset_seconds': 2.0, 'text': 'reply'}],
                            3.0 if self.partial else 5.0,
                            'partial' if self.partial else 'complete',
                            include_sample_type=not self.STRICT_CONFIG)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.samples = [dict(task=f'task{i}', mode='agent', ability='delegation',
                             video=f'{i}.mp4', question='Please act', question_time='00:01',
                             sample_type='no_context', available_tools=[],
                             responses=[dict(window_start='00:02', window_end='00:03',
                                             response='secret')],
                             generation_context={'secret': True}) for i in range(3)]
        (root / 'data.json').write_text(json.dumps(self.samples))
        (root / 'prompt.txt').write_text('Base prompt')
        self.args = SimpleNamespace(data=root / 'data.json', system_prompt=root / 'prompt.txt',
                                    output_dir=root / 'results', cache_dir=root / 'cache',
                                    max_samples=0, sample_start=0, sample_step=1)

    def execute(self, adapter):
        with contextlib.redirect_stdout(io.StringIO()):
            run(self.args, adapter)

    def test_first_run_persists_predictions_prompts_raw_and_metrics(self):
        adapter = FakeAdapter()
        self.execute(adapter)
        predictions = read_jsonl(self.args.output_dir / 'predictions.jsonl')
        prompts = read_jsonl(self.args.output_dir / 'sample_system_prompts.jsonl')
        self.assertEqual(adapter.calls, [0, 1, 2])
        self.assertEqual(adapter.loaded, 1)
        self.assertEqual(len(prompts), 3)
        self.assertEqual(predictions[0]['system_prompt_sha256'],
                         hashlib.sha256(prompts[0]['system_prompt'].encode()).hexdigest())
        self.assertEqual(predictions[0]['sample_type'], 'no_context')
        self.assertTrue((self.args.output_dir / 'sample_000.json').exists())
        metrics = json.loads((self.args.output_dir / 'metrics.json').read_text())
        self.assertEqual(metrics['evaluated_samples'], 3)
        self.assertEqual(metrics['groups']['overall']['hits'], 3)
        for context in adapter.contexts:
            self.assertNotIn('responses', context.sample)
            self.assertNotIn('generation_context', context.sample)

    def test_resume_skips_completed_samples_without_loading_model(self):
        self.execute(FakeAdapter())
        original = (self.args.output_dir / 'predictions.jsonl').read_bytes()
        adapter = FakeAdapter()
        self.execute(adapter)
        self.assertEqual(adapter.loaded, 0)
        self.assertEqual((self.args.output_dir / 'predictions.jsonl').read_bytes(), original)

    def test_interrupted_sample_restarts_and_completed_predictions_are_preserved(self):
        with self.assertRaisesRegex(RuntimeError, 'Interrupted'):
            self.execute(FakeAdapter(fail_index=1))
        first = read_jsonl(self.args.output_dir / 'predictions.jsonl')
        self.assertEqual([item['sample_index'] for item in first], [0])
        metrics = json.loads((self.args.output_dir / 'metrics.json').read_text())
        self.assertEqual(metrics['evaluated_samples'], 1)
        adapter = FakeAdapter()
        self.execute(adapter)
        self.assertEqual(adapter.calls, [1, 2])
        predictions = read_jsonl(self.args.output_dir / 'predictions.jsonl')
        self.assertEqual(predictions[0], first[0])
        self.assertEqual([item['sample_index'] for item in predictions], [0, 1, 2])

    def test_partial_results_keep_existing_completion_and_scoring_semantics(self):
        self.args.max_samples = 1
        self.execute(FakeAdapter(partial=True, strict=True))
        prediction = read_jsonl(self.args.output_dir / 'predictions.jsonl')[0]
        self.assertEqual(prediction['status'], 'partial')
        self.assertEqual(prediction['observed_seconds'], 3.0)
        self.assertNotIn('sample_type', prediction)
        adapter = FakeAdapter(strict=True)
        self.execute(adapter)
        self.assertEqual(adapter.loaded, 0)
        self.assertEqual(len(read_jsonl(self.args.output_dir / 'predictions.jsonl')), 1)

    def test_configuration_policy_remains_model_specific(self):
        self.execute(FakeAdapter(strict=True))
        self.args.system_prompt.write_text('Changed')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            self.execute(FakeAdapter(strict=True))
        self.execute(FakeAdapter())
        config = json.loads((self.args.output_dir / 'run_config.json').read_text())
        self.assertEqual(config['prompt'], 'Changed')

    def test_model_specific_limit_and_shard_selection(self):
        self.args.max_samples = 1
        self.assertEqual(select_samples(self.samples, {0}, self.args, before_resume=True), [])
        self.assertEqual([i for i, _ in select_samples(self.samples, {0}, self.args)], [1])
        self.args.max_samples = 0
        self.args.sample_start, self.args.sample_step = 1, 2
        self.assertEqual([i for i, _ in select_samples(self.samples, set(), self.args,
                                                     before_resume=True)], [1])
        self.args.sample_step = 0
        with self.assertRaises(ValueError):
            select_samples(self.samples, set(), self.args, before_resume=True)


if __name__ == '__main__':
    unittest.main()
