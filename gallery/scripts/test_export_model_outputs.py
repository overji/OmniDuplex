import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('export_model_outputs', Path(__file__).with_name('export-model-outputs.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ModelOutputExportTests(unittest.TestCase):
    def fixture(self, root):
        sample = {'mode': 'agent', 'ability': 'event', 'task': 'event_agent', 'video': 'clip.mp4',
                  'question': 'Do it', 'question_time': '00:00', 'responses': [
                      {'role': 'assistant', 'tool_calls': [{'function': {'name': 'agent_call'}}],
                       'window_start': '00:01', 'window_end': '00:02'},
                      {'role': 'assistant', 'window_start': '00:03', 'window_end': '00:04'}]}
        (root / 'data.json').write_text(json.dumps([sample]))
        run = root / 'run'
        run.mkdir()
        prediction = {'sample_index': 0, 'video': 'clip.mp4', 'task': 'event_agent', 'question': 'Do it',
                      'status': 'complete', 'events': [{'onset_seconds': 1.234, 'text': 'Calling'},
                                                      {'onset_seconds': 3.5, 'text': 'Finished'}]}
        (run / 'predictions.jsonl').write_text(json.dumps(prediction) + '\n')
        config = root / 'config.json'
        config.write_text(json.dumps({'dataset': 'data.json', 'runs': [{'id': 'm', 'name': 'Model', 'path': 'run'}]}))
        return config, prediction

    def test_official_agent_scoring_preserves_predictions_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prediction = self.fixture(root)
            before = (root / 'run/predictions.jsonl').read_bytes()
            models = module.export(config, root / 'outputs')
            exported = json.loads((root / 'outputs/results.json').read_text())
            record = exported['models'][0]['records'][0]
            self.assertEqual(record['prediction'], prediction)
            self.assertEqual((record['score']['windows'], record['score']['hits'], record['score']['extra_predictions']), (1, 1, 1))
            self.assertEqual(models[0]['overall']['timing_precision'], 0.5)
            self.assertEqual((root / 'run/predictions.jsonl').read_bytes(), before)

    def test_identity_mismatch_and_duplicate_indices_are_rejected(self):
        for duplicate in [False, True]:
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, prediction = self.fixture(root)
                if not duplicate:
                    prediction['question'] = 'Wrong sample'
                raw = json.dumps(prediction) + '\n'
                (root / 'run/predictions.jsonl').write_text(raw * (2 if duplicate else 1))
                with self.assertRaises(ValueError):
                    module.export(config, root / 'outputs')

    def test_invalid_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prediction = self.fixture(root)
            prediction['events'][0]['onset_seconds'] = -1
            (root / 'run/predictions.jsonl').write_text(json.dumps(prediction) + '\n')
            with self.assertRaises(ValueError):
                module.export(config, root / 'outputs')
