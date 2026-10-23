import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest

from benchmark.cli.main import MODELS, build_parser
from benchmark.data.paths import ROOT


class CliTests(unittest.TestCase):
    def test_model_specific_arguments_and_defaults(self):
        parser = build_parser()
        for model in MODELS:
            command = ['run', model, '--model-path', '/models/example']
            if model in ('mmduet2', 'livestar'):
                command += ['--official-code', '/code']
            if model == 'gander':
                command += ['--base-model-path', '/models/base', '--official-code', '/code']
            if model == 'roma':
                command += ['--roma-code', '/code']
            args = parser.parse_args(command)
            self.assertEqual(args.adapter_name, model)
            self.assertEqual(args.data, Path('/remote-home/yushengliao/speech/jimodels/datasets/AOI-Bench/all_qa.json'))
            self.assertEqual(args.videos_dir, ROOT / 'cache/aoi_bench_videos')
            self.assertEqual(args.max_samples, 0)
            if model == 'minicpmo':
                self.assertEqual(args.decode_mode, 'sampling')
                self.assertEqual(args.sliding_window_mode, 'off')
                self.assertEqual(args.output_dir, ROOT / 'results/minicpmo45_realtime')
            if model == 'joyai':
                self.assertEqual(args.model, 'jdopensource/JoyAI-VL-Interaction')

    def test_explicit_options_override_shared_defaults(self):
        args = build_parser().parse_args([
            'parallel', 'minicpmo', '--model-path', '/models/test', '--gpus', '4', '5',
            '--system-prompt', '/tmp/custom.txt', '--max-samples', '7', '--seed', '123',
            '--output-dir', '/tmp/new-run', '--poll-seconds', '2',
        ])
        self.assertEqual(args.system_prompt, Path('/tmp/custom.txt'))
        self.assertEqual((args.max_samples, args.seed, args.poll_seconds), (7, 123, 2))
        self.assertEqual(args.output_dir, Path('/tmp/new-run'))
        self.assertEqual(args.gpus, ['4', '5'])

    def test_package_and_default_resources_from_another_directory(self):
        script = """
from benchmark.cli.main import build_parser
from benchmark.config.paths import SYSTEM_PROMPT, MANIFESTS, DATA
args = build_parser().parse_args(['run', 'minicpmo', '--model-path', '/models/test'])
assert args.data == DATA and DATA.is_absolute()
assert args.system_prompt == SYSTEM_PROMPT and SYSTEM_PROMPT.is_file()
assert (MANIFESTS / 'model_manifest.json').is_file()
assert (MANIFESTS / 'data_manifest.json').is_file()
"""
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, PYTHONPATH=str(ROOT.parent))
            result = subprocess.run([sys.executable, '-c', script], cwd=directory,
                                    env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_and_reports_do_not_import_optional_dependencies(self):
        script = '''
import importlib.abc
import sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'numpy', 'PIL',
                                     'requests', 'huggingface_hub', 'streaming_infer'}:
            raise AssertionError('Unexpected optional dependency: ' + fullname)
sys.meta_path.insert(0, RejectOptional())
from benchmark.cli.main import main
for command in [[], ['run'], *[['run', model] for model in
                ('minicpmo', 'mmduet2', 'livestar', 'roma', 'joyai')],
                ['collect'], ['compare'], ['download'], ['queue'], ['parallel'],
                ['parallel', 'minicpmo'], ['parallel', 'joyai']]:
    try:
        main(command + ['--help'])
    except SystemExit as exc:
        assert exc.code == 0
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
