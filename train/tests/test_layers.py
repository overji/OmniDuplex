"""验证分层边界、公共入口和CPU可覆盖的恢复契约，不启动真实训练。"""
import ast
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch


class LayerTests(unittest.TestCase):
    def test_public_facades_forward_to_one_implementation(self):
        from train.src import agent, checkpoints, token_rows
        from train.src.domain import agent as domain
        from train.src.infrastructure import checkpoints as storage, legacy_token_rows as legacy
        for facade, implementation in ((agent, domain), (checkpoints, storage), (token_rows, legacy)):
            for name in facade.__all__:
                self.assertIs(getattr(facade, name), getattr(implementation, name))

    def test_layers_have_no_reverse_dependencies_or_cycles(self):
        root = Path(__file__).parents[1] / 'src'
        allowed = {
            'domain': {'domain'},
            'infrastructure': {'domain', 'infrastructure'},
            'models': {'domain', 'infrastructure', 'models'},
            'training': {'domain', 'infrastructure', 'models', 'training'},
            'application': {'domain', 'infrastructure', 'models', 'training', 'application'},
        }
        graph = {}
        for path in root.glob('*/*.py'):
            module = 'train.src.' + '.'.join(path.relative_to(root).with_suffix('').parts)
            layer = path.parent.name
            dependencies = set()
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('train.src.'):
                    destination = node.module.split('.')[2]
                    self.assertIn(destination, allowed[layer], (path, node.module))
                    if node.module != module:
                        dependencies.add(node.module)
            graph[module] = dependencies
        visited, visiting = set(), set()

        def visit(module):
            self.assertNotIn(module, visiting, module)
            if module in visited:
                return
            visiting.add(module)
            for dependency in graph.get(module, ()):
                visit(dependency)
            visiting.remove(module)
            visited.add(module)

        for module in graph:
            visit(module)

    def test_help_and_prepare_remain_lightweight_outside_repo(self):
        root = Path(__file__).resolve().parents[2]
        # Raising at import time catches accidental eager CUDA/backend imports,
        # even on a host where those dependencies happen to be installed.
        code = '''
import builtins, json, sys
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'deepspeed', 'peft', 'transformers'}:
        raise AssertionError('Heavy import on CPU command: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from train.src.application.cli import main
from train.src.infrastructure.config import ROOT, load_config
config = load_config(ROOT/'train/configs/minicpmo_full_sft.json')
assert config['media_cache'] == str(ROOT/'benchmark/cache')
sys.argv = ['train', '--help']
try:
    main()
except SystemExit as exc:
    assert exc.code == 0
samples = []
for i in range(2):
    Path(str(i)+'.mp4').touch()
    samples.append(dict(mode='pr', video=str(i)+'.mp4', question_time='00:00',
        responses=[dict(response='Ready.', window_start='00:01', window_end='00:02')]))
Path('qa.json').write_text(json.dumps(samples))
config['videos_dir'] = str(Path.cwd())
Path('config.json').write_text(json.dumps(config))
sys.argv = ['train', '--config', 'config.json', 'prepare', '--data', 'qa.json',
            '--output', 'prepared', '--max-samples', '2']
main()
assert json.loads(Path('prepared/manifest.json').read_text())['record_format'] == 'qa_trajectory_v1'
'''
        env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run([sys.executable, '-c', code], cwd=directory, env=env,
                                       capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_random_state_restore_keeps_python_numpy_and_torch_sequences(self):
        from train.src.infrastructure.distributed import rng_state, restore_rng
        saved_python, saved_numpy, saved_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            fake_cuda = torch.tensor([1, 2], dtype=torch.uint8)
            with patch('torch.cuda.get_rng_state', return_value=fake_cuda), patch('torch.cuda.set_rng_state') as restore_cuda:
                state = rng_state()
                expected = (random.random(), np.random.rand(), torch.rand(3))
                random.random(); np.random.rand(); torch.rand(2)
                restore_rng(state)
                self.assertEqual(random.random(), expected[0])
                self.assertEqual(np.random.rand(), expected[1])
                torch.testing.assert_close(torch.rand(3), expected[2])
                restore_cuda.assert_called_once_with(fake_cuda)
        finally:
            random.setstate(saved_python); np.random.set_state(saved_numpy); torch.set_rng_state(saved_torch)

    def test_checkpoint_save_keeps_cursor_and_rank_state_without_cuda(self):
        from train.src.infrastructure.checkpoints import save
        engine, tokenizer = MagicMock(), MagicMock()
        config, identity, state = {'seed': 42}, {'prepared': 'digest'}, {'test': 'rng'}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'checkpoint'
            with patch('train.src.infrastructure.distributed.rng_state', return_value=state), \
                 patch('torch.save') as save_rng, patch('torch.distributed.barrier') as barrier:
                save(engine, output, 2, 17, identity, config, SimpleNamespace(tokenizer=tokenizer), 0)
                engine.save_checkpoint.assert_called_once_with(str(output), tag='state', client_state=dict(
                    epoch=2, cursor=17, identity=identity, config=config))
                save_rng.assert_called_once_with(state, output/'rng_rank0.pt')
                tokenizer.save_pretrained.assert_called_once_with(output/'tokenizer')
                barrier.assert_called_once()


if __name__ == '__main__':
    unittest.main()
