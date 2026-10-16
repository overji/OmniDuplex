"""Opt-in CUDA integration: RUN_ZERO_CHECKPOINT_TEST=1 python -m unittest ..."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch


@unittest.skipUnless(os.environ.get('RUN_ZERO_CHECKPOINT_TEST') == '1' and torch.cuda.is_available(),
                     'Opt-in DeepSpeed CUDA checkpoint check')
class ZeroCheckpointTests(unittest.TestCase):
    def test_optimizer_cursor_and_rng_restore_match_uninterrupted_update(self):
        import deepspeed
        from train.src.full import rng_state, restore_rng
        os.environ.update(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0',
                          MASTER_ADDR='127.0.0.1', MASTER_PORT='29549')
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False)
        config = json.loads((Path(__file__).parents[1] / 'configs/zero3_cpu.json').read_text())
        config['gradient_accumulation_steps'] = 1
        config['optimizer']['params']['lr'] = .01
        def create():
            m = torch.nn.Linear(8, 4).cuda().bfloat16()
            return deepspeed.initialize(model=m, model_parameters=m.parameters(), config=copy.deepcopy(config))[0]
        def step(engine):
            inputs = torch.randn(2, 8, device='cuda', dtype=torch.bfloat16)
            loss = engine(inputs).float().square().mean()
            engine.backward(loss); engine.step()
        def state(engine):
            with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
                return {k:v.detach().cpu().clone() for k,v in engine.module.state_dict().items()}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                original = create(); step(original)
                original.save_checkpoint(tmp, tag='state', client_state={'epoch': 0, 'cursor': 4})
                saved_rng = rng_state()
                step(original); expected = state(original)
                restored = create()
                _, client = restored.load_checkpoint(tmp, tag='state')
                self.assertEqual(client['cursor'], 4)
                restore_rng(saved_rng)
                step(restored)
                self.assertEqual(original.global_steps, restored.global_steps)
                for key, value in state(restored).items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        finally:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
