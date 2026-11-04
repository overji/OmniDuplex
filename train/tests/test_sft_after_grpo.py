import json
import pytest
from train.scripts.sft_after_grpo import completed_grpo


def test_sft_requires_every_rank_and_complete_checkpoint(tmp_path):
    assert not completed_grpo(tmp_path, 2, 20)
    for rank in range(2):
        (tmp_path / f'memory_report.rank{rank}.json').write_text(json.dumps(
            dict(optimizer_steps=20, checkpoint_saved=True, last_loss=.1)))
    assert not completed_grpo(tmp_path, 2, 20)
    checkpoint = tmp_path / 'checkpoint-final'
    (checkpoint / 'state').mkdir(parents=True)
    paths = ['training_config.json', 'state/mp_rank_00_model_states.pt']
    for rank in range(2):
        paths += [f'rng_rank{rank}.pt', f'state/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt']
    for path in paths:
        (checkpoint / path).write_text('test')
    assert completed_grpo(tmp_path, 2, 20)
    (tmp_path / 'memory_report.rank1.json').write_text(json.dumps(
        dict(optimizer_steps=19, checkpoint_saved=True, last_loss=.1)))
    with pytest.raises(RuntimeError, match='requested finite updates'):
        completed_grpo(tmp_path, 2, 20)
