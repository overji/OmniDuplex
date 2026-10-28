"""Opt-in two-GPU ZeRO-2 check against single-process averaged QA gradients."""
import copy
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import deepspeed
import torch

from train.src.models.graph import TrainGraph
from train.src.training.data_parallel import step
from train.tests.helpers import TinyMedia, TinyDuplex


@torch.no_grad()
def prefill(self, text_list=None, **kwargs):
    if text_list:
        self.decoder.feed(self.decoder.embed_token(self.tokenizer.encode(text_list[0])))
    else:
        p = self.model.vpm.weight
        x = torch.ones(1, 3, device=p.device, dtype=p.dtype)
        self.decoder.feed(self.model.resampler(self.model.vpm(x)))
        self.decoder.feed(self.model.audio_projection_layer(self.model.apm(x)))
    return {'success': True}


def main():
    accumulation = int(os.environ.get('DP_TEST_ACCUMULATION', '1'))
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    device = torch.device('cuda', rank)
    deepspeed.init_distributed('nccl', auto_mpi_discovery=False, timeout=timedelta(seconds=90))
    TinyDuplex.streaming_prefill = prefill
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, 'video.mp4').touch()
        config = dict(videos_dir=directory, max_prefix_seconds=32, max_sequence_tokens=1024,
                      max_speak_tokens_per_chunk=20, agent_output_protocol='task_start_token_sft_v1',
                      supervise_silence=True, decision_weight=1., content_weight=1., call_weight=1.,
                      loss_projection_chunk=2, sft_activation_offload=True,
                      sft_activation_gpu_budget_gib=.01)
        base = TinyMedia()
        reference = TrainGraph(copy.deepcopy(base), TinyDuplex(copy.deepcopy(base)), config)
        # Keep the duplex and graph bound to the same reference model.
        reference.duplex = TinyDuplex(reference.model)
        reference.eval()
        graph = TrainGraph(base.to(device), TinyDuplex(base), config)
        optimizer = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
        dsconfig = json.loads(Path('train/configs/zero2_cpu.json').read_text())
        dsconfig.update(bf16={'enabled': False}, gradient_clipping=0., gradient_accumulation_steps=accumulation)
        config['gradient_accumulation_steps'] = accumulation
        dsconfig['optimizer']['params'].update(lr=.001)
        engine, _, _, _ = deepspeed.initialize(model=graph,
            model_parameters=[p for p in graph.parameters() if p.requires_grad], config=dsconfig)
        from train.src.infrastructure.zero import preserve_cpu_accumulation
        preserve_cpu_accumulation(engine)
        engine.eval()
        reference.to(device)
        records = []
        for i in range(2):
            ref = dict(response='a' if i == 0 else 'b', window_start=f'00:0{i+1}', window_end='00:04')
            records.append(dict(sample_index=i, sample=dict(mode='rtd', question_time='00:00', video='video.mp4', responses=[ref]),
                                targets=[dict(response_index=0, onset_seconds=i+1, reference=ref)]))
        def clip(record, cfg, horizon):
            return SimpleNamespace(chunks=[{}] * horizon, horizon=horizon, duration=4, question_time=0,
                                   preprocessing_cache=lambda p: nullcontext())
        graph.padding_record = records[0]
        checks = []
        with patch('train.src.models.streaming.user_request', return_value='question'), \
             patch('train.src.models.streaming.system_prompt', return_value='system'), \
             patch('train.src.training.data_parallel.Clip', side_effect=clip):
            for iteration in range(2):
                optimizer.zero_grad(set_to_none=True)
                # With accumulation>1 the final microbatch is all padding. It must
                # still finish the update, with the same zero contribution as reference.
                windows = [(records if micro == 0 else [records[(micro + iteration) % 2]])
                           if accumulation > 1 and micro < accumulation - 1 else
                           (records if iteration == 0 else records[:1]) if accumulation == 1 else []
                           for micro in range(accumulation)]
                expected_loss = sum(sum(reference(r, clip(r, config, r['targets'][0]['onset_seconds']))
                                         for r in wanted) / len(wanted)
                                    for wanted in windows if wanted) / accumulation
                expected_loss.backward(); optimizer.step()
                for micro, wanted in enumerate(windows):
                    record = wanted[rank] if rank < len(wanted) else None
                    value, info = step(engine, record, config, device, 2)
                    if micro == accumulation - 1:
                        from deepspeed.utils import safe_get_full_grad
                        for (name, parameter), (_, other) in zip(graph.named_parameters(), reference.named_parameters()):
                            if parameter.requires_grad:
                                actual_grad = safe_get_full_grad(parameter)
                                torch.testing.assert_close(actual_grad, other.grad, atol=2e-6, rtol=2e-4, msg=lambda message: name + '\n' + message)
                    before_step = engine.global_steps
                    engine.step()
                    assert engine.global_steps == before_step + int(micro == accumulation - 1)
                    assert info['valid_qa'] == len(wanted)
                assert info['global_batch_size'] == 2 * accumulation
                max_error = 0.
                for (name, parameter), (_, expected) in zip(graph.named_parameters(), reference.named_parameters()):
                    error = (parameter.detach().cpu() - expected.detach().cpu()).abs().max().item()
                    max_error = max(max_error, error)
                    torch.testing.assert_close(parameter.detach().cpu(), expected.detach().cpu(), atol=3e-5, rtol=3e-4, msg=lambda message: name + '\n' + message)
                checks.append(dict(accumulation=accumulation, optimizer_step=engine.global_steps, last_valid_qa=info['valid_qa'], max_weight_error=max_error))
        if rank == 0:
            print('DP_CUDA_EQUIVALENCE ' + json.dumps(checks), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
