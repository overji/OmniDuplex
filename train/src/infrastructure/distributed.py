"""分布式状态与诊断：rank一致性、随机状态恢复及ZeRO分片梯度范数。"""

import os
import random
import torch


def all_true(value, device):
    """所有rank必须以相同顺序调用；仅全部rank同意时才进入下一步反向或更新。"""
    flag = torch.tensor(int(value), device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def rng_state():
    """记录Python、NumPy、CPU及当前rank CUDA随机状态，供checkpoint精确恢复。"""
    import numpy as np
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    """恢复当前rank的采样状态，不重新设置seed以免改变续训轨迹。"""
    import numpy as np
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def gradient_norms(model):
    from deepspeed.utils import safe_get_local_grad
    result = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = name.split('.')[0]
        grad = safe_get_local_grad(parameter)
        if grad is not None:
            result[group] = result.get(group, 0.) + grad.float().square().sum().item()
    for group in sorted(result):
        value = torch.tensor(result[group], device=torch.cuda.current_device(), dtype=torch.float64)
        torch.distributed.all_reduce(value)
        result[group] = value.sqrt().item()
    return result


def initialize_distributed(args, config):
    """初始化单机rank设备与NCCL，返回(device, rank, world_size)。不创建训练输出或修改超参数。"""
    import deepspeed
    local_rank = int(os.environ.get('LOCAL_RANK', max(args.local_rank, 0)))
    device = torch.device(f'cuda:{local_rank}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.cuda.set_device(device)
    # Also permit a foreground single-process invocation without torchrun.
    if 'RANK' not in os.environ:
        os.environ.update(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0',
                          MASTER_ADDR='127.0.0.1', MASTER_PORT=str(config.get('master_port', 29547)))
    deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False)
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    return device, rank, world
