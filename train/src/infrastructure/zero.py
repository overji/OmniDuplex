"""原生媒体模块绕过forward读取权重时，显式告知ZeRO参数生命周期。"""


def register_external_parameters(graph, model):
    import torch
    import deepspeed
    for module in graph.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            for parameter in module.out_proj.parameters():
                deepspeed.zero.register_external_parameter(module, parameter)
    positions = getattr(getattr(model, 'apm', None), 'embed_positions', None)
    if positions is not None:
        deepspeed.zero.register_external_parameter(model.apm, positions.weight)
        deepspeed.zero.register_external_parameter(graph, positions.weight)


def wrap_audio_position_table(model):
    import deepspeed
    positions = getattr(getattr(model, 'apm', None), 'embed_positions', None)
    if positions is None:
        return
    weight = positions.weight
    native = model.get_audio_embedding_streaming

    def audio(*args, **kwargs):
        if not hasattr(weight, 'ds_status') or weight.ds_status.name == 'AVAILABLE':
            return native(*args, **kwargs)
        with deepspeed.zero.GatheredParameters([weight]):
            return native(*args, **kwargs)
    model.get_audio_embedding_streaming = audio


def preserve_cpu_accumulation(engine):
    """Persist each accumulated microgradient in the installed ZeRO-2 CPU path.

    This DeepSpeed version reads its CPU cache on microsteps >0 but only writes
    it on microstep 0. Without writing it back, GAS>2 loses intermediate grads.
    Keep the fix local to this engine rather than modifying site-packages.
    """
    optimizer = engine.optimizer
    if not getattr(optimizer, 'cpu_offload', False) or engine.gradient_accumulation_steps() <= 1:
        return
    original = optimizer.async_accumulate_grad_in_cpu_via_gpu

    def accumulate(param):
        original(param)
        if optimizer.micro_step_id > 0:
            gradient = optimizer.get_param_gradient_attribute(param).data.view(-1)
            cache = optimizer.accumulated_grads_in_cpu[optimizer.get_param_id(param)]
            if optimizer.fp16_master_weights_and_gradients:
                _, offset, _, count = optimizer.grad_position[optimizer.get_param_id(param)]
                gradient = gradient.narrow(0, offset, count)
            cache.copy_(gradient, non_blocking=True)

    optimizer.async_accumulate_grad_in_cpu_via_gpu = accumulate
