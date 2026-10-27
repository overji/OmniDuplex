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
        if weight.ds_status.name == 'AVAILABLE':
            return native(*args, **kwargs)
        with deepspeed.zero.GatheredParameters([weight]):
            return native(*args, **kwargs)
    model.get_audio_embedding_streaming = audio
