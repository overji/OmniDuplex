"""Ordinary Transformers generation; imports and weight loading are lazy."""
import math
from types import SimpleNamespace


MODEL_CLASSES = {
    'qwen3_omni': ('Qwen3OmniMoeForConditionalGeneration', 'Qwen3OmniMoeProcessor'),
    'qwen25_omni': ('Qwen2_5OmniForConditionalGeneration', 'Qwen2_5OmniProcessor'),
    'humanomni_v2': ('Qwen2_5OmniThinkerForConditionalGeneration', 'Qwen2_5OmniProcessor'),
}
PROBE_MODELS = tuple(MODEL_CLASSES) + ('minicpmo', 'ming_omni', 'qwen3_vl', 'baichuan_omni', 'omnivinci')


def load(args):
    if args.probe_model not in MODEL_CLASSES:
        from benchmark.probe import native_models
        return native_models.load(args)
    import torch
    import transformers
    from qwen_omni_utils import process_mm_info
    model_name, processor_name = MODEL_CLASSES[args.probe_model]
    dtype = 'auto' if args.dtype == 'auto' else getattr(torch, args.dtype)
    model = getattr(transformers, model_name).from_pretrained(str(args.model_path), dtype=dtype,
        device_map=args.device_map, attn_implementation=args.attn_implementation, local_files_only=True)
    if hasattr(model, 'disable_talker'):
        model.disable_talker()
    model.eval()
    processor = getattr(transformers, processor_name).from_pretrained(str(args.model_path), local_files_only=True)
    return SimpleNamespace(model=model, processor=processor, process_mm_info=process_mm_info, torch=torch)


def conversation(probe, args):
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': probe.system_prompt}]},
        {'role': 'user', 'content': [
            {'type': 'video', 'video': str(probe.video), 'fps': args.fps, 'max_pixels': args.max_pixels,
             # Do not let the utility's generic frame cap silently shorten long prefixes.
             'max_frames': max(2, math.ceil(probe.prefix_seconds * args.fps) + 2)},
            {'type': 'text', 'text': probe.user_prompt}]}]


def generated_sequences(output):
    # Qwen3 may return (thinker output, None), whereas text-only Qwen2.5
    # can return a tensor directly. Neither includes generated audio here.
    if isinstance(output, tuple):
        output = output[0]
    return output.sequences if hasattr(output, 'sequences') else output


def generate(args, state, probe):
    name = getattr(args, 'probe_model', 'qwen25_omni')
    if name not in MODEL_CLASSES:
        from benchmark.probe import native_models
        return native_models.generate(args, state, probe)
    messages = conversation(probe, args)  # Fresh conversation for EVERY pre/post request.
    processor, model = state.processor, state.model
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = state.process_mm_info(messages, use_audio_in_video=probe.has_audio)
    if videos is None or len(videos) != 1:
        raise ValueError('Expected one complete prefix video')
    minimum_frames = max(1, math.floor(probe.prefix_seconds * args.fps) - 2)
    if len(videos[0]) < minimum_frames:
        raise ValueError('Video utility downsampled below the configured probe FPS')
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
        return_tensors='pt', padding=True, use_audio_in_video=probe.has_audio,
        videos_kwargs={'fps': args.fps, 'max_pixels': args.max_pixels, 'do_sample_frames': False})
    inputs = inputs.to(model.device).to(model.dtype)
    input_length = inputs['input_ids'].shape[1]
    thinker_config = getattr(model.config, 'thinker_config', model.config)
    limit = getattr(thinker_config, 'max_position_embeddings', None)
    if limit and input_length + args.max_new_tokens > limit:
        raise ValueError('Complete probe prefix exceeds model context; input is not truncated')
    with state.torch.inference_mode():
        if name == 'humanomni_v2':
            output = model.generate(**inputs, use_audio_in_video=probe.has_audio,
                                    do_sample=False, max_new_tokens=args.max_new_tokens)
        else:
            output = model.generate(**inputs, use_audio_in_video=probe.has_audio, return_audio=False,
                thinker_do_sample=False, thinker_max_new_tokens=args.max_new_tokens)
    sequences = generated_sequences(output)
    return processor.batch_decode(sequences[:, input_length:], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()
