"""Native non-duplex probe adapters; dependencies and weights load only on run."""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def local_alias(args):
    alias = args.cache_dir / 'model_aliases' / ('model_' + hashlib.sha256(str(args.model_path.resolve()).encode()).hexdigest()[:16])
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        try:
            alias.symlink_to(args.model_path.resolve(), target_is_directory=True)
        except FileExistsError:
            pass
    if alias.resolve() != args.model_path.resolve():
        raise ValueError('Model alias points to a different checkpoint')
    return alias


def load(args):
    import torch
    import transformers as tf
    path = local_alias(args)
    kwargs = dict(trust_remote_code=True, local_files_only=True,
                  torch_dtype='auto' if args.dtype == 'auto' else getattr(torch, args.dtype),
                  device_map=args.device_map, attn_implementation=args.attn_implementation)
    name = args.probe_model
    if name == 'qwen3_vl':
        model = tf.Qwen3VLForConditionalGeneration.from_pretrained(str(path), **kwargs)
        processor = tf.AutoProcessor.from_pretrained(str(path), local_files_only=True)
    elif name == 'ming_omni':
        if not args.official_code or not (args.official_code / 'modeling_bailingmm.py').is_file():
            raise ValueError('Ming requires --official-code pointing to inclusionAI/Ming v1.5 source')
        sys.path.insert(0, str(args.official_code.resolve()))
        cls = importlib.import_module('modeling_bailingmm').BailingMMNativeForConditionalGeneration
        config = cls.config_class.from_pretrained(str(path), local_files_only=True)
        config.talker_config = None  # Probe generates text; retain the complete audio-input encoder.
        config.llm_config._attn_implementation = args.attn_implementation
        model = cls.from_pretrained(str(path), config=config, load_image_gen=False, **kwargs)
        # Ming v1.5 resolves am.mvn relative to the process working directory.
        # Scope that directory only around processor loading, then restore it.
        previous_directory = Path.cwd()
        try:
            os.chdir(args.official_code.resolve())
            processor = tf.AutoProcessor.from_pretrained(str(args.official_code.resolve()),
                trust_remote_code=True, local_files_only=True)
        finally:
            os.chdir(previous_directory)
    elif name == 'baichuan_omni':
        model = tf.AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        processor = model.bind_processor(tf.AutoTokenizer.from_pretrained(str(path), trust_remote_code=True,
                                                                          local_files_only=True), training=False)
        chunk_size = getattr(args, 'attention_query_chunk_size', 0)
        if chunk_size:
            from benchmark.probe.attention import install_baichuan_attention
            install_baichuan_attention(model, chunk_size)
        processor.video_processor.max_pixels = args.max_pixels
        processor.visual_processor.max_pixels = args.max_pixels
    else:
        if name == 'minicpmo':
            kwargs.update(init_vision=True, init_audio=True, init_tts=False)
        model = tf.AutoModel.from_pretrained(str(path), **kwargs)
        processor = tf.AutoProcessor.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    return SimpleNamespace(model=model.eval(), processor=processor, torch=torch)


def check_context(model, length, extra):
    configs = [model.config, getattr(model.config, 'text_config', None), getattr(model.config, 'llm_config', None),
               getattr(getattr(model, 'llm', None), 'config', None)]
    limits = [getattr(c, 'max_position_embeddings', None) for c in configs if c is not None]
    limits += [getattr(model.config, 'model_max_length', None)]
    limits = [int(v) for v in limits if v and int(v) < 10**9]
    if limits and length + extra > min(limits):
        raise ValueError('Complete probe prefix exceeds model context; input is not truncated')


def audio_samples(video):
    import numpy as np
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vn', '-ac', '1',
                             '-ar', '16000', '-f', 'f32le', 'pipe:1'], check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype='<f4').copy()


def sampled_frames(probe, args):
    from decord import VideoReader, cpu
    from PIL import Image
    reader = VideoReader(str(probe.video), ctx=cpu(0))
    for index in range(math.ceil(probe.prefix_seconds * args.fps)):
        t = index / args.fps
        frame = Image.fromarray(reader[min(len(reader) - 1, int(t * reader.get_avg_fps()))].asnumpy()).convert('RGB')
        ratio = min(1.0, math.sqrt(args.max_pixels / (frame.width * frame.height)))
        if ratio < 1:
            frame = frame.resize((max(1, int(frame.width * ratio)), max(1, int(frame.height * ratio))))
        yield t, frame


def generate_minicpmo(args, state, probe):
    contents = []
    audio = audio_samples(probe.video) if probe.has_audio else None
    for t, frame in sampled_frames(probe, args):
        contents.extend([f'{t:.3f}s:', frame])
        if audio is not None:
            contents.append(audio[int(t * 16000):int(min(t + 1 / args.fps, probe.prefix_seconds) * 16000)])
    contents.append(probe.user_prompt)
    messages = [{'role': 'system', 'content': probe.system_prompt}, {'role': 'user', 'content': contents}]
    def guard(module, positional, keyword):
        tensor = keyword.get('inputs_embeds')
        if tensor is None:
            tensor = keyword.get('input_ids')
        if tensor is not None:
            check_context(state.model, tensor.shape[1], args.max_new_tokens)
    handle = state.model.llm.register_forward_pre_hook(guard, with_kwargs=True)
    try:
        result = state.model.chat(msgs=messages, processor=state.processor, generate_audio=False,
                                  omni_mode=probe.has_audio, do_sample=False, enable_thinking=False,
                                  max_new_tokens=args.max_new_tokens, max_inp_length=2**30,
                                  max_slice_nums=1, use_image_id=False)
    finally:
        handle.remove()
    return result if isinstance(result, str) else result[0]


def generate_baichuan(args, state, probe):
    if args.fps != 1:
        raise ValueError('Baichuan native decoder requires the prepared 1 FPS protocol')
    p, model = state.processor, state.model
    split = p._split_video_to_frames
    p._split_video_to_frames = lambda info, **kw: split(info, max_frame_number=max(1, math.ceil(probe.prefix_seconds)), decode_way='1fps')
    try:
        video = p.video_start_tag + json.dumps({'local': str(probe.video)}) + p.video_end_tag
        audio = ''
        if probe.has_audio:
            import soundfile as sf
            path = args.cache_dir / 'audio' / (hashlib.sha256(str(probe.video).encode()).hexdigest() + '.wav')
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                sf.write(path, audio_samples(probe.video), 16000)
            audio = p.audio_start_tag + json.dumps({'path': str(path)}) + p.audio_end_tag
        messages = [{'role': 'system', 'content': probe.system_prompt},
                    {'role': 'user', 'content': video + audio + probe.user_prompt}]
        text = p.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = p([text])
    finally:
        p._split_video_to_frames = split
    keys = ('input_ids', 'attention_mask', 'audios', 'encoder_length', 'bridge_length', 'images',
            'patch_nums', 'images_grid', 'videos', 'videos_patch_nums', 'videos_grid')
    def move(value):
        if isinstance(value, list):
            return [move(v) for v in value]
        return value.to(model.device) if hasattr(value, 'to') else value
    inputs = {key: move(getattr(batch, key)) for key in keys if getattr(batch, key, None) is not None}
    length = inputs['input_ids'].shape[1]
    check_context(model, length, args.max_new_tokens)
    output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    return p.tokenizer.decode(output[0, length:], skip_special_tokens=True)


def generate(args, state, probe):
    model, p = state.model, state.processor
    with state.torch.inference_mode():
        if args.probe_model == 'minicpmo':
            return generate_minicpmo(args, state, probe)
        if args.probe_model == 'baichuan_omni':
            return generate_baichuan(args, state, probe)
        if args.probe_model == 'qwen3_vl':
            from qwen_vl_utils import process_vision_info
            from benchmark.probe.models import conversation
            messages = conversation(probe, args)
            text = p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, video_entries = process_vision_info(messages, return_video_metadata=True,
                image_patch_size=p.image_processor.patch_size)
            videos = [entry[0] for entry in video_entries] if video_entries else None
            metadata = [entry[1] for entry in video_entries] if video_entries else None
            if not videos or len(videos[0]) < max(1, math.floor(probe.prefix_seconds * args.fps) - 2):
                raise ValueError('Visual processor dropped prefix frames')
            inputs = p(text=[text], images=images, videos=videos, return_tensors='pt',
                       videos_kwargs={'do_sample_frames': False, 'video_metadata': metadata})
        elif args.probe_model == 'ming_omni':
            content = [{'type': 'video', 'video': str(probe.video), 'fps': args.fps, 'sample': 'uniform',
                        'max_pixels': args.max_pixels, 'nframes': max(2, 2 * math.ceil(probe.prefix_seconds * args.fps / 2))}]
            if probe.has_audio:
                content.append({'type': 'audio', 'audio': str(probe.video)})
            content.append({'type': 'text', 'text': probe.system_prompt + '\n' + probe.user_prompt})
            messages = [{'role': 'HUMAN', 'content': content}]
            text = p.apply_chat_template(messages, add_generation_prompt=True)
            images, videos, audios = p.process_vision_info(messages)
            if not videos or len(videos[0]) < max(1, math.floor(probe.prefix_seconds * args.fps) - 2):
                raise ValueError('Ming processor dropped prefix frames')
            inputs = p(text=[text], images=images, videos=videos, audios=audios, return_tensors='pt',
                       audio_kwargs={'use_whisper_encoder': True})
        else:
            for config in (model.config, p.config):
                config.load_audio_in_video = probe.has_audio
                config.num_video_frames = max(1, math.ceil(probe.prefix_seconds * args.fps))
                config.audio_chunk_length = 'max_3600'
            messages = [{'role': 'user', 'content': [{'type': 'video', 'video': str(probe.video)},
                {'type': 'text', 'text': probe.system_prompt + '\n' + probe.user_prompt}]}]
            text = p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = p([text])
            check_context(model, inputs.input_ids.shape[1], args.max_new_tokens)
            def guard(module, positional, keyword):
                tensor = keyword.get('inputs_embeds')
                if tensor is None:
                    tensor = keyword.get('input_ids')
                if tensor is not None:
                    check_context(module, tensor.shape[1], args.max_new_tokens)
            handle = model.llm.register_forward_pre_hook(guard, with_kwargs=True)
            try:
                output = model.generate(input_ids=inputs.input_ids.to(model.device),
                    media=getattr(inputs, 'media', None), media_config=getattr(inputs, 'media_config', None),
                    do_sample=False, max_new_tokens=args.max_new_tokens)
            finally:
                handle.remove()
            return p.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        inputs = inputs.to(model.device)
        for key, value in inputs.items():
            if getattr(value, 'is_floating_point', lambda: False)():
                inputs[key] = value.to(model.dtype)
        length = inputs['input_ids'].shape[1]
        check_context(model, length, args.max_new_tokens)
        kwargs = {'eos_token_id': p.gen_terminator} if args.probe_model == 'ming_omni' else {}
        output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, **kwargs)
        return p.batch_decode(output[:, length:], skip_special_tokens=True)[0].strip()
