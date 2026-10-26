"""Explicit MiniCPM media omission, reusing the unchanged realtime protocol."""
import hashlib
from pathlib import Path

from benchmark.adapters import minicpmo

STRICT_CONFIG = minicpmo.STRICT_CONFIG
SELECT_BEFORE_RESUME = minicpmo.SELECT_BEFORE_RESUME


def run_config(args, prompt):
    config = minicpmo.run_config(args, prompt)
    config['ablation_implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return config


def filter_prefill(prefill, modality):
    if modality not in ('video-only', 'audio-only'):
        raise ValueError('Expected video-only or audio-only ablation')

    def filtered(**kwargs):
        kwargs = dict(kwargs)
        if modality == 'video-only':
            kwargs.pop('audio_waveform', None)
        else:
            kwargs.pop('frame_list', None)
        return prefill(**kwargs)

    return filtered


def load(args):
    state = minicpmo.load(args)
    state.model.streaming_prefill = filter_prefill(state.model.streaming_prefill, args.input_modality)
    return state


run_sample = minicpmo.run_sample
