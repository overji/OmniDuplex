"""CLI defaults. Explicit command-line arguments keep their existing precedence."""
from pathlib import Path

from benchmark.config.paths import MANIFESTS, SYSTEM_PROMPT, DATA, VIDEOS, CACHE, RESULTS

COMMON = {
    'data': DATA,
    'videos_dir': VIDEOS,
    'system_prompt': SYSTEM_PROMPT,
    'max_samples': 0,
}

DEFAULTS = {
    'minicpmo': {
        **COMMON,
        'output_dir': RESULTS / "minicpmo45_realtime",
        'cache_dir': CACHE,
        'device': "cuda:0",
        'seed': 42,
        'sample_start': 0,
        'sample_step': 1,
        'max_seconds': 0,
        'decode_mode': "sampling",
        'max_speak_tokens': 20,
        'sliding_window_mode': "off",
        'window_high_tokens': 8000,
        'window_low_tokens': 6000,
        'cpu_threads': 4,
        'unused_tts_device': "cpu",
    },
    'mmduet2': {
        **COMMON,
        'cache_dir': CACHE,
        'output_dir': RESULTS / "mmduet2",
        'device': "cuda:0",
    },
    'livestar': {
        **COMMON,
        'cache_dir': CACHE,
        'output_dir': RESULTS / "livestar",
    },
    'roma': {
        **COMMON,
        'output_dir': RESULTS / "roma",
    },
    'joyai': {
        **COMMON,
        'cache_dir': CACHE,
        'output_dir': RESULTS / "joyai",
        'adapter_url': "http://127.0.0.1:8070/v1/chat/completions",
        'model': "jdopensource/JoyAI-VL-Interaction",
    },
    'collect': {
        'data': DATA,
        'output_dir': RESULTS / 'minicpmo45_realtime',
    },
    'compare': {
        'output': Path("benchmark/results/model_comparison"),
    },
    'download': {
        'repo': "openbmb/MiniCPM-o-4_5",
        'manifest': MANIFESTS / "model_manifest.json",
    },
    'parallel': {
        'poll_seconds': 10,
    },
    'queue': {
    },
}
