"""CLI defaults. Explicit command-line arguments keep their existing precedence."""
from pathlib import Path

from benchmark.config.paths import ROOT, MANIFESTS, SYSTEM_PROMPT, DATA, VIDEOS, CACHE, RESULTS

COMMON = {
    'data': DATA,
    'videos_dir': VIDEOS,
    'system_prompt': SYSTEM_PROMPT,
    'max_samples': 0,
}

DEFAULTS = {
    'probe': {
        'data': DATA,
        'videos_dir': VIDEOS,
        'manifest_dir': ROOT / 'probes/default',
        'cache_dir': CACHE / 'probe_prefix',
        'system_prompt': SYSTEM_PROMPT.parent / 'probe.txt',
        'pre_seconds': 1.0,
        'fps': 1.0,
        'max_pixels': 360 * 420,
        'max_new_tokens': 256,
        'max_pairs': 0,
        'dtype': 'bfloat16',
        'device_map': 'auto',
        'attn_implementation': 'sdpa',
    },
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
        'logits_mode': 'full',
        'mlp_chunk_tokens': 0,
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
    'qwen_realtime': {
        **COMMON,
        'output_dir': RESULTS / 'qwen35_omni_plus_realtime',
        'cache_dir': CACHE,
        'model': 'qwen3.5-omni-plus-realtime',
        'api_key_env': 'DASHSCOPE_API_KEY',
        'base_url_env': 'DASHSCOPE_REALTIME_URL',
    },
    'gemini_live': {
        **COMMON,
        'output_dir': RESULTS / 'gemini25_flash_live',
        'cache_dir': CACHE,
        'model': 'gemini-2.5-flash-native-audio-preview-12-2025',
        'api_key_env': 'GEMINI_API_KEY',
        'base_url_env': 'GEMINI_BASE_URL',
        'api_version': 'v1beta',
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
    'cloud_realtime': {
        'api_timeout': 180.0,
        'max_seconds': 0.0,
        'sample_start': 0,
        'sample_step': 1,
        'frame_max_edge': 640,
        'jpeg_quality': 75,
    },
}

# These native outer model classes do not advertise SDPA support.
PROBE_ADAPTER_DEFAULTS = {name: {'attn_implementation': 'eager'}
                          for name in ('ming_omni', 'baichuan_omni', 'omnivinci')}
