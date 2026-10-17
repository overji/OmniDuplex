"""Pinned model metadata and existing native inference settings.

These values preserve the evaluated protocols; changing them creates a new experiment.
"""
from benchmark.config.paths import MODEL_ROOT

MODEL_SETTINGS = {
    'qwen_realtime': {
        'native_api': 'DashScope Realtime WebSocket',
        'fps': 1,
        'audio_sample_rate': 16000,
        'output_modality': 'text',
        'turn_control': 'manual_commit_then_response_create_per_media_second',
        'missing_audio': 'synthetic_silence_required_by_image_buffer_protocol',
        'minimum_audio_commit_ms': 100,
        'timing_clock': 'observed_input_interval_end',
    },
    'gemini_live': {
        'native_api': 'google-genai aio.live',
        'fps': 1,
        'audio_sample_rate': 16000,
        'output_modality': 'audio_with_output_transcription',
        'turn_control': 'manual_activity_end_per_media_second',
        'missing_audio': 'omit_audio',
        'timing_clock': 'observed_input_interval_end',
    },
    'minicpmo': {
        'native_api': 'as_duplex/prepare/streaming_prefill/streaming_generate',
        'generate_audio': False,
        'fps': 1,
        'audio_sample_rate': 16000,
        'timing_clock': 'observed_input_video_time_at_first_text_token',
        'temperature': 0.7,
        'top_k': 100,
        'top_p': 0.8,
    },
    'mmduet2': {
        'model': 'MMDuet2',
        'model_revision': '3224235ebb6256e2af7c196f2c265d6cf20c311a',
        'native_api': 'ProactiveInferenceClient/input_query_stream/inference',
        'frame_interval_seconds': 1,
        'sampling': {
            'do_sample': False,
            'max_new_tokens': 512,
        },
    },
    'livestar': {
        'model': 'LiveStar base',
        'model_revision': 'fe12f966639ace9562a6db8843dfa5733ccc0d24',
        'official_code_commit': 'aa1cc540300f3c318f3511df5581605519ecc2a1',
        'native_api': 'StreamingSession SVeD perplexity gate',
        'sample_fps': 1.0,
        'max_num': 1,
        'alpha': 1.06,
        'num_runs': 3,
        'max_new_tokens': 384,
        'pro': False,
        'visual_feature_prefetch': False,
    },
    'roma': {
        'model': 'ROMA',
        'model_revision': 'cc245ea5dab25a39fa6a12a7e6dc7405eebd9254',
        'official_code_commit': 'e5bced66527a18c80e9dffe3a426f107a0bb027c',
        'transformers_source': 'Eureka-Maggie/transformers@roma_patch',
        'native_api': 'Qwen2.5-Omni thinker SpeakHead threshold and streaming generation',
        'video_fps': 2.0,
        'audio_sample_rate': 16000,
        'use_audio_in_video': True,
        'head': 'mlp',
        'threshold': 0.975,
        'thinker_max_new_tokens': 25,
    },
    'joyai': {
        'model': 'JoyAI-VL-Interaction',
        'model_revision': '86124c620faddd7be7b7f2722f29cd5192f21c38',
        'official_code_commit': '32add65c0f0f1525d4a9cb2cfdea6a496d34b4d8',
        'native_api': 'JoyAI StreamingInferAdapter OpenAI endpoint',
        'frame_seconds': 1.0,
        'chunk_frames': 100,
        'compression_every_n_chunks': 5,
        'summarizer': str(MODEL_ROOT / "Qwen3-VL-4B-Instruct"),
        'audio_input': False,
    },
}

# Arguments passed directly to each official implementation.
NATIVE_SETTINGS = {
    'mmduet2': {
        'inference': {
            'input_assistant_turns': False,
            'temperature': 1.0,
            'top_k': 40,
            'attn_implementation': 'flash_attention_2',
        },
    },
    'livestar': {
        'generation': {
            'temperature': 0.0,
            'top_p': 0.1,
            'num_beams': 1,
            'repetition_penalty': 1.05,
        },
        'session': {
            'l_max': 1000000000,
            'sigma': 0.75,
            'beta': 0.3,
            'beam_k': 3,
            'trace': False,
            'no_tshm': True,
            'no_retrieval': True,
            'max_recall': 0,
            'kv': False,
            'no_viscache': True,
            'recall_verbatim': False,
            'recall_min_gap': 0,
        },
    },
    'roma': {
        'processor': {
            'image_max_pixels': 262144,
            'image_min_pixels': 1024,
            'image_do_pan_and_scan': False,
            'crop_to_patches': False,
            'video_max_pixels': 65536,
            'video_min_pixels': 256,
            'video_maxlen': 14400,
        },
    },
}
