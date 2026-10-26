"""Gander Unit-8 Thinker using its official native streaming decoder."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from benchmark.adapters import minicpmo
from benchmark.config.models import MODEL_SETTINGS
from benchmark.config.paths import CONFIGS
from benchmark.data.prompts import available_function_schemas

STRICT_CONFIG = True
SELECT_BEFORE_RESUME = True


def run_config(args, prompt):
    config = minicpmo.run_config(args, prompt)
    config.update(MODEL_SETTINGS['gander'])
    root = args.official_code / 'minicpm_ft' / 'mcpmft'
    config['official_source_sha256'] = hashlib.sha256(json.dumps({
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*.py'))}, sort_keys=True).encode()).hexdigest()
    name = 'agent_task_start_gander.txt' if getattr(args, 'agent_call_schema', 'legacy') == 'task-start' else 'gander_agent.txt'
    config['native_agent_prompt_sha256'] = hashlib.sha256((CONFIGS / 'prompts' / name).read_bytes()).hexdigest()
    config['native_tool_event_protocol'] = 'structured_metadata_v2'
    return config


class StreamingThinker:
    def __init__(self, bundle, params, runner_class):
        self.bundle, self.params, self.runner_class = bundle, params, runner_class
        self.runner = None
        self.tools = []

    def prepare(self, prefix_system_prompt):
        # The native runner resets its KV/audio state in prepare(). Reuse it:
        # rebuilding duplex wrappers per video can retain cyclic GPU caches.
        if self.runner is None:
            self.runner = self.runner_class(self.bundle, self.params)
        self.runner.prepare(system_prompt=prefix_system_prompt, tools=self.tools)
        self.runner.duplex.force_listen_count = self._force_listen_count

    @property
    def force_listen_count(self):
        return self._force_listen_count

    @force_listen_count.setter
    def force_listen_count(self, value):
        self._force_listen_count = value

    @property
    def tokenizer(self):
        return self.bundle.tokenizer

    @property
    def total_ids(self):
        return self.runner.duplex.total_ids

    def streaming_prefill(self, **kwargs):
        return self.runner.duplex.streaming_prefill(**kwargs)

    def streaming_generate(self, **kwargs):
        return self.runner.streaming_generate(**kwargs)


def load(args):
    if getattr(args, 'lora_adapter', None) or getattr(args, 'agent_output_protocol', 'text') == 'agent-token':
        raise ValueError('Gander uses its released native control/tool grammar, without MiniCPM training adapters')
    source = args.official_code / 'minicpm_ft'
    if not (source / 'mcpmft' / 'infer' / 'online.py').is_file():
        raise FileNotFoundError('Incomplete official Gander runtime checkout')
    sys.path.insert(0, str(source.resolve()))
    from mcpmft.args import ModelArguments
    from mcpmft.infer.common import load_for_infer
    from mcpmft.infer.online import DuplexParams, OnlineRunner
    # This benchmark scores Thinker text onset. Talker weights are downloaded,
    # but speech synthesis is not needed for the shared text-timing protocol.
    base_alias = args.cache_dir / 'gander_base'
    base_alias.parent.mkdir(parents=True, exist_ok=True)
    if not base_alias.exists():
        base_alias.symlink_to(args.base_model_path.resolve(), target_is_directory=True)
    if base_alias.resolve() != args.base_model_path.resolve():
        raise ValueError('Gander base checkpoint alias changed')
    thinker = args.model_path / 'thinker'
    bundle = load_for_infer(ModelArguments(model_name_or_path=str(base_alias),
        processor_name_or_path=str(base_alias), init_tts=False, device_map=args.device),
        checkpoint=str(thinker), init_token2wav=False)
    params = DuplexParams(generate_audio=False, speak_text_tokens_per_unit=8,
        max_new_speak_tokens_per_chunk=args.max_speak_tokens, decode_mode=args.decode_mode,
        sliding_window_mode=args.sliding_window_mode,
        max_new_tool_tokens=getattr(args, 'max_tool_tokens', 96))
    return SimpleNamespace(model=StreamingThinker(bundle, params, OnlineRunner))


def run_sample(args, state, context):
    state.model.tools = ([] if getattr(args, 'agent_output_protocol', 'text') == 'task-text'
                         else available_function_schemas(context.sample))
    if context.sample.get('mode') == 'agent':
        name = 'agent_task_start_gander.txt' if context.sample.get('_agent_call_schema') == 'task-start' else 'gander_agent.txt'
        context.native_protocol = (CONFIGS / 'prompts' / name).read_text()
    return minicpmo.run_sample(args, state, context)
