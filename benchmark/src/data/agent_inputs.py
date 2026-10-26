"""Version identity for model-facing input, adapter and full-model protocols."""
AGENT_INPUT_PROTOCOL = "post_media_generation_without_agent_results_v3"


def input_protocol(args):
    from benchmark.config.paths import CONFIGS
    import hashlib
    protocol = getattr(args, 'agent_output_protocol', 'text')
    schema = getattr(args, 'agent_call_schema', 'legacy')
    if protocol == 'task-text':
        if schema != 'task-start' or getattr(args, 'lora_adapter', None):
            raise ValueError('task-text requires task-start schema and unmodified checkpoint weights')
        from benchmark.data.task_text import identity
        return AGENT_INPUT_PROTOCOL + ':task_text_v1:' + identity()
    if schema == 'task-start':
        from types import SimpleNamespace
        if protocol == 'agent-token':
            raise ValueError('agent-token requires --agent-call-schema legacy')
        previous = input_protocol(SimpleNamespace(**{**vars(args), 'agent_call_schema': 'legacy'}))
        files = ('agent_task_start_structured.txt', 'agent_task_start_gander.txt')
        identity = hashlib.sha256(b''.join((CONFIGS / 'prompts' / name).read_bytes() for name in files)).hexdigest()
        return previous + ':task_start_instruction_v1:' + identity
    if hasattr(args, 'max_tool_tokens'):
        prompt = hashlib.sha256((CONFIGS / 'prompts/gander_agent.txt').read_bytes()).hexdigest()
        return AGENT_INPUT_PROTOCOL + ':gander_native_v2:' + prompt
    if protocol == 'agent-token':
        from benchmark.data.agent_token import adapter_identity, full_model_identity, digest
        if getattr(args, 'lora_adapter', None):
            identity = adapter_identity(args.lora_adapter)
        elif getattr(args, 'model_path', None) and (args.model_path / 'full_training_model.json').is_file():
            identity = full_model_identity(args.model_path)
        else:
            raise ValueError('agent-token requires --lora-adapter or an exported full-training --model-path')
        prompt = hashlib.sha256((CONFIGS / 'prompts/agent_token.txt').read_bytes()).hexdigest()
        return AGENT_INPUT_PROTOCOL + ':agent_token_v1:' + digest(identity) + ':' + prompt
    if getattr(args, 'lora_adapter', None):
        raise ValueError('Agent-token adapter requires --agent-output-protocol agent-token')
    if protocol == 'structured':
        digest = hashlib.sha256((CONFIGS / 'prompts/agent_structured.txt').read_bytes()).hexdigest()
        return AGENT_INPUT_PROTOCOL + ':structured_v1:' + digest
    return AGENT_INPUT_PROTOCOL
