"""Version identity for model-facing input, adapter and full-model protocols."""
AGENT_INPUT_PROTOCOL = "post_media_generation_without_agent_results_v3"


def input_protocol(args):
    from benchmark.config.paths import CONFIGS
    import hashlib
    protocol = getattr(args, 'agent_output_protocol', 'text')
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
