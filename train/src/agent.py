"""Read-only adaptation of generated annotations to the single training contract."""
from benchmark.data.agent_schema import adapt_sample, validate_task_start
from benchmark.data.task_text import SEPARATOR

PROTOCOL = 'task_text_sft_v1'
from benchmark.evaluation.agent_calls import normalize_call


def training_sample(sample):
    if sample.get('mode') != 'agent':
        return sample
    for ref in sample.get('responses', []):
        calls = ref.get('tool_calls') or []
        if len(calls) > 1:
            raise ValueError('Training supports exactly one task_start per response')
        for call in calls:
            call = normalize_call(call)
            name = call['function']['name']
            if name not in ('agent_call', 'task_start'):
                raise ValueError('Unsupported source Agent function')
            if name == 'task_start' and not validate_task_start(call):
                raise ValueError('task_start requires only a nonempty instruction')
    result = adapt_sample(sample, 'task-start')
    for ref in result.get('responses', []):
        if ref.get('tool_calls'):
            ref['tool_calls'] = [normalize_call(call) for call in ref['tool_calls']]
    return result


def training_record(record):
    if record['sample'].get('mode') != 'agent':
        return record
    sample = training_sample(record['sample'])
    reference = training_sample({**record['sample'], 'responses': [record['reference']]})['responses'][0]
    if 'response_index' in record and sample.get('responses'):
        if reference != sample['responses'][record['response_index']]:
            raise ValueError('Training reference disagrees with sample.responses')
    return {**record, 'sample': sample, 'reference': reference}


def require_protocol(config):
    if config.get('agent_output_protocol') != PROTOCOL:
        raise ValueError(f'Training requires {PROTOCOL}; regenerate prepared data or retrain old checkpoints')


def validate_prepared(manifest, config):
    require_protocol(config)
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('Prepared data uses an old protocol; run prepare again')
    require_protocol(manifest['config'])


def instruction(reference):
    calls = reference.get('tool_calls') or []
    if not calls:
        return None
    if len(calls) != 1:
        raise ValueError('Training supports one delegation per response')
    call = normalize_call(calls[0])
    if call['function']['name'] not in ('agent_call', 'task_start'):
        raise ValueError('Unsupported source Agent function')
    text = call['function']['arguments'].get('instruction')
    if not isinstance(text, str) or not text.strip() or SEPARATOR in text:
        raise ValueError('Expected one nonempty plain instruction without a separator')
    return text.strip()


def target_text(reference):
    response = reference['response']
    if SEPARATOR in response:
        raise ValueError('Reserved task_start separator in reference response')
    task = instruction(reference)
    if task is not None and not response.strip():
        raise ValueError('Agent target requires a nonempty user-facing response')
    return response + (SEPARATOR + task if task is not None else '')


def plain_prompt(base, sample):
    from benchmark.data.prompts import TOOLS, describe_tool
    tools = sample.get('available_tools') or []
    if len(tools) != len(set(tools)) or any(t not in TOOLS for t in tools):
        raise ValueError('Invalid available tools')
    return base.strip() + '\n\nAvailable backend capabilities (only these may be delegated):\n' + (
        '\n'.join(f'- {t}: {describe_tool(t)}' for t in tools) or 'None')


def system_prompt(sample, config):
    from pathlib import Path
    from benchmark.data.prompts import build_system_prompt
    sample = training_sample(sample)
    if sample.get('mode') == 'agent':
        return plain_prompt(Path(config['agent_prompt']).read_text(), sample)
    return build_system_prompt(Path(config['system_prompt']).read_text(), sample)
