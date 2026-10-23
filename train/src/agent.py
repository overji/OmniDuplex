"""Read-only adaptation of generated annotations to the single training contract."""
from benchmark.data.agent_schema import adapt_sample, validate_task_start
from benchmark.data.agent_token import PROTOCOL
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
