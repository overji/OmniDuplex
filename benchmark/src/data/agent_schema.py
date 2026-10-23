"""Benchmark-only views of legacy annotations; never rewrite source data."""
from copy import deepcopy
import json

TASK_START = {'type': 'function', 'function': {
    'name': 'task_start', 'description': 'Delegate a task to the backend agent.',
    'parameters': {'type': 'object', 'properties': {
        'instruction': {'type': 'string', 'minLength': 1,
                        'description': 'One concise, self-contained task instruction using observed facts.'}},
        'required': ['instruction'], 'additionalProperties': False}}}


def adapt_sample(sample, schema='legacy'):
    if schema not in ('legacy', 'task-start'):
        raise ValueError('Unknown Agent call schema')
    if schema == 'legacy' or sample.get('mode') != 'agent':
        return sample
    result = deepcopy(sample)
    result['_agent_call_schema'] = schema
    frontend = result.setdefault('frontend_context', {})
    frontend['system_prompt'] = frontend.get('system_prompt', '').replace('agent_call', 'task_start')
    frontend['tools'] = [deepcopy(TASK_START)] if result.get('available_tools') else []
    for response in result.get('responses', []):
        for call in response.get('tool_calls', []) or []:
            function = call.get('function', call)
            if function.get('name') not in ('agent_call', 'task_start'):
                continue
            args = function.get('arguments')
            encoded = isinstance(args, str)
            args = json.loads(args) if encoded else args
            instruction = args.get('instruction') if isinstance(args, dict) else None
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('Reference task_start requires a nonempty instruction')
            function['name'] = 'task_start'
            simplified = {'instruction': instruction}
            function['arguments'] = json.dumps(simplified, ensure_ascii=False) if encoded else simplified
    return result


def validate_task_start(call):
    function = call.get('function', call)
    args = function.get('arguments')
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return False
    return (function.get('name') == 'task_start' and isinstance(args, dict)
            and set(args) == {'instruction'} and isinstance(args['instruction'], str)
            and bool(args['instruction'].strip()))


def run_schema(directory):
    """Old result directories default to legacy, independent of CLI defaults."""
    for name in ('run_config.json', 'parallel_plan.json'):
        path = directory / name
        if path.exists():
            config = json.loads(path.read_text())
            return config.get('agent_call_schema', config.get('arguments', {}).get('agent_call_schema', 'legacy'))
    return 'legacy'
