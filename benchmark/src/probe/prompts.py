"""Build the whitelisted input boundary for a single independent probe."""
from dataclasses import dataclass
from pathlib import Path

from benchmark.data.prompts import build_system_prompt
from benchmark.probe.data import REQUEST_FIELDS

TASK_PROMPTS = {'rtd': 'probe_rtd.txt', 'pr': 'probe_pr.txt', 'agent': 'probe_agent.txt'}


def task_prompt_files():
    from benchmark.config.paths import CONFIGS
    return [CONFIGS/'prompts'/name for name in [*TASK_PROMPTS.values(), 'probe_agent_note_examples.txt']]


@dataclass(frozen=True)
class ProbeInput:
    system_prompt: str
    user_prompt: str
    video: Path
    has_audio: bool
    prefix_seconds: float


def model_input(row, video, template, output_protocol='yes-no', prompt_layout='shared'):
    request = {key: row['request'][key] for key in REQUEST_FIELDS if key in row['request']}
    if prompt_layout not in ('shared', 'task-specific'):
        raise ValueError('Unsupported prompt layout')
    if prompt_layout == 'task-specific' and output_protocol != 'reply-task-text':
        raise ValueError('task-specific layout requires reply-task-text')
    if output_protocol == 'reply-task-text':
        from benchmark.data.prompts import TOOLS, describe_tool
        from benchmark.config.paths import CONFIGS
        available = request.get('available_tools') or []
        if len(available) != len(set(available)) or any(t not in TOOLS for t in available):
            raise ValueError('Invalid available capabilities')
        system = template.strip()
        if prompt_layout == 'task-specific':
            if request['mode'] not in TASK_PROMPTS:
                raise ValueError('Unsupported probe task mode')
            system += '\n\n' + (CONFIGS/'prompts'/TASK_PROMPTS[request['mode']]).read_text().strip()
            if request['mode'] == 'agent' and 'note_write' in available:
                system += '\n\n' + (CONFIGS/'prompts/probe_agent_note_examples.txt').read_text().strip()
        if prompt_layout == 'shared' or request['mode'] == 'agent':
            system += '\n\nAvailable backend capabilities (only these may be delegated):\n' + (
                '\n'.join(f'- {name}: {describe_tool(name)}' for name in available) or 'None')
        system += '\nRequest task mode: ' + request['mode']
    else:
        system = build_system_prompt(template, request)
    if output_protocol == 'yes-no':
        instruction = 'Answer NO, or YES: followed by a brief explanation.'
    elif prompt_layout == 'task-specific':
        instruction = ('Output exactly <silent> if none is warranted. Otherwise give the actual reply.' +
            (' Append exactly one <|task_start|> followed by the backend instruction.' if request['mode']=='agent' else
             ' Output only the reminder.' if request['mode']=='pr' else ' Output only the relevant update.'))
    else:
        instruction = ('Output exactly <silent> if none is warranted. Otherwise give the actual reply; '
                       'for Agent also append <|task_start|>instruction. Do not output YES/NO labels.')
    user = (f"Original standing request: {request['question']}\n"
            f"Request issued at video time {request['question_time']}.\n"
            f"This video prefix ends at {row['prefix_end_seconds']:g} seconds from video start.\n"
            f"Earlier reply opportunities already handled: {row['handled_count']}.\n"
            "Has at least one ADDITIONAL opportunity to respond become warranted?\n"
            + instruction)
    return ProbeInput(system, user, video, bool(row['has_audio']), row['prefix_end_seconds'])
