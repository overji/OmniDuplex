"""Opt-in literal separator protocol. No tokenizer changes or reference reconstruction."""
import hashlib
from pathlib import Path

SEPARATOR = '<|task_start|>'
PROTOCOL = 'task_text_v1'
MODELS = ('minicpmo', 'gander', 'joyai', 'livestar')


def identity():
    from benchmark.config.paths import CONFIGS
    paths = [Path(__file__), CONFIGS / 'prompts/agent_task_text.txt',
             *[CONFIGS / f'prompts/agent_task_text_{name}.txt' for name in MODELS]]
    from benchmark.data import prompts
    paths.append(Path(prompts.__file__))
    src = Path(__file__).resolve().parents[1]
    paths += [src / 'adapters' / (name + '.py') for name in MODELS]
    paths.append(src / 'runtime/runner.py')
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def build_prompt(sample, model):
    from benchmark.config.paths import CONFIGS
    from benchmark.data.prompts import TOOLS, describe_tool
    if model not in MODELS:
        raise ValueError('Unsupported task-text streaming model')
    available = sample.get('available_tools') or []
    if len(available) != len(set(available)) or any(t not in TOOLS for t in available):
        raise ValueError('Invalid available capabilities')
    parts = [(CONFIGS / 'prompts/agent_task_text.txt').read_text().strip(),
             (CONFIGS / f'prompts/agent_task_text_{model}.txt').read_text().strip()]
    parts.append('Available backend capabilities (only these may be delegated):\n' +
                 ('\n'.join(f'- {name}: {describe_tool(name)}' for name in available) or 'None'))
    return '\n\n'.join(parts)


def event(text, onset, *, complete=None, separator_time=None, end_time=None):
    count = text.count(SEPARATOR)
    response, _, instruction = text.partition(SEPARATOR)
    response, instruction = response.strip(), instruction.strip()
    valid_format = count == 1 and bool(response) and bool(instruction)
    if complete is False:
        status = 'truncated'
    elif not count:
        status = 'absent'
    elif count > 1:
        status = 'duplicate_separator'
    elif not response or not instruction:
        status = 'empty_field'
    else:
        status = 'valid'
    calls = ([{'type': 'function', 'function': {'name': 'task_start',
              'arguments': {'instruction': instruction}}}] if status == 'valid' else [])
    return {'onset_seconds': onset, 'text': text, 'response_text': response,
            'agent_instruction': instruction, 'tool_calls': calls,
            'agent_attempted': count > 0, 'task_text_status': status,
            'task_text_format_valid': valid_format, 'task_text_generation_complete': complete,
            'task_text_separator_count': count,
            'agent_parse_status': ('structured' if calls else 'absent' if status == 'absent'
                                   else 'truncated' if status == 'truncated' else 'malformed'),
            'agent_call_onset_seconds': separator_time if count else None,
            'agent_call_complete_seconds': end_time if calls else None}


class TaskTextStream:
    """Decode all lexical IDs together, preserving split separators and UTF-8 text."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special = set(tokenizer.all_special_ids)
        # MiniCPM native control IDs are not all marked special in its tokenizer.
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            for token in ('<unit>', '</unit>', '<|listen|>', '<|speak|>', '<|turn_eos|>',
                          '<|chunk_eos|>', '<|chunk_tts_eos|>', '<|tts_bos|>', '<|tts_eos|>',
                          '<|tts_pad|>', '<|interrupt|>', '<|backchannel|>'):
                value = tokenizer.convert_tokens_to_ids(token)
                if value is not None and value >= 0 and value != getattr(tokenizer, 'unk_token_id', None):
                    self.special.add(value)
        self.reset()

    def reset(self):
        self.ids = []
        self.onset = self.separator_time = self.last_time = None

    def feed(self, ids, clock, *, ended=False, interrupted=False):
        lexical = [int(i) for i in ids if int(i) not in self.special]
        if lexical:
            if self.onset is None:
                self.onset = clock
            self.ids.extend(lexical)
            self.last_time = clock
            text = self.tokenizer.decode(self.ids, skip_special_tokens=False)
            if self.separator_time is None and SEPARATOR in text:
                self.separator_time = clock
        if ended or interrupted:
            return self.finish(complete=not interrupted, clock=clock)
        return None

    def finish(self, *, complete=False, clock=None):
        if self.onset is None:
            self.reset()
            return None
        text = self.tokenizer.decode(self.ids, skip_special_tokens=False)
        result = event(text, self.onset, complete=complete, separator_time=self.separator_time,
                       end_time=clock if clock is not None else self.last_time)
        self.reset()
        return result
