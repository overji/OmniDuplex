"""Versioned response/silence parsing, separate from correctness of the decision."""
import re
from benchmark.data.task_text import SEPARATOR, event

PROTOCOL = 'probe_reply_task_text_v1'
SILENT = '<silent>'


class GeneratedText(str):
    def __new__(cls, text, *, complete=None, token_count=None):
        value = super().__new__(cls, text)
        value.generation_complete = complete
        value.generated_tokens = token_count
        return value


def decoded_output(text, sequences, input_length, tokenizer, maximum, eos=None):
    """Use actual generated IDs, never infer truncation from decoded text length."""
    try:
        ids = sequences[0, input_length:].tolist()
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise TypeError('No generated token IDs')
        stops = eos if eos is not None else getattr(tokenizer, 'eos_token_id', None)
        stops = {stops} if isinstance(stops, int) else set(stops or [])
        ended = bool(ids) and ids[-1] in stops
        complete = True if ended else False if len(ids) >= maximum else None
        return GeneratedText(text, complete=complete, token_count=len(ids))
    except (TypeError, AttributeError, IndexError):
        return GeneratedText(text)


def parse_reply(text, mode, *, complete=None):
    clean = text.strip()
    result = event(text, 0, complete=complete)
    result.pop('onset_seconds')
    if (not clean or (SILENT in clean and clean != SILENT)
            or re.match(r'^(?:YES|NO)(?:\s*:|[.!]?\s*$)', clean, re.I)):
        decision, status = None, 'invalid'
    elif clean == SILENT:
        decision, status = 'NO', 'silent'
    else:
        decision, status = 'YES', 'response'
    valid = (status == 'silent' or
             status == 'response' and (result['task_text_status'] == 'valid' if mode == 'agent'
                                      else SEPARATOR not in clean)) and complete is not False
    if status != 'response' or mode != 'agent':
        result['tool_calls'] = []
    result.update(decision=decision, parse_status='valid' if decision else 'invalid',
                  output_format_valid=bool(valid), output_status='truncated' if complete is False else status,
                  response_text='' if status == 'silent' else result['response_text'])
    return result
