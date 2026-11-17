"""Normalize native Agent actions, never reconstruct missing output from labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from benchmark.data.agent_schema import TASK_START, validate_task_start
from benchmark.evaluation.agent_calls import normalize_call

VERSION = 'native_agent_instruction_v1'


def identity():
    from benchmark.config.paths import CONFIGS
    src = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), src/'data/agent_schema.py', src/'runtime/runner.py',
             *[src/f'adapters/{name}.py' for name in ('gander','joyai','minicpmo')],
             *[CONFIGS/f'prompts/agent_native_{name}.txt' for name in ('gander','joyai')]]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)+
        json.dumps(TASK_START,sort_keys=True).encode()).hexdigest()


def tools(sample):
    return [deepcopy(TASK_START)] if sample.get('available_tools') else []


def build_prompt(sample, model):
    from benchmark.config.paths import CONFIGS
    from benchmark.data.prompts import TOOLS, describe_tool
    if model not in ('gander','joyai') or sample.get('_agent_call_schema')!='task-start':
        raise ValueError('Native Agent mode requires JoyAI/Gander and task-start schema')
    available=sample.get('available_tools') or []
    if len(available)!=len(set(available)) or any(name not in TOOLS for name in available):
        raise ValueError('Invalid available capabilities')
    # Frontend legacy format prompts and reference fields must not override the
    # native action grammar. Backend capabilities remain explicitly available.
    return (CONFIGS/f'prompts/agent_native_{model}.txt').read_text().strip()+'\n\n'+(
        'Available backend capabilities (only these may be delegated):\n'+
        ('\n'.join(f'- {name}: {describe_tool(name)}' for name in available) or 'None'))


def call_event(result, clock):
    raw=deepcopy(result.get('tool_calls') or [])
    calls=[]
    complete=result.get('tool_generation_complete')
    status='structured'
    try:
        if (result.get('tool_error') or complete is not True
                or result.get('tool_parse_valid') is False
                or result.get('tool_schema_valid') is False):
            raise ValueError('Invalid or incomplete native tool action')
        if not isinstance(raw,list) or len(raw)!=1:
            raise ValueError('Expected exactly one task_start action')
        calls=[normalize_call(raw[0])]
        if not validate_task_start(calls[0]):raise ValueError('Invalid instruction schema')
    except (ValueError,TypeError):
        calls=[];status='truncated' if complete is False else 'malformed'
    return dict(onset_seconds=clock,text='',response_text='',tool_calls=calls,
        agent_attempted=True,agent_parse_status=status,native_protocol_version=VERSION,
        agent_call_onset_seconds=clock,agent_call_complete_seconds=clock if complete is True else None,
        native_generation_complete=complete,native_tool_calls=raw,
        native_raw_tool_text=result.get('raw_tool_text',''),native_tool_error=result.get('tool_error'),
        native_tool_parse_valid=result.get('tool_parse_valid'),
        native_tool_schema_valid=result.get('tool_schema_valid'))


def joyai_event(raw_text, clock, *, complete=None):
    """Accept the pinned runtime's two explicit delegation-marker spellings."""
    text=raw_text.strip()
    if not text or text=='</silence>':return None
    markers=list(re.finditer(r'</?delegation>',text))
    response=text[:markers[0].start()] if markers else text
    response=response.removeprefix('</response>').strip()
    instruction=text[markers[0].end():].strip() if len(markers)==1 else ''
    status='absent';calls=[]
    if markers:
        status=('truncated' if complete is False else 'malformed')
        if len(markers)==1 and instruction and complete is not False:
            calls=[dict(type='function',function=dict(name='task_start',arguments=dict(instruction=instruction)))]
            status='structured'
    return dict(onset_seconds=clock,text=raw_text,response_text=response,
        response_onset_seconds=clock if response else None,tool_calls=calls,
        agent_attempted=bool(markers),agent_parse_status=status,native_protocol_version=VERSION,
        native_raw_text=raw_text,native_generation_complete=complete,
        agent_call_onset_seconds=clock if markers else None,
        agent_call_complete_seconds=clock if markers and complete is True else None)


class NativeAgentEvents:
    """Associate one adjacent speech turn and native call, in either order.

    Adjacency is measured in input units, not guessed from reference windows.
    Silence/interruptions and other actions break it. Each component is consumed
    at most once; repeated calls are never deduplicated.
    """
    def __init__(self):
        self.unit=0
        self.pending=None
        self.speech=None

    @staticmethod
    def _clean(event):
        return {k:v for k,v in event.items() if not k.startswith('_')}

    def _flush_pending(self,out):
        if self.pending is not None:
            out.append(self._clean(self.pending));self.pending=None

    def _close_speech(self,out,*,complete):
        if self.speech is None:return
        event=self.speech;self.speech=None
        event['response_generation_complete']=complete
        if event.get('agent_attempted'):
            out.append(self._clean(event))
        else:
            self._flush_pending(out)
            event['_kind']='speech';event['_unit']=self.unit
            self.pending=event

    @staticmethod
    def _merge(speech,call):
        result={**call,**{k:v for k,v in speech.items() if not k.startswith('_')}}
        # Keep the actual call status/counters, never infer a call from speech.
        result.update({k:v for k,v in call.items() if k not in ('text','response_text','onset_seconds')})
        result['onset_seconds']=min(speech['onset_seconds'],call['onset_seconds'])
        return result

    def feed(self,result,clock):
        self.unit+=1;out=[]
        if result.get('is_interrupt') or result.get('is_listen'):
            self._close_speech(out,complete=not result.get('is_interrupt',False))
            self._flush_pending(out)
            return out
        if result.get('is_tool_call') or result.get('tool_calls'):
            # A tool action follows the previous speaking unit, never text in
            # this unit. A prior tool+speech pair cannot consume a second call.
            if self.speech is not None:self._close_speech(out,complete=True)
            call=call_event(result,clock)
            if (self.pending is not None and self.pending['_kind']=='speech'
                    and self.pending['_unit'] in (self.unit-1,self.unit)):
                out.append(self._clean(self._merge(self.pending,call)));self.pending=None
            else:
                self._flush_pending(out)
                call.update(_kind='call',_unit=self.unit);self.pending=call
            return out
        text=result.get('text') or ''
        if text:
            if self.speech is None:
                speech=dict(onset_seconds=clock,text='',response_text='',response_onset_seconds=clock,
                            tool_calls=[],agent_attempted=False,agent_parse_status='absent',native_protocol_version=VERSION)
                if (self.pending is not None and self.pending['_kind']=='call'
                        and self.pending['_unit']==self.unit-1):
                    speech=self._merge(speech,self.pending);self.pending=None
                else:self._flush_pending(out)
                self.speech=speech
            self.speech['response_text']+=text
            self.speech['text']=self.speech['response_text']
            self.speech['last_text_seconds']=clock
        elif not result.get('end_of_turn'):
            # An unrecognized/empty action is a barrier, not evidence that a
            # speech turn continued across this input unit.
            self._close_speech(out,complete=False)
            self._flush_pending(out)
        if result.get('end_of_turn'):
            self._close_speech(out,complete=True)
        return out

    def finish(self):
        out=[]
        self._close_speech(out,complete=False)
        self._flush_pending(out)
        return out
