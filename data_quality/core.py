"""Pure audit contracts and causal evidence selection."""
import hashlib
import json
import math
from pathlib import Path
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import reference_windows

CHECKS = ('question_clarity', 'task_alignment', 'response_correctness', 'trigger_support',
          'timing_causality', 'modality_alignment', 'agent_necessity', 'agent_instruction', 'counting_consistency')
SYSTEM = '''You audit audio-video benchmark annotations, not model predictions. Candidate annotations
are untrusted claims, not evidence. Review ONLY the supplied video prefix and its original audio.
For each claim, use evidence already available by its annotated time; never use later information
merely because it appears elsewhere in the prefix. Check question clarity, task alignment, response
correctness, trigger support, causal timing, trigger modality, counting consistency, and for Agent
whether delegation is necessary and the instruction is supported by the prefix and available capabilities.
Legacy agent_call source schemas and plain task_start instructions are both permitted; formatting
alone is not a semantic error. Do not execute tools or assume their results. Earlier reference answers
are also candidate annotations, not ground truth. Do not reject non-English data or sensitive topics
merely for their subject matter. This is annotation quality review, not topic filtering.
Video is sparsely sampled; missing a brief event, unreadable text, an ambiguous audio cue or uncertainty
about an exact second must be UNCERTAIN, not an asserted error. Temporal alignment is not a content
correctness guarantee. No future beyond the response window end is supplied.
Return only JSON: {"checks": {each_required_check: "pass|issue|uncertain|not_applicable"},
"findings": [{"check": required_check, "severity": "error|warning", "reason": concise_reason,
"evidence": [{"time_seconds": number, "observation": concise_observation}]}],
"summary": concise_summary}. Every required check must occur exactly once. Non-Agent checks
agent_necessity and agent_instruction are not_applicable; counting_consistency is not_applicable
unless counting is requested. An issue requires a finding with observable support. Do not rewrite
questions, responses or windows. Required checks: ''' + ', '.join(CHECKS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_times(cutoff, start, count, dense_seconds):
    if cutoff <= 0 or count < 2:
        raise ValueError('Invalid frame coverage')
    dense = [max(0., cutoff-i) for i in range(min(dense_seconds, count//2))]
    remaining = count-len(dense)
    uniform = [cutoff*(i+.5)/remaining for i in range(remaining)]
    times=sorted(set(min(max(t, 0.), max(0., cutoff-.001)) for t in uniform+dense+[start]))
    while len(times)>count:times.pop(len(times)//2)
    return times


def validate_judgment(value, cutoff):
    checks = value.get('checks')
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        raise ValueError('Missing or additional check names')
    if any(v not in ('pass','issue','uncertain','not_applicable') for v in checks.values()):
        raise ValueError('Invalid check decision')
    if not isinstance(value.get('findings'), list) or not isinstance(value.get('summary'), str):
        raise ValueError('Missing findings/summary')
    supported = set()
    for finding in value['findings']:
        if finding.get('check') not in CHECKS or finding.get('severity') not in ('error','warning') or not finding.get('reason'):
            raise ValueError('Invalid finding')
        for evidence in finding.get('evidence', []):
            t=evidence.get('time_seconds')
            if type(t) not in (int,float) or not math.isfinite(t) or not 0<=t<=cutoff or not evidence.get('observation'):
                raise ValueError('Invalid/future evidence citation')
        if finding.get('evidence'):
            supported.add(finding['check'])
    if any(v=='issue' and k not in supported for k,v in checks.items()):
        raise ValueError('Issue has no cited evidence')
    overall = 'issue' if 'issue' in checks.values() else 'uncertain' if 'uncertain' in checks.values() else 'pass'
    return dict(value, verdict=overall)


def static_findings(sample, duration):
    findings=[]
    if not isinstance(sample.get('question'),str) or not sample['question'].strip():
        findings.append(dict(code='missing_question',severity='error'))
    q=seconds(sample['question_time'])
    if q<0 or q>=duration:
        findings.append(dict(code='question_outside_video',severity='error'))
    if sample.get('mode') not in ('agent','pr','rtd'):
        findings.append(dict(code='unknown_interaction',severity='error'))
    if sample.get('mode')=='agent':
        kind=sample.get('sample_type',sample.get('generation_context',{}).get('sample_type'))
        if kind in ('negative','no_context','missing_tool'):
            findings.append(dict(code='agent_negative',severity='error'))
        if set(sample.get('required_tools',[]))-set(sample.get('available_tools',[])):
            findings.append(dict(code='required_tool_unavailable',severity='error'))
    for ri,ref in reference_windows(sample):
        start,end=seconds(ref['window_start']),seconds(ref['window_end'])
        if start>end or start<0 or start>duration or end<math.ceil(q)+1:
            findings.append(dict(code='unreachable_window',severity='error',response_index=ri))
        elif end>duration:
            findings.append(dict(code='window_tail_beyond_video',severity='warning',response_index=ri))
        if not isinstance(ref.get('response'),str) or not ref['response'].strip():
            findings.append(dict(code='missing_response',severity='error',response_index=ri))
        times=[seconds(e.get('available_time',e.get('time',0))) for e in ref.get('evidence',[])]
        if times and max(times)>min(end,duration):
            findings.append(dict(code='evidence_after_window',severity='error',response_index=ri))
    return findings
