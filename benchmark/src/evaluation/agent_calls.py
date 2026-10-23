"""Conservative extraction of explicit calls, never reconstruction from references."""
import json
import re


def normalize_call(call):
    """Accept explicit OpenAI envelopes or native name/arguments objects only."""
    if not isinstance(call, dict):
        raise ValueError('Tool call must be an object')
    function = call.get('function', call)
    if not isinstance(function, dict) or not isinstance(function.get('name'), str):
        raise ValueError('Tool call has no function name')
    arguments = function.get('arguments')
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not function['name'].strip() or not isinstance(arguments, dict):
        raise ValueError('Invalid tool call arguments')
    return {'type': 'function', 'function': {'name': function['name'], 'arguments': arguments}}


def native_event(result, clock, active=None):
    """Keep native calls separate from spoken text, including failed attempts."""
    event = dict(active) if active is not None else {'onset_seconds': clock, 'text': result.get('text', '')}
    event.update(response_text=event.get('text', ''), tool_calls=result.get('tool_calls') or [],
                 agent_attempted=True, agent_call_onset_seconds=clock,
                 agent_call_complete_seconds=clock if result.get('tool_generation_complete', True) else None,
                 agent_parse_status=('malformed' if result.get('tool_error') else 'structured'))
    for key in ('tool_error', 'raw_tool_text', 'tool_parse_valid', 'tool_schema_valid'):
        if key in result:
            event[key] = result[key]
    return event


def parse_event(event):
    text = event.get("text", "")
    response = event.get("response_text", text)
    calls = event.get("tool_calls")
    status = event.get("agent_parse_status", "structured" if calls is not None else "absent")
    if calls is None:
        candidate = text.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.S)
        if fence:
            candidate = fence.group(1)
        tags = re.fullmatch(r'\s*(?:<tool_call>\s*.*?\s*</tool_call>\s*)+', candidate, re.S)
        if tags:
            try:
                calls = [json.loads(payload) for payload in re.findall(
                    r'<tool_call>\s*(.*?)\s*</tool_call>', candidate, re.S)]
                status, response = 'structured', ''
            except (ValueError, TypeError):
                calls, status = [], 'malformed'
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            obj = None
            if calls is None and ('"tool_calls"' in text or '"agent_call"' in text or '"task_start"' in text or '<tool_call>' in text):
                status = "malformed"
        if isinstance(obj, dict) and "tool_calls" in obj:
            calls = obj["tool_calls"]
            response = obj.get("response", obj.get("content", ""))
            status = "structured"
        elif isinstance(obj, dict) and isinstance(obj.get("function"), dict) and obj["function"].get("name") in {"agent_call", "task_start"}:
            calls, response, status = [obj], "", "structured"
        elif isinstance(obj, dict) and isinstance(obj.get('name'), str) and 'arguments' in obj:
            calls, response, status = [obj], '', 'structured'
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        return {"response": response if isinstance(response, str) else "", "calls": [],
                "status": "malformed", "attempted": True}
    valid = []
    for call in calls:
        try:
            normalized = normalize_call(call)
        except (ValueError, TypeError):
            status = "malformed"
            continue
        valid.append(normalized)
    return {"response": response if isinstance(response, str) else "", "calls": valid,
            "status": status, "attempted": bool(calls) or bool(event.get('agent_attempted'))
            or status in {'malformed', 'truncated', 'duplicate_separator'}}
