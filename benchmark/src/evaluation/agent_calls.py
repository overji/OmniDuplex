"""Conservative extraction of explicit calls, never reconstruction from references."""
import json
import re


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
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            obj = None
            if '"tool_calls"' in text or '"agent_call"' in text:
                status = "malformed"
        if isinstance(obj, dict) and "tool_calls" in obj:
            calls = obj["tool_calls"]
            response = obj.get("response", obj.get("content", ""))
            status = "structured"
        elif isinstance(obj, dict) and isinstance(obj.get("function"), dict) and obj["function"].get("name") == "agent_call":
            calls, response, status = [obj], "", "structured"
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        return {"response": response if isinstance(response, str) else "", "calls": [],
                "status": "malformed", "attempted": True}
    valid = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            status = "malformed"
            continue
        function = dict(call["function"])
        arguments = function.get("arguments")
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict) or not isinstance(function.get("name"), str):
                raise ValueError("invalid function")
        except (ValueError, TypeError):
            status = "malformed"
            continue
        valid.append({**call, "function": {**function, "arguments": arguments}})
    return {"response": response if isinstance(response, str) else "", "calls": valid,
            "status": status, "attempted": bool(calls) or bool(event.get('agent_attempted'))
            or status in {'malformed', 'truncated', 'duplicate_separator'}}
