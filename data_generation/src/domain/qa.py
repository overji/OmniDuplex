"""Pure adaptation of validated interaction responses."""
from typing import Any
from .common import timestamp_seconds


def adapt_responses(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapted = []
    for response in responses:
        messages = response.get("messages")
        if messages is None:
            adapted.append(response)
            continue
        if not isinstance(messages, list) or not messages:
            raise ValueError("agent response must contain messages")
        parent_fields = {
            key: value for key, value in response.items()
            if key not in {"trigger_time", "messages"}
        }
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError("agent message must be an object")
            timestamp = message.get("timestamp")
            timestamp_seconds(timestamp)
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("agent message content must be a string")
            entry = {"trigger_time": timestamp}
            if index == 0:
                entry["source_trigger_time"] = response.get("trigger_time")
                entry.update(parent_fields)
            entry["response"] = content
            entry.update({
                key: value for key, value in message.items()
                if key not in {"timestamp", "content"}
            })
            if index == 2 and message.get("role") == "assistant":
                entry["agent_result_time"] = messages[1]["timestamp"]
            adapted.append(entry)
    return sorted(adapted, key=lambda item: timestamp_seconds(item["trigger_time"]))


