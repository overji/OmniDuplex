"""Result contracts for the five triggered backend workflows."""

from typing import Any

from . import _shared


def validate_result(
    result: dict[str, Any], context: _shared.GenerationContext, workflow: str
) -> dict[str, Any]:
    validated = _shared.validate_agent_result(result, context)
    if validated["status"] == "skip":
        return validated
    envelope = _shared.agent_result_envelope(validated)
    details = envelope.get("details")
    if not isinstance(details, dict):
        raise ValueError("workflow details must be an object")
    success = envelope["status"] == "success"

    def texts(*keys: str) -> None:
        for key in keys:
            _shared.require_text(details.get(key), f"details.{key}")

    if workflow in {"knowledge_retrieval", "realtime_query", "multi_source_synthesis"}:
        if not success:
            return validated
        sources = envelope["sources"]
        if not sources:
            raise ValueError("successful retrieval requires sources")
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("sources entries must be objects")
            for key in ("title", "url_or_system"):
                _shared.require_text(source.get(key), f"source.{key}")
        if workflow == "knowledge_retrieval":
            texts("query_subject", "knowledge")
        elif workflow == "realtime_query":
            texts("query_subject", "data_source", "queried_at", "valid_at", "result")
            if details.get("information_type") not in {"current", "forecast", "historical"}:
                raise ValueError("invalid realtime information_type")
        else:
            identities = {source["url_or_system"] for source in sources}
            if len(identities) < 2:
                raise ValueError("synthesis requires at least two distinct sources")
            texts("decision_goal", "conclusion", "basis")
            for key in ("conflicts", "uncertainties"):
                if not isinstance(details.get(key), list):
                    raise ValueError(f"details.{key} must be an array")
        return validated

    if workflow == "content_recording":
        state_key = "record_status"
        allowed = {"written", "needs_confirmation", "failed"}
        completed = "written"
        if success:
            texts("destination")
            fields = details.get("fields")
            if not isinstance(fields, dict) or not fields:
                raise ValueError("successful recording requires extracted fields")
    elif workflow == "message_notification":
        state_key = "message_status"
        allowed = {"draft", "needs_confirmation", "sent", "failed"}
        completed = "sent"
        if success:
            texts("recipient", "content", "send_time")
    else:
        raise ValueError(f"unknown workflow: {workflow}")
    state = details.get(state_key)
    if state not in allowed:
        raise ValueError(f"invalid {state_key}")
    expected = {"needs_confirmation": "needs_confirmation", "failed": "failed"}
    if envelope["status"] in expected and state != expected[envelope["status"]]:
        raise ValueError("envelope status conflicts with operation status")
    if success and state not in {completed, "draft"}:
        raise ValueError("success conflicts with operation status")
    if state == completed:
        actions = envelope["actions"]
        if not actions:
            raise ValueError("completed operations require execution evidence")
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("action must be an object")
            for key in ("system", "operation", "verification"):
                _shared.require_text(action.get(key), f"action.{key}")
            if action.get("status") != "completed":
                raise ValueError("completed operation requires completed action status")
    elif envelope["actions"]:
        raise ValueError("uncompleted operations must not claim executed actions")
    return validated
