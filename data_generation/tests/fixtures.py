"""Small independent v2 annotations and simulated traces for contract tests."""
import copy
import json
from dataclasses import asdict

from data_generation.src.domain.agent_tools import TOOLS
from data_generation.src.domain.common import SAMPLE_SCHEMA, format_mmss
from data_generation.src.application.registry import get_task
from data_generation.src.domain.tasks import _agent


def caption(duration=120):
    return [
        {"start": format_mmss(start), "end": format_mmss(min(start + 30, duration)),
         "caption": "A person puts a parcel down.", "visual": "A parcel on a table.",
         "audio": "A thump.", "speech": "None"}
        for start in range(0, duration, 30)
    ]


def observation(ability, mode, time=20):
    timestamp = format_mmss(time)
    details = {
        "attribute_perception": {"target": "shirt", "attributes": "red", "uncertainty": "none"},
        "grounding": {"event": "clap", "localization": "moment", "start": timestamp, "end": timestamp},
        "action_perception": {"actor": "person", "action": "clap", "phase": "onset", "uncertainty": "none"},
        "object_perception": {"target": "parcel", "category": "package", "position": "left", "uncertainty": "none"},
        "emotion_recognition": {"subject": "speaker", "emotion": "happy", "basis": "smiling", "confidence": 0.8, "claim_type": "inference"},
        "event_understanding": {"event": "delivery", "participants": "courier", "relations": "parcel handed over"},
        "text_rich_understanding": {"text": "EXIT", "text_source": "scene", "interpretation": "exit here", "uncertainty": "none"},
        "object_counting": {"counting_mode": "current", "count": 2, "target_ids": ["a", "b"], "counting_basis": "visible cups", "deduplication_rule": "track each cup"},
        "action_counting": {"counting_mode": "cumulative", "counting_start": "00:00", "count": 1 if time == 20 else 2,
                            "occurrences": [{"id": "a", "time": "00:20"}] + ([] if time == 20 else [{"id": "b", "time": timestamp}]), "counting_basis": "one complete clap"},
        "spatial_understanding": {"target": "ball", "reference": "table edge", "relation": "near", "uncertainty": "no metric scale"},
        "prospective_reasoning": {"prediction": "ball may fall", "basis": "rolling toward edge", "confidence": 0.8, "claim_type": "prediction", "update_reason": "closer to edge"},
        "causal_reasoning": {"event": "cup fell", "candidate_causes": ["hand contact", "unstable support"], "basis": "hand near cup", "uncertainty": "contact partly hidden", "claim_type": "inference"},
    }
    item = {"window_start": timestamp, "trigger_time": timestamp, "window_end": timestamp, "response": f"Observed update at {timestamp}.",
            "trigger_type": "visual", "trigger_type_reason": "visible change",
            "evidence": [{"time": timestamp, "available_time": timestamp, "fact": "The change is visible."}],
            "details": copy.deepcopy(details[ability])}
    if mode != "rtd":
        item["trigger"] = {"kind": "event", "satisfied": True, "episode_id": f"event-{time}", "confirmed_time": timestamp, "definition": "each relevant change"}
    return item


def agent_context(task="event_understanding_agent", sample_type="positive"):
    missing = "todo_create" if sample_type != "positive" else None
    return _agent.GenerationContext(
        120, "vision", "start", agent_delay_seconds=20 if sample_type == "positive" else None,
        initial_response_delay_seconds=0, sample_type=sample_type,
        available_tools=() if sample_type == "no_context" else tuple(t for t in TOOLS if t != missing),
        required_missing_tool=missing,
    )


def agent_sample(task="event_understanding_agent", sample_type="positive", work_type="tools"):
    ability = task.removesuffix("_agent")
    context = agent_context(task, sample_type)
    required = [] if work_type == "reasoning" else ["todo_create"]
    result = {"status": "ok", "question": "When each relevant change happens, create a todo for it.",
              "question_time": "00:00", "work_type": work_type, "required_tools": required,
              "audio_dependency": "none", "vision_dependency": "required"}
    item = observation(ability, "agent")
    if sample_type != "positive":
        result["annotations"] = [item]
        result["responses"] = [{"window_start": "00:00", "trigger_time": "00:00", "window_end": "00:00", "response": "I cannot create todos with my available tools.",
                                "refusal_reason": "Todo creation is unavailable.", "missing_tools": ["todo_create"]}]
        return result, context
    arguments = {
        "instruction": "Process the observed task.", "observed_context": "00:20: change observed.",
        "user_goal": "Organize handling it.", "constraints_and_authorization": "Only this task.",
        "suggested_capabilities": required, "expected_output": "Return a verified result.",
        "extracted_parameters": [{"name": "event", "value": "change", "evidence_time": "00:20", "evidence": "Visible change."}],
    }
    envelope = {
        "simulated": True, "status": "success", "summary": "Task processed.",
        "observed_facts": ["Change observed."], "tool_information": [] if not required else ["Task created."],
        "reasoning": ["Handle it after the meeting."] if work_type != "tools" else [],
        "uncertainties": [], "sources": [], "artifacts": [],
        "actions": [{"tool": "todo_create", "arguments": {"title": "Handle change", "owner": "user", "deadline": "unspecified"},
                     "operation": "create", "status": "completed", "evidence": "Record task-1 exists."}] if required else [],
    }
    item["messages"] = [
        {"role": "assistant", "timestamp": "00:20", "content": "I will handle it.",
         "tool_calls": [{"type": "function", "function": {"name": "agent_call", "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "name": "agent_call", "timestamp": "00:40", "content": json.dumps(envelope)},
        {"role": "assistant", "timestamp": "00:40", "content": "The task is handled.", "outcome": "success"},
    ]
    item["response"] = item["messages"][0]["content"]
    result["responses"] = [item]
    return result, context


def task_sample(task):
    if task.endswith("_agent"):
        return agent_sample(task)
    ability, mode = task.rsplit("_", 1)
    return {"status": "ok", "question": "Track the relevant changes.", "question_time": "00:00",
            "audio_dependency": "none", "vision_dependency": "required",
            "responses": [observation(ability, mode, t) for t in ([20, 40] if mode == "rtd" else [20])]}, _agent.GenerationContext(120, "vision", "start")


def saved_agent(task="event_understanding_agent", sample_type="positive"):
    result, context = agent_sample(task, sample_type)
    result = get_task(task).validate_result(result, context)
    result.update(schema=SAMPLE_SCHEMA, task=task, video="test.mp4", generation_context=asdict(context))
    return result
