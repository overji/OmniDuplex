"""Small valid annotations and simulated traces for contract tests."""
import json
from dataclasses import asdict

from data_generation.src.agent_tools import TASK_TOOLS, WRITE_TOOLS
from data_generation.src.common import SAMPLE_SCHEMA, format_mmss
from data_generation.src.tasks import _agent


def caption(duration=120):
    return [
        {"start": format_mmss(start), "end": format_mmss(min(start + 30, duration)),
         "caption": "A person puts a parcel down.", "visual": "A parcel on a table.",
         "audio": "A thump.", "speech": "None"}
        for start in range(0, duration, 30)
    ]


def agent_context(task="event_agent", sample_type="positive"):
    catalog = TASK_TOOLS[task]
    missing = catalog[0] if sample_type != "positive" else None
    return _agent.GenerationContext(
        120, "vision", "start", agent_delay_seconds=20 if sample_type == "positive" else None,
        initial_response_delay_seconds=10 if sample_type == "positive" else None,
        sample_type=sample_type,
        available_tools=() if sample_type == "no_context" else tuple(t for t in catalog if t != missing),
        required_missing_tool=missing,
    )


def agent_sample(task="event_agent", sample_type="positive", work_type="tools"):
    context = agent_context(task, sample_type)
    tool = TASK_TOOLS[task][0]
    required = [] if work_type == "reasoning" else [tool]
    result = {
        "status": "ok", "question": "When the next parcel arrives, create a todo for it.",
        "question_time": "00:00", "work_type": work_type, "required_tools": required,
    }
    if sample_type != "positive":
        result["responses"] = [{"trigger_time": "00:00", "response": "I cannot create todos with my available tools.",
                                "refusal_reason": "Todo creation is unavailable.", "missing_tools": [tool]}]
        return result, context
    trigger = {"kind": task.removesuffix("_agent"), "satisfied": True,
               "evidence": [{"time": "00:20", "fact": "The specified parcel is visible."}]}
    if task == "condition_agent":
        trigger["conditions"] = [{"condition": "The parcel is present", "satisfied": True,
                                   "time": "00:20", "evidence": "Visible on the table."}]
    elif task == "count_agent":
        trigger.update(counting_mode="distinct", threshold=3, previous_count=2, count=3,
                       counting_basis="Three unique tasks.", deduplication_rule="Merge repeated task names.")
    elif task == "target_agent":
        trigger.update(target="parcel", attributes_match=True)
    arguments = {
        "instruction": "Process the parcel task.", "observed_context": "00:20: parcel appears.",
        "user_goal": "Organize handling the parcel.", "constraints_and_authorization": "Only this task.",
        "suggested_capabilities": required, "expected_output": "Return a verified result.",
        "extracted_parameters": [{"name": "item", "value": "parcel", "evidence_time": "00:20", "evidence": "Visible parcel."}],
    }
    envelope = {
        "simulated": True, "status": "success", "summary": "Task processed.",
        "observed_facts": ["Parcel arrived."], "tool_information": [] if not required else ["Task created."],
        "reasoning": ["Handle the parcel after the meeting."] if work_type != "tools" else [],
        "uncertainties": [], "sources": [{"tool": tool, "evidence": "Matching record found."}] if required and tool not in WRITE_TOOLS else [], "artifacts": [],
        "actions": [{"tool": tool, "operation": "create", "status": "completed", "evidence": "Record task-1 exists."}] if required and tool in WRITE_TOOLS else [],
    }
    result.update(audio_dependency="none", vision_dependency="required", responses=[{
        "trigger_time": "00:20", "trigger_type": "visual", "trigger_type_reason": "Parcel is visible.",
        "event_description": "Parcel arrives.", "trigger": trigger,
        "messages": [
            {"role": "assistant", "timestamp": "00:20", "content": "I will handle it.",
             "tool_calls": [{"type": "function", "function": {"name": "agent_call", "arguments": json.dumps(arguments)}}]},
            {"role": "tool", "name": "agent_call", "timestamp": "00:40", "content": json.dumps(envelope)},
            {"role": "assistant", "timestamp": "00:40", "content": "The task is handled."},
        ],
    }])
    return result, context


def saved_agent(task="event_agent", sample_type="positive"):
    result, context = agent_sample(task, sample_type)
    result = _agent.validate_result(result, context, task)
    result.update(schema=SAMPLE_SCHEMA, task=task, video="test.mp4", generation_context=asdict(context))
    return result
