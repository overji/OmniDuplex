"""Build per-sample system prompts without exposing benchmark annotations."""
from __future__ import annotations

import json
import sys

from benchmark.data.paths import ROOT
sys.path.insert(0, str(ROOT.parent / "data_generation" / "src"))
from domain.agent_tools import TOOLS, describe_tool  # noqa: E402


def available_function_schemas(sample: dict) -> list:
    """Expose only the frontend delegate when backend capabilities exist."""
    if not sample.get('available_tools'):
        return []
    name = 'task_start' if sample.get('_agent_call_schema') == 'task-start' else 'agent_call'
    return [tool for tool in (sample.get('frontend_context') or {}).get('tools', [])
            if (tool.get('function') or tool).get('name') == name]


def build_system_prompt(base_prompt: str, sample: dict, native_protocol: str = "") -> str:
    """Add only frontend-provided available capabilities and schema to system context."""
    frontend = sample.get("frontend_context") or {}
    available = sample.get("available_tools") or []
    if len(available) != len(set(available)) or any(name not in TOOLS for name in available):
        raise ValueError(f"Invalid available_tools in sample {sample.get('id', '?')}")

    sections = [base_prompt.strip()]
    if native_protocol.strip():
        sections.append(native_protocol.strip())
    application = frontend.get("system_prompt", "").strip()
    if application:
        sections.append("Application capability instructions:\n" + application)
    if available:
        sections.append("Available backend tools (only these may be delegated):\n" + "\n".join(
            f"- {name}: {describe_tool(name)}" for name in available
        ))
    # Preserve the supplied agent_call schema, filtering any function definitions
    # whose name is not available. Empty availability deliberately means no schema.
    schemas = available_function_schemas(sample)
    if schemas:
        sections.append("Available function schema:\n" + json.dumps(schemas, ensure_ascii=False))
    return "\n\n".join(section for section in sections if section)


def user_request(sample: dict) -> str:
    return f"User request (issued now at video time {sample['question_time']}): {sample['question']}"
