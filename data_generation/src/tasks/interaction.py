"""Composable request generation and validation for the interaction matrix."""
import json
import math
import random
from dataclasses import asdict

from .. import config
from ..agent_tools import TOOLS, describe_tool
from ..catalog import ABILITIES, split_task
from ..common import format_mmss
from ..video_caption import load_caption, probe_video_duration
from . import _agent, _shared
from .annotations import validate_annotation

GenerationContext = _agent.GenerationContext
GenerationRequest = _shared.GenerationRequest


def validate_context(context, mode):
    duration = context.duration_seconds
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid duration")
    if context.question_position not in {"start", "non_start"} or context.preferred_modality not in {"audio", "vision"}:
        raise ValueError("invalid sampling context")
    if not isinstance(context.available_tools, (tuple, list)) or not all(isinstance(t, str) for t in context.available_tools):
        raise ValueError("available_tools must be a list of tool names")
    if len(set(context.available_tools)) != len(context.available_tools):
        raise ValueError("available_tools must be unique")
    if context.required_missing_tool is not None and not isinstance(context.required_missing_tool, str):
        raise ValueError("invalid missing tool")
    if context.initial_response_delay_seconds != 0:
        raise ValueError("v2 has no initial silent interval")
    if mode == "agent":
        if context.sample_type not in {"positive", "no_context", "missing_tool"}:
            raise ValueError("invalid agent sample type")
        if context.sample_type == "positive":
            lower = max(config.AGENT_DELAY_MIN_SECONDS, math.ceil(duration * config.AGENT_DELAY_MIN_FRACTION))
            upper = math.floor(duration * config.AGENT_DELAY_MAX_FRACTION)
            delay = context.agent_delay_seconds
            if lower <= upper and (isinstance(delay, bool) or not isinstance(delay, int) or not lower <= delay <= upper):
                raise ValueError("invalid backend delay")
            if lower > upper and delay is not None:
                raise ValueError("short video cannot supply backend delay")
        elif context.agent_delay_seconds is not None:
            raise ValueError("negative samples have no backend delay")
    elif context.sample_type != "positive" or context.available_tools or context.required_missing_tool is not None or context.agent_delay_seconds is not None:
        raise ValueError("non-agent context cannot contain agent capabilities")


class InteractionTask:
    GenerationContext = GenerationContext

    def __init__(self, name):
        self.name = name
        self.ability, self.mode = split_task(name)

    def build_request(self, path, video_path):
        video = video_path.expanduser().resolve()
        duration = probe_video_duration(video)
        caption = load_caption(path, duration)
        sample_type = config.choose_agent_sample_type() if self.mode == "agent" else "positive"
        missing = random.choice(tuple(TOOLS)) if sample_type != "positive" else None
        available = tuple(t for t in TOOLS if t != missing) if self.mode == "agent" and sample_type != "no_context" else ()
        context = GenerationContext(
            duration_seconds=duration,
            preferred_modality=config.choose_preferred_modality(),
            question_position=config.choose_question_position(duration),
            agent_delay_seconds=config.choose_agent_delay(duration) if self.mode == "agent" and sample_type == "positive" else None,
            initial_response_delay_seconds=0,
            sample_type=sample_type, available_tools=available, required_missing_tool=missing,
        )
        spec = ABILITIES[self.ability]
        root = _shared.PROMPT_ROOT
        mode_path = root / ("agent/generation.md" if self.mode == "agent" else f"{self.mode}.md")
        system = (root / "common.md").read_text() + "\n" + mode_path.read_text()
        system += f"\nTask: {self.name}\nAbility: {spec.description}\nRequired annotation: {spec.annotation}\n"
        system += f"Vision must be required: {spec.visual_required}. Duration: {format_mmss(duration)}.\n"
        system += "Sampled constraints (immutable): " + json.dumps(asdict(context))
        system += "\nquestion_position=start means question_time=00:00; non_start means 00:00 < question_time < duration.\n"
        user = {"caption": caption}
        if self.mode == "agent":
            user["frontend_context"] = _agent.frontend_context(context)
            system += "\nBackend delay is agent_delay_seconds; if null in a positive branch, return skip.\n"
            system += "Generator-only full catalog, NOT frontend availability:\n"
            system += "\n".join(f"- {name}: {describe_tool(name)}" for name in TOOLS)
            if missing:
                system += f"\nThe requested work MUST genuinely require missing tool {missing}. Do not substitute reasoning or another tool."
        return GenerationRequest(system, json.dumps(user, ensure_ascii=False, indent=2), context, video)

    def validate_result(self, result, context):
        validate_context(context, self.mode)
        if not isinstance(result, dict):
            raise ValueError("sample must be an object")
        for key, expected in (("ability", self.ability), ("mode", self.mode), ("task", self.name)):
            if key in result and result[key] != expected:
                raise ValueError(f"model changed {key}")
        if self.mode != "agent" and any(key in result for key in ("annotation", "frontend_context", "required_tools", "available_tools", "work_type", "sample_type")):
            raise ValueError("non-agent samples cannot carry Agent execution metadata")
        if self.mode == "agent":
            result = _agent.validate_result(result, context)
        if result.get("status") == "skip":
            return {"status": "skip", "reason": _shared.require_text(result.get("reason"), "reason"),
                    "ability": self.ability, "mode": self.mode}
        negative = self.mode == "agent" and context.sample_type != "positive"
        if not negative and "annotation" in result:
            raise ValueError("separate future annotation is only for capability negatives")
        if negative:
            annotation = result.get("annotation")
            if not isinstance(annotation, dict) or "messages" in annotation or "response" in annotation:
                raise ValueError("negative requires annotation-only real trigger evidence")
            observations = [annotation]
        else:
            observations = result.get("responses")
        shadow = {**result, "responses": [
            {**item, "response": "annotation"} if (negative or self.mode == "agent") and isinstance(item, dict) else item
            for item in observations
        ]} if isinstance(observations, list) else result
        _, question = _shared.validate_common(shadow, context, min_responses=2 if self.mode == "rtd" else 1)
        if ABILITIES[self.ability].visual_required and result["vision_dependency"] != "required":
            raise ValueError("ability requires vision")
        seen = set()
        previous = None
        for item in observations:
            if self.mode != "agent" and any(key in item for key in ("messages", "tool_calls", "role", "outcome")):
                raise ValueError("direct responses cannot contain Agent calls or results")
            validate_annotation(item, context, self.ability, self.mode, question, previous)
            if self.mode != "rtd":
                episode = item["trigger"]["episode_id"]
                if episode in seen:
                    raise ValueError("duplicate trigger episode")
                seen.add(episode)
            previous = item
        return {**result, "ability": self.ability, "mode": self.mode}
