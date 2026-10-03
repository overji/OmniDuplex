"""Structural and temporal contracts for observable evidence, not semantic judging."""
from ..catalog import ABILITIES
from ._shared import checked_time, require_count, require_text


def object_field(item, key):
    value = item.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def array_field(item, key):
    value = item.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty array")
    return value


def predicate(value, operator, threshold):
    operations = {"ge": value >= threshold, "gt": value > threshold,
                  "le": value <= threshold, "lt": value < threshold, "eq": value == threshold}
    if operator not in operations:
        raise ValueError("invalid count operator")
    return operations[operator]


def validate_annotation(item, context, ability, mode, question, previous=None):
    if not isinstance(item, dict):
        raise ValueError("annotation must be an object")
    time = checked_time(item.get("trigger_time"), context, "trigger_time")
    if time < question:
        raise ValueError("annotation precedes question")
    for evidence in array_field(item, "evidence"):
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be objects")
        require_text(evidence.get("fact"), "evidence fact")
        observed = checked_time(evidence.get("time"), context, "evidence time")
        available = checked_time(evidence.get("available_time"), context, "available_time")
        if not observed <= available <= time:
            raise ValueError("evidence must be available by response time")
    if mode == "rtd":
        start = checked_time(item.get("window_start"), context, "window_start")
        end = checked_time(item.get("window_end"), context, "window_end")
        if not question <= start <= end == time:
            raise ValueError("invalid RTD window")
        if "trigger" in item:
            raise ValueError("RTD must not contain a discrete trigger")
        if previous and (start < checked_time(previous["window_end"], context, "previous end")
                         or time <= checked_time(previous["trigger_time"], context, "previous time")
                         or item["response"] == previous["response"]):
            raise ValueError("RTD updates must advance without overlap or exact repetition")
    else:
        trigger = object_field(item, "trigger")
        kind = trigger.get("kind")
        if kind not in {"event", "condition", "count", "target"} or trigger.get("satisfied") is not True:
            raise ValueError("invalid trigger kind or satisfaction")
        require_text(trigger.get("episode_id"), "episode_id")
        require_text(trigger.get("definition"), "trigger definition")
        if checked_time(trigger.get("confirmed_time"), context, "confirmed_time") != time:
            raise ValueError("trigger cannot be delayed after confirmation")
        if previous:
            earlier = checked_time(previous["trigger_time"], context, "previous trigger")
            if time <= earlier:
                raise ValueError("trigger episodes must advance in time")
            if kind != previous["trigger"]["kind"] or trigger["definition"] != previous["trigger"]["definition"]:
                raise ValueError("one standing instruction requires a consistent criterion")
            if kind != "event" and not earlier < checked_time(trigger.get("reset_time"), context, "reset_time") < time:
                raise ValueError("persistent conditions must reset before retriggering")
        if kind == "condition":
            for condition in array_field(trigger, "conditions"):
                if not isinstance(condition, dict) or condition.get("satisfied") is not True:
                    raise ValueError("all conditions must be satisfied")
                require_text(condition.get("condition"), "condition")
                require_text(condition.get("evidence"), "condition evidence")
                if checked_time(condition.get("time"), context, "condition time") > time:
                    raise ValueError("future condition evidence")
        if kind == "target":
            require_text(trigger.get("target"), "target")
            if trigger.get("attributes_match") is not True:
                raise ValueError("target must match attributes")
        if kind == "count":
            before = require_count(trigger.get("previous_count"), "previous_count")
            count = require_count(trigger.get("count"), "count")
            threshold = require_count(trigger.get("threshold"), "threshold")
            if predicate(before, trigger.get("operator"), threshold) or not predicate(count, trigger.get("operator"), threshold):
                raise ValueError("count predicate must transition from false to true")

    details = object_field(item, "details")
    for key in ABILITIES[ability].text_fields:
        require_text(details.get(key), key)
    if ability == "grounding":
        start = checked_time(details.get("start"), context, "grounding start")
        end = checked_time(details.get("end"), context, "grounding end")
        if not start <= end <= time:
            raise ValueError("grounding end cannot use future evidence")
        if details.get("localization") not in {"moment", "span"} or (
            (details["localization"] == "moment") != (start == end)
        ):
            raise ValueError("invalid moment/span boundaries")
    if ability == "action_perception" and details.get("phase") not in {"onset", "ongoing", "end"}:
        raise ValueError("invalid action phase")
    if ability == "text_rich_understanding" and details.get("text_source") not in {"screen", "scene"}:
        raise ValueError("invalid text source")
    if ability in {"emotion_recognition", "prospective_reasoning", "causal_reasoning"}:
        expected = "prediction" if ability == "prospective_reasoning" else "inference"
        if details.get("claim_type") != expected:
            raise ValueError("interpretations must not be labeled observed facts")
        if ability != "causal_reasoning":
            confidence = details.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not 0 <= confidence <= 1:
                raise ValueError("confidence must be in [0,1]")
        else:
            for cause in array_field(details, "candidate_causes"):
                require_text(cause, "candidate cause")
    if ability in {"object_counting", "action_counting"}:
        count = require_count(details.get("count"), "count")
        if mode != "rtd" and item["trigger"]["kind"] == "count" and item["trigger"]["count"] != count:
            raise ValueError("trigger count differs from annotated count")
        if ability == "object_counting":
            ids = details.get("target_ids")
            if details.get("counting_mode") != "current" or not isinstance(ids, list):
                raise ValueError("object counting requires current identities")
            for target in ids:
                require_text(target, "target_id")
            if count != len(ids) or len(set(ids)) != len(ids):
                raise ValueError("object count must equal unique current targets")
        else:
            start = checked_time(details.get("counting_start"), context, "counting_start")
            occurrences = details.get("occurrences")
            if details.get("counting_mode") != "cumulative" or not isinstance(occurrences, list) or count != len(occurrences):
                raise ValueError("action count must match cumulative occurrences")
            if start > time:
                raise ValueError("counting origin is in the future")
            ids, times = [], []
            for occurrence in occurrences:
                if not isinstance(occurrence, dict):
                    raise ValueError("occurrence must be an object")
                ids.append(require_text(occurrence.get("id"), "occurrence id"))
                when = checked_time(occurrence.get("time"), context, "occurrence time")
                if not start <= when <= time:
                    raise ValueError("counted occurrence outside observed interval")
                times.append(when)
            if len(set(ids)) != len(ids) or times != sorted(times):
                raise ValueError("occurrences must be unique and chronological")
            if previous:
                prior = previous["details"]
                if details["counting_start"] != prior["counting_start"] or occurrences[:prior["count"]] != prior["occurrences"] or count <= prior["count"]:
                    raise ValueError("cumulative updates must preserve and extend history")
    if ABILITIES[ability].visual_required and "visual" not in item.get("trigger_type", "").split("+"):
        raise ValueError("this cognitive ability requires visual evidence")
