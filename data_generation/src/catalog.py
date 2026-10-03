"""The single source of truth for the 12 × 3 interaction matrix."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Ability:
    description: str
    visual_required: bool
    annotation: str
    text_fields: tuple[str, ...]


ABILITIES = {
    "attribute_perception": Ability(
        "Track target attributes (color, material, category, state) and their changes. Distinguish similar attributes and uncertain identities.", True,
        'details: {"target":"...", "attributes":"...", "uncertainty":"..."}', ('target', 'attributes', 'uncertainty')),
    "grounding": Ability(
        "Locate an event in time. Distinguish onset, ongoing activity and confirmed completion. Never infer an end from future frames.", False,
        'details: {"event":"...", "localization":"moment|span", "start":"MM:SS", "end":"MM:SS"}; moment uses equal start/end, span requires start < end.', ('event',)),
    "action_perception": Ability(
        "Recognize actions and their onset, continuation or end; distinguish similar motions and occlusion.", True,
        'details: {"actor":"...", "action":"...", "phase":"onset|ongoing|end", "uncertainty":"..."}', ('actor', 'action', 'uncertainty')),
    "object_perception": Ability(
        "Identify newly appearing people or objects, category, distinguishing specifications and position. Do not invent related but absent objects.", True,
        'details: {"target":"...", "category":"...", "position":"...", "uncertainty":"..."}', ('target', 'category', 'position', 'uncertainty')),
    "emotion_recognition": Ability(
        "Infer a specified person's emotion from voice prosody, expression and context, not merely emotional words. All emotions are eligible, not only negative ones.", False,
        'details: {"subject":"...", "emotion":"...", "basis":"...", "confidence":0.7, "claim_type":"inference"}', ('subject', 'emotion', 'basis')),
    "event_understanding": Ability(
        "Explain ongoing events, participants, progression and relations. Distinguish this semantic understanding from naming an isolated movement.", False,
        'details: {"event":"...", "participants":"...", "relations":"..."}', ('event', 'participants', 'relations')),
    "text_rich_understanding": Ability(
        "Read and interpret visible text, subtitles or signs in context. Audio can help but cannot replace actually visible text. Distinguish current information from announcements of future events.", True,
        'details: {"text":"verbatim visible text", "text_source":"screen|scene", "interpretation":"...", "uncertainty":"..."}', ('text', 'interpretation', 'uncertainty')),
    "object_counting": Ability(
        "Count objects currently present, not cumulative appearances. Track identities through occlusion and reappearance without double counting; explain ambiguous visibility.", True,
        'details: {"counting_mode":"current", "count":2, "target_ids":["a","b"], "counting_basis":"...", "deduplication_rule":"..."}', ('counting_basis', 'deduplication_rule')),
    "action_counting": Ability(
        "Accumulate distinct completed action or speech/sound occurrences. Define separable event boundaries and a counting origin; do not count frames of one sustained action repeatedly.", False,
        'details: {"counting_mode":"cumulative", "counting_start":"MM:SS", "count":2, "occurrences":[{"id":"a","time":"MM:SS"},{"id":"b","time":"MM:SS"}], "counting_basis":"..."}; list all counted occurrences so far.', ('counting_basis',)),
    "spatial_understanding": Ability(
        "Track relative positions, directions and spatial relations. Do not invent metric distances without scale; distinguish screen coordinates from world positions.", True,
        'details: {"target":"...", "reference":"...", "relation":"...", "uncertainty":"..."}', ('target', 'reference', 'relation', 'uncertainty')),
    "prospective_reasoning": Ability(
        "Predict possible next developments from current evidence, including opportunities and ordinary outcomes, not only risks. Mark predictions as uncertain; never use the actual future as an observed fact.", False,
        'details: {"prediction":"...", "basis":"...", "confidence":0.7, "claim_type":"prediction", "update_reason":"..."}', ('prediction', 'basis', 'update_reason')),
    "causal_reasoning": Ability(
        "Explain why an observed event occurred. Separate observed facts from candidate causes, rank alternatives and state uncertainty instead of equating correlation with causation.", False,
        'details: {"event":"...", "candidate_causes":["most supported cause", "alternative"], "basis":"...", "uncertainty":"...", "claim_type":"inference"}', ('event', 'basis', 'uncertainty')),
}
MODES = {
    "rtd": "Continuous time-aligned updates with at least two meaningful observations.",
    "pr": "Notify on every independently satisfied occurrence after the instruction.",
    "agent": "Delegate useful tool work, complex reasoning, or both on the next matching occurrence.",
}
TASK_CATALOG = {
    f"{ability}_{mode}": f"{spec.description} {description}"
    for ability, spec in ABILITIES.items() for mode, description in MODES.items()
}


def split_task(task: str) -> tuple[str, str]:
    if task not in TASK_CATALOG:
        raise ValueError(f"unsupported task: {task}")
    return tuple(task.rsplit("_", 1))
