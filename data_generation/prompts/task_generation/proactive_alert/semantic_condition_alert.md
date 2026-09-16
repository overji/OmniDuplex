You create one dataset sample for the task "Semantic Condition Alert".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is the primary source of truth; supplied events and items are fallible candidate indexes.

{{QUESTION_TIME_POLICY}}

Create one English standing instruction about an abstract, contextual condition.
Examples: "Let me know when the discussion seems to be at an impasse", "Tell me
when the environment starts to feel chaotic", or "Let me know when there is a
good photo opportunity". Select a condition relevant to this video's content.
Prefer evidence requiring {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}}
when necessary.

A valid condition requires interpretation of context, relationships, or developing
patterns. Invalid examples: "When that car appears again", "When the door opens",
or "When someone says budget". Renaming a single concrete event as an abstract
condition is not sufficient. Do not infer hidden intentions or emotions.
Set semantic_condition to the abstract state and observable_criteria to concrete
cues that operationalize it without disclosing future outcomes in the question.
For example, an impasse may be supported by repeated incompatible proposals and
explicit inability to agree, not merely a pause. A photo opportunity may depend
on composition, visibility, and lighting, not simply an object entering the frame.

Use earlier context as needed, but alert only when sufficient observable evidence
has accumulated at or after question_time. condition_evidence must explain how
the cues jointly support the abstract condition at that time. Count a sustained
condition once until it ends and begins again. Describe borderline evidence as
uncertain; if no defensible abstract condition exists, skip rather than falling
back to an ordinary event alert.

At least one dependency must be required, and each trigger_type must contain every required modality. All natural-language fields must be English.

Return exactly one JSON object:

{
  "status": "ok",
  "question": "Let me know when ...",
  "question_time": "MM:SS",
  "semantic_condition": "The discussion reaches an impasse",
  "observable_criteria": ["Repeated incompatible proposals", "Explicit inability to agree"],
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [{
    "trigger_time": "MM:SS",
    "response": "...",
    "condition_evidence": "...",
    "confidence": 0.0,
    "uncertainty": "... or none",
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If no defensible semantic condition has a later trigger, return only {"status":"skip","reason":"..."}.

## Independent video evidence audit

First inspect the original video to choose candidates; then consult events and
objects only as fallible navigation aids, and return to the video to verify each
claim and onset. Reject false annotations, correct inaccurate times, and include
verified omissions. Events and objects derived from those events are not independent
corroboration. Confidence scores are not proof. Retain and examine supplied
uncertainties; legacy metadata without evidence fields is unverified reference.
Never copy a caption claim or timestamp without checking the original video.
Distinguish visible actions, audible sounds, spoken claims, and on-screen text:
a narrator mentioning an action does not prove that action is visible.
Whole-video inspection is an annotation audit only. Each response and agent_call
may use only evidence available by its own timestamp. Remain silent without a
verified trigger; if the task cannot be grounded, return skip.
