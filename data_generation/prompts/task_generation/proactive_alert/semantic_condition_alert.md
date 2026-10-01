You create one dataset sample for the task "Semantic Condition Alert".

## Response-time causality and no future leakage

You may inspect the complete video to choose the question and target and to audit
coverage. That full-video access must not leak later information into a
timestamped response: each response represents only what is knowable by its own
`trigger_time`. Every field nested in that response, including response text,
counts, names or identities, locations, evidence, interpretations, event lists,
trigger reasons and descriptions, and other metadata, may use only information
observable or independently confirmed no later than `trigger_time`. A historical
summary emitted at `question_time` uses `question_time` as its cutoff. Later
video evidence may support later responses but must never revise, complete,
rename, disambiguate, or otherwise backfill an earlier response; use generic,
unknown, or explicitly uncertain wording when cutoff evidence is insufficient.

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is the primary source of truth; supplied dense captions are fallible candidate indexes.

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

The requester is the video viewer, the people shown or heard are video
participants, and the assistant answers the viewer. First-person wording in the
question may express the viewer's needs, preferences, or wish to receive results;
it must not attribute a participant's actions, speech, location, possessions, or
experiences to the viewer. Describe those participants in the third person or
by an observable role, even for POV footage. Keep quoted speech attributed to
its original speaker. In assistant replies, "you/your" refers to the viewer,
not a video participant.

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

## Completed-event timing and timestamp representation

Before generating any response, wait until the triggering event's observable
boundary has ended and the event is confirmed. Never respond while the event is
ongoing, from its onset alone, or by relying on future evidence. `trigger_time`
and every event-time field represent the response after that completed boundary.
Represent video times in whole-second buckets by discarding fractional seconds;
never round or carry an event into a later bucket. Preserve chronological order
and apply this task's existing deduplication when multiple observations share a
bucket. If completion cannot be confirmed, remain silent or return `skip`.

## Independent video evidence audit

First inspect the original video to choose candidates; then consult dense captions only as fallible navigation aids, and return to the video to verify each
claim and onset. Reject false annotations, correct inaccurate times, and include
verified omissions. Caption fields are not independent corroboration. Confidence scores are not proof. Retain and examine supplied
uncertainties in the captions; unsupported caption claims remain unverified.
Never copy a caption claim or timestamp without checking the original video.
Distinguish visible actions, audible sounds, spoken claims, and on-screen text:
a narrator mentioning an action does not prove that action is visible.
Whole-video inspection is an annotation audit only. Each response and agent_call
may use only evidence observable at or before its own trigger_time or message
timestamp; never use later video evidence. User-visible response text must not
include an MM:SS video timestamp; keep video timestamps only in structured time
fields. Remain silent without a verified trigger; if the task cannot be grounded,
return skip.

Input: original audiovisual video and the complete timestamped `caption` array.
Caption is supplementary reference only; the original video and audio are authoritative.
Verify each claim and timestamp against the video. Never use future caption content in an earlier response.
