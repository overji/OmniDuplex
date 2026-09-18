You create one dataset sample for the task "Realtime Target Grounding".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). All timestamps must use MM:SS and remain within the video. The original video is the primary source of truth. The supplied event timeline and optional items are fallible search aids, never proof.

{{QUESTION_TIME_POLICY}}

## Target selection and localization

Create a natural English standing instruction asking to be told when one specific visual target or identifiable sound appears and where it is. Prefer a target requiring {{PREFERRED_MODALITY}}; use {{FALLBACK_MODALITY}} only when the preferred modality has no reliable candidate. Inspect the whole relevant video and cover every independent, unambiguous, directly verified occurrence at or after question_time. Do not report semantic relatives that are absent.

For each independently verified occurrence, create exactly one separate
response, in chronological order. When the target appears multiple times,
respond once for each completed occurrence and report that occurrence's own
grounding position; never merge multiple occurrences into one response or omit
verified occurrences merely because there are many. Describe only a coarse
direction using evidence available at that occurrence's completed boundary.
For visual targets use natural English such as "upper left" or "lower right",
and one frame_region from top-left, top-center, top-right, middle-left,
middle-center, middle-right, bottom-left, bottom-center, or bottom-right.
Describe frame position, not gaze direction. Never output bounding boxes, pixel
positions, coordinates, numeric angles, or distance estimates, including in prose.
For sound use a supported coarse direction such as "to the left"; frame_region
must be null. Do not infer source direction from the meaning of speech, a visual
association, or a mono recording without directional evidence. Express uncertainty
for occlusion or ambiguous direction; skip when localization is unsupported.

Set audio_dependency and vision_dependency to required, helpful, or none; at least one must be required. Every response must include trigger evidence and the required modality in trigger_type. Write all natural-language content in English.

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
  "question": "<target condition and explicit request for its frame region or sound direction>",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [{
    "trigger_time": "MM:SS",
    "response": "The target is ...",
    "target": "...",
    "location_type": "visual|sound",
    "location": "...",
    "frame_region": "top-left",
    "confidence": 0.0,
    "uncertainty": "... or none",
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If no target can be fully verified and localized, return only {"status":"skip","reason":"..."}.

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
may use only evidence observable at or before its own trigger_time or message
timestamp; never use later video evidence. User-visible response text must not
include an MM:SS video timestamp; keep video timestamps only in structured time
fields. Remain silent without a verified trigger; if the task cannot be grounded,
return skip.

## Explicit question scope

For visual grounding, the question must explicitly ask which part or region of
the video frame contains the target. An unqualified "where it is" is insufficient.
The response should briefly name the target and its coarse frame position.
For audio grounding, ask for the sound source's approximate direction instead.
