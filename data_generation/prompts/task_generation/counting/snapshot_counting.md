You create one dataset sample for the task "Triggered Snapshot Counting".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; event descriptions and optional items only identify candidates.

{{QUESTION_TIME_POLICY}}

Write a natural English instruction that names one discrete audio or visual trigger and one visibly countable target, such as "Whenever the whistle sounds, tell me how many players are visible." Prefer a trigger requiring {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}} only when necessary.

Choose a concrete object category explicitly named in the question, with clearly
distinguishable instances in the trigger frame. Do not count generic "objects",
unclear tiny shapes, guessed occupants, or heavily occluded targets. If a complete
count is uncertain, choose a different category or skip. target_locations is an
array of short coarse English directions, such as ["upper left", "lower right"],
or [] when unnecessary. Never output boxes, coordinates, pixel positions, or
numeric spatial measurements in any field.

Audit the complete video after question_time. For every independently verified trigger onset, count targets in the scene at that exact moment. Count each visible target once, handle brief occlusion conservatively, and never replace the current snapshot count with a cumulative count. Do not report anything when the trigger does not occur. Choose another target or return skip if any complete count is ambiguous.

Vision must be required because the reported quantity comes from the current image. Audio may also be required when the trigger is audible. Include the evidence supporting both the trigger and count. All natural-language content must be English.

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
  "question": "Whenever ..., tell me how many ... are visible.",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required",
  "trigger_times": ["MM:SS"],
  "responses": [{
    "trigger_time": "MM:SS",
    "response": "There are N ... visible now.",
    "count": 0,
    "counting_basis": "Observable targets counted at the trigger frame.",
    "target_locations": [],
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If no reliable trigger/count pair exists, return only {"status":"skip","reason":"..."}.

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

## Clean-scene selection and complete trigger audit

Only select a scene with a clean background, separated targets, clear boundaries,
and an explicitly defined counting area. Reject crowded scenes, heavy occlusion,
tiny targets and motion blur. The count must be reliable at EVERY matching trigger.
First fix a natural question_time, independently scan all later video for the
complete trigger list, then verify the snapshot count at every listed onset.
Return that sorted, unique list in top-level trigger_times. It must match the
response trigger_time sequence one-to-one. Do not omit a difficult occurrence or
move question_time to exclude it. If any occurrence is uncountable, choose a new
target/trigger definition at the same question_time, or skip the sample.
