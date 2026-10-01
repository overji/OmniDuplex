You create one dataset sample for the task "Event-Driven Realtime Narration".

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

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; supplied metadata only helps locate candidate changes.

{{QUESTION_TIME_POLICY}}

Write one natural English instruction asking for realtime descriptions of a coherent event class, subject, or scene after each update is complete. Prefer a focus whose important updates require {{PREFERRED_MODALITY}}, using {{FALLBACK_MODALITY}} only if necessary. Select a focus with exactly 3-6 important, directly verifiable updates that complete at or after question_time.

Describe each update only after that specific event or change has fully occurred and its outcome can be confirmed. Set trigger_time to the earliest moment when the completed update is observable and confirmable, never to its onset. For a sustained event, wait until it ends; for a state change, wait until the new state is established. Do not report an update whose completion cannot be confirmed before the video ends. Use only evidence observable at or before trigger_time. Cover significant actions, state changes, entrances, abnormal events, or relevant relationships; omit camera cuts, persistent backgrounds, repetitions, and trivial motion. Keep context consistent. observed_facts must contain only observable facts. interpretation may offer a cautious explanation or be null and must never be presented as fact. Do not predict later events or interrupt for unimportant changes.

At least one dependency must be required. Each response must be at most 40 English words and include the required modalities in trigger_type. All natural-language content must be English.

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
  "question": "Describe each important ... once it has happened.",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [{
    "trigger_time": "MM:SS",
    "response": "...",
    "observed_facts": ["..."],
    "interpretation": null,
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If 3-6 meaningful updates cannot be verified, return only {"status":"skip","reason":"..."}.

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
