You create one dataset sample for the task "Realtime Cumulative Event Counting".

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

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; supplied events and optional items are fallible references.

{{QUESTION_TIME_POLICY}}

Write one natural English standing instruction asking to count every independent occurrence of a specific observable event. Prefer a target requiring {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}} only when needed. Define a meaningful onset and end so a sustained event counts once, while a later event after a clear end counts again. Audit the whole video and do not omit unannotated occurrences or count duplicate annotations.

Occurrences before question_time are historical: group all of them into one response at question_time, without the sampled silent interval, with their sorted times and cumulative count. If no historical occurrence exists, the first response must observe the required silent interval. When a historical summary exists, later occurrences are not delayed by that initial interval; each gets its own response after its completed boundary when permitted by the response timing rule and increases count by one. Never use future evidence to alter a count silently; choose an unambiguous target or skip. event_times across responses must be unique and chronological.

At least one modality dependency must be required and represented in every trigger_type. All natural-language content must be English.

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
  "question": "Keep count whenever ...",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [{
    "trigger_time": "MM:SS",
    "event_times": ["MM:SS"],
    "response": "The running count is N.",
    "count": 1,
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If no reliably countable repeated event exists, return only {"status":"skip","reason":"..."}.

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
