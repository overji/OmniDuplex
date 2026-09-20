You create one dataset sample for the task "Instant Event Alert".

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

The video duration is {{VIDEO_DURATION}}. Every timestamp in the output must use
MM:SS and fall between 00:00 and {{VIDEO_DURATION}}, inclusive.

The user gives a standing instruction while a video is in progress. The
assistant must account for matching events that happened before the instruction
and continue watching and listening for later occurrences.

## Generation controls

Prefer a target event whose {{PREFERRED_DEPENDENCY}} is `required`. If no
suitable {{PREFERRED_MODALITY}}-dependent target exists, fall back to a target
whose {{FALLBACK_DEPENDENCY}} is `required`. Return `skip` only if neither the
preferred nor fallback modality has a suitable target.

{{QUESTION_TIME_POLICY}}

## Evidence and target selection

You are given the original video together with timestamped `audio_events` and
`visual_events` derived from it. The original video is the primary source of
truth. Treat the event descriptions and timestamps only as fallible references
for finding candidate targets and likely time ranges; they may contain false
positives, duplicate descriptions, inaccurate boundaries, or missing
occurrences. Never treat an event description as proof that an occurrence
happened.

Use the event descriptions to identify a candidate target, then inspect the
video itself to confirm the target's observable definition and exact onset.
Choose exactly one target event that is clearly verifiable in the video, has a
meaningful onset, is natural to request as an alert, and is specific enough to
identify every occurrence consistently. Useful visual events include a person
or object appearing, an action beginning, a visible state changing, or relevant
on-screen text appearing or changing.

Avoid continuous background music, ambient noise, silence, routine camera cuts,
persistent watermarks, and broad activities spanning most of the video. A
distinct transition within such a segment may be used when it is a natural
alert target.

## Full-video occurrence audit

After choosing a candidate target, independently scan the entire video from
beginning to end and build the final occurrence list from the video evidence:

- Exclude a referenced occurrence when the claimed target is not actually
  visible or audible in the video.
- Include every clearly matching occurrence found in the video even when the
  supplied event descriptions omit it.
- Count multiple descriptions of one uninterrupted occurrence only once.
- Count a later occurrence separately when the target clearly ended and began
  again, or when a new independently identifiable onset occurs.
- Determine every onset timestamp from the video. Do not copy an event timestamp
  without checking it against the video.

Every occurrence in the final list must be directly verifiable in the video.
Before writing the output, audit the list once for both omissions and duplicate
counts. If the complete count or the onset boundaries remain ambiguous, choose
a different target that can be counted reliably. Return `skip` if no suitable
target can be fully verified. Do not invent an event, occurrence, detail, sound,
speech content, or timestamp.

## Question and temporal responses

Write one natural standing instruction at `question_time`. It must sound like a
real person speaking to a smart assistant. It must not contain a timestamp,
reveal when an occurrence happens, or state the outcome in advance.
Write the question, responses, reasons, and descriptions in English. The
user-facing `response` must not contain any video timestamp such as `MM:SS`;
put times only in structured fields.

The requester is the video viewer, the people shown or heard are video
participants, and the assistant answers the viewer. First-person wording in the
question may express the viewer's needs, preferences, or wish to receive results;
it must not attribute a participant's actions, speech, location, possessions, or
experiences to the viewer. Describe those participants in the third person or
by an observable role, even for POV footage. Keep quoted speech attributed to
its original speaker. In assistant replies, "you/your" refers to the viewer,
not a video participant.

Divide every video-verified occurrence of the chosen target by its onset
relative to `question_time`:

- Historical occurrences have onset earlier than `question_time`. If one or
  more exist, create exactly one summary response at `question_time`; this
  historical summary is exempt from the sampled silent interval. Put every
  historical onset, sorted, in its `event_times` array. Briefly summarize all
  of them in one natural reply.
- Current or future occurrences have onset at or after `question_time`. Create
  one separate response for each occurrence. Put its onset in an
  `event_times` array containing only that onset, and set `trigger_time` to the
  completed-boundary response time, which must not precede that onset.
- Except for the historical summary, if no historical summary exists, do not
  create the first response until the required silent interval after
  `question_time` has elapsed. Never mix historical and future times in one
  response, omit a video-verified occurrence, or invent a repetition.
- Order responses by `trigger_time`. Each response must be brief and
  conversational. After the historical summary, continue producing
  responses for all later occurrences.

The target's total occurrence count is the total number of timestamps across
all `event_times` arrays, not the number of response objects. This total must
exactly equal the number of independently verified occurrences in the video;
historical occurrences remain grouped into one response at `question_time`
without being collapsed into one occurrence.

An event counts as historical only when its observable boundary was completed
and confirmed before `question_time`; an event still ongoing at that point is
not eligible for the historical summary until completion is confirmed.

## Dependency and trigger labels

Set both `audio_dependency` and `vision_dependency` to `required`, `helpful`, or
`none`:

- `required`: the modality is necessary to detect the target or provide the
  response's essential information.
- `helpful`: the target can be determined without it, but it provides useful
  confirmation or disambiguation.
- `none`: it contributes no meaningful evidence.

Exactly one dependency must be `required`; the other must be `helpful` or
`none`. The preferred {{PREFERRED_MODALITY}} modality should be `required`
unless selection had to fall back to {{FALLBACK_MODALITY}}.

For every response, set `trigger_type` to `visual`, `sound`, `speech`, or a
plus-joined combination such as `visual+sound` or `visual+speech`. Include only
modalities that support that occurrence. A vision-required target must include
`visual` in every trigger type; an audio-required target must include `sound` or
`speech`. For each response, provide a brief evidence-based
`trigger_type_reason` and a concise factual `event_description`, both derived
from what is actually observable in the video. Do not copy unsupported details
from the supplied event descriptions.

## Output

Return exactly one JSON object and no Markdown:

{
  "status": "ok",
  "question": "...",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [
    {
      "trigger_time": "MM:SS",
      "event_times": ["MM:SS"],
      "response": "...",
      "trigger_type": "...",
      "trigger_type_reason": "...",
      "event_description": "..."
    }
  ]
}

If no suitable discrete event is supported, return only:

{"status":"skip","reason":"..."}

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
