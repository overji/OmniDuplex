You create one dataset sample for the task "Instant Event Alert".

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
Write the question, responses, reasons, and descriptions in English.

Divide every video-verified occurrence of the chosen target by its onset
relative to `question_time`:

- Historical occurrences have onset earlier than `question_time`. If one or
  more exist, create exactly one immediate summary response. Set its
  `trigger_time` to `question_time` and put every historical onset, sorted, in
  its `event_times` array. Briefly summarize all of them in one natural reply.
- Current or future occurrences have onset at or after `question_time`. Create
  one separate response for each occurrence. Set `trigger_time` to that onset
  and set `event_times` to an array containing only that onset.
- Never create a response before `question_time`, mix historical and future
  times in one response, omit a video-verified occurrence, or invent a
  repetition.
- Order responses by `trigger_time`. Each response must be brief and
  conversational. After an immediate historical summary, continue producing
  responses for all later occurrences.

The target's total occurrence count is the total number of timestamps across
all `event_times` arrays, not the number of response objects. This total must
exactly equal the number of independently verified occurrences in the video;
historical occurrences remain grouped into one immediate response without being
collapsed into one occurrence.

An event whose onset is before `question_time` counts as historical even if it
is still ongoing when the question is asked.

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
