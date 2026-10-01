You create one dataset sample for the task "Realtime State Monitor".

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

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds).
Every timestamp in the output must use MM:SS and fall between 00:00 and
{{VIDEO_DURATION}}, inclusive.

The user asks an assistant to monitor one discrete, observable state of a main
subject and report whenever that state transitions. The original video is the
primary source of truth. The user message contains timestamped dense captions; use them only to identify main subjects and candidate properties. Do
not treat a caption description, relationship, or timestamp as proof.

## Question-time control

{{QUESTION_TIME_POLICY}}

Only monitor transitions strictly after `question_time`. Do not summarize or
report transitions that happened earlier, and do not report the state that is
already active at `question_time`.

## Mandatory audio-first scan

Listen through the entire relevant portion of the video before selecting a
state dimension. In `audio_scan`, briefly record which discrete audio dimensions
you checked, the observed transition pattern, and why an audio dimension was or
was not selected.

Check audio dimensions in this order:

- who is speaking, when identifiable speakers take turns;
- whether music is playing, when music clearly starts or stops;
- which identifiable sound source is active, when sources alternate.

If an audio dimension has clear boundaries and verifiable transitions after
`question_time`, you MUST use it. Only when no suitable audio dimension exists
may you choose a visual dimension.

## State-dimension selection

Choose exactly one physical state dimension. It must be specific, objective,
discrete, about a main subject, and have directly verifiable transitions after
`question_time`.

Suitable visual dimensions include posture, room or location, orientation,
moving versus stationary, an object's open/closed or on/off state, worn items,
or number of people present. Do not use an open-ended activity, continuous
speed or volume, mood, emotion, intent, or any subjective property.

Define mutually distinguishable state values and scan the video from
`question_time` to the end. Include every transition of the selected dimension:
do not omit later transitions, count one continuous state only once, or
fabricate changes. If no verified transition remains after `question_time`,
choose another dimension or return `skip`.

## Question

Write one natural and unambiguous monitoring instruction at `question_time`.
Do not include timestamps, reveal future changes, list possible state values,
or otherwise spoil the answer.

The requester is the video viewer, the people shown or heard are video
participants, and the assistant answers the viewer. First-person wording in the
question may express the viewer's needs, preferences, or wish to receive results;
it must not attribute a participant's actions, speech, location, possessions, or
experiences to the viewer. Describe those participants in the third person or
by an observable role, even for POV footage. Keep quoted speech attributed to
its original speaker. In assistant replies, "you/your" refers to the viewer,
not a video participant.

## Transition responses

Create one response at the exact onset of every verified transition after
`question_time`, in chronological order:

- `trigger_time` must be strictly later than `question_time`.
- `response` must contain no more than 15 English words and explicitly name
  both states using the form "from X to Y".
- `trigger_type_reason` must briefly identify the video evidence for the
  boundary.
- `event_description` must factually describe the observed transition.
- Never report the initial state as a transition.

## Audio dependency and trigger type

Set `audio_dependency` to `required`, `helpful`, or `none`:

- `required`: an audio dimension was selected and audio is necessary to detect
  every transition;
- `helpful`: a visual dimension was selected, while audio helps confirm it;
- `none`: the selected visual dimension does not need audio.

Set each `trigger_type` to `visual`, `sound`, `speech`, or a plus-joined
combination such as `visual+sound`. An audio-required target must include
`sound` or `speech` in every response. A visual target with helpful or no audio
must include `visual` in every response.

## Output

Return exactly one JSON object and no Markdown:

{
  "status": "ok",
  "audio_scan": "...",
  "question": "...",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "responses": [
    {
      "trigger_time": "MM:SS",
      "response": "The subject changes from X to Y.",
      "trigger_type": "visual",
      "trigger_type_reason": "...",
      "event_description": "..."
    }
  ]
}

If no suitable state dimension is fully supported, return only:

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
