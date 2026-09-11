You create one dataset sample for the task "Realtime State Monitor".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds).
Every timestamp in the output must use MM:SS and fall between 00:00 and
{{VIDEO_DURATION}}, inclusive.

The user asks an assistant to monitor one discrete, observable state of a main
subject and report whenever that state transitions. The original video is the
primary source of truth. The user message contains `items` extracted from dense
captions; use them only to identify main subjects and candidate properties. Do
not treat an item description, relationship, or timestamp as proof.

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

If an audio dimension has clear boundaries and exactly 2-5 transitions after
`question_time`, you MUST use it. Only when no suitable audio dimension exists
may you choose a visual dimension.

## State-dimension selection

Choose exactly one physical state dimension. It must be specific, objective,
discrete, about a main subject, and have exactly 2-5 directly verifiable
transitions after `question_time`.

Suitable visual dimensions include posture, room or location, orientation,
moving versus stationary, an object's open/closed or on/off state, worn items,
or number of people present. Do not use an open-ended activity, continuous
speed or volume, mood, emotion, intent, or any subjective property.

Define mutually distinguishable state values and scan the video from
`question_time` to the end. Include every transition of the selected dimension:
do not omit later transitions to satisfy the 2-5 limit, count one continuous
state only once, or fabricate changes. If the complete verified sequence does
not contain exactly 2-5 transitions, choose another dimension or return `skip`.

## Question

Write one natural and unambiguous monitoring instruction at `question_time`.
Do not include timestamps, reveal future changes, list possible state values,
or otherwise spoil the answer.

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
