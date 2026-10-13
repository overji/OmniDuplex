You are an expert video analyst producing annotations for a video
understanding dataset. Watch this video carefully — pay close
attention to the visual content, sounds/music, and speech/dialogue
simultaneously.

This video is {duration_mmss} long ({duration_sec:.0f} seconds).

Produce a dense temporal caption — divide the video into
fine-grained segments. These captions will be used downstream to
generate question-answer pairs about the video, so they must
contain enough detail to support questions about what happened,
when, why, what changed, and what can be inferred.

Segment guidelines:
- Each segment should be roughly 5-30 seconds long. No segment
  should exceed 45 seconds.
- For a {duration_sec:.0f}-second video, produce approximately
  {suggested_segments} segments.
- Start a new segment when there is: a scene/location change, a
  new activity or action, a topic shift in speech, a new person
  appearing or leaving, an audio event (music change, sound
  effect, silence), or any notable visual change.

Segment start/end values indicate coverage only. A segment may contain multiple
events at different times; do not imply that every event occurs at its start.
When you can confirm an event's time, describe its absolute video time in the
existing text fields. Never guess a time from segment boundaries or use a
segment-relative offset as an absolute timestamp. Downstream generators must
verify event times from the original video.

For each segment, provide:

1. caption: A detailed, information-dense paragraph integrating
   visual, audio, and speech into one coherent description.
   Include: Who (appearance, actions), What (objects, text),
   Action (specific verbs, direction), Change (differences from
   previous segment), Audio-visual correlation.

2. visual: Exhaustive visual details — scene, lighting, colors,
   objects, people, camera work, on-screen text verbatim.

3. audio: Precise sound description — music (genre, tempo,
   instruments), sound effects, ambient sounds, voice quality.
   Note onset and cessation of sounds.

4. speech: Detailed summary of what is said — key claims, names,
   numbers, facts. If no speech, write "None".

Return a JSON array:
[
  {
    "start": "MM:SS",
    "end": "MM:SS",
    "caption": "...",
    "visual": "...",
    "audio": "...",
    "speech": "..."
  }
]

Rules:
- Segments must cover the entire video from 00:00 to
  {duration_mmss} with no gaps or overlaps.
- Timestamps in MM:SS format.
- Be maximally specific — names, colors, counts, positions.
- Return raw JSON only, no markdown.

Also preserve target attributes, action onset/continuation/end, verbatim visible
text, current object counts and reappearing identities, independent action
occurrences, relative spatial relations, emotion cues and audiovisual alignment.
Distinguish observable cues from uncertain causal explanations or predictions.
Do not invent details to populate these dimensions. Downstream tasks will verify
precise moments in the original video rather than using segment boundaries.
