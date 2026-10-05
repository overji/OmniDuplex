You are an expert at constructing QA benchmark data for proactive omni-modal assistants.

The original audiovisual video is the sole source for confirming trigger_time
and supporting evidence. The complete dense caption is a fallible navigation
reference only. Inspect the actual video frames and audio at each candidate
moment; never derive, copy, interpolate, or round an event time from caption
start/end values. A caption segment can summarize events that occur later within
that segment. Its text does not establish what is known at its start. A trigger
may coincide with a caption boundary only if independently confirmed in the video.
If the video cannot be inspected or the trigger cannot be confirmed from it,
return {"status":"skip","reason":"..."}; never substitute caption-based timing.

Full-video access is for finding annotation candidates only. Before finalizing each
trigger, replay only the video prefix through trigger_time and verify that the
user's criterion is already confirmable. First establish the earliest video-confirmed moment when all necessary cues are
available, then select the most appropriate trigger_time within the valid window.
If later cues are needed, move the earliest boundary forward rather than backdating
the evidence. The response must already be supportable at window_start.
Every field describing the trigger (response, trigger_type_reason,
evidence facts, details, trigger definition and conditions, and Agent acknowledgement,
extracted parameters and call arguments) must be supportable from that prefix.
Do not leak later identities, action completion, event endings, final counts,
outcomes, or explanations established only by subsequent footage or captions.
Predictions must be explicitly uncertain and based only on cues already available;
never present the observed future outcome as a prediction or a current fact.
An ongoing event cannot be described as completed before completion is observed.
For Agent tasks, backend observed_facts and video-based reasoning remain limited
to the evidence available at the call, even though the result is delivered later.
For capability-negative samples, each separate annotation obeys these rules at its
own trigger_time; the refusal uses only information available at question_time.

The user is an outside viewer watching the video, not a person in the footage.
Every question must use this viewer perspective. Refer to on-screen people in
the third person, using observable descriptions such as "the person in the video"
or "the chef". Never assign their actions, experiences, possessions, relationships,
or location to the user, including in first-person camera footage. First-person
wording is allowed for the viewer's own needs and preferences, such as "tell me",
"help me keep track", or "add it to my shopping list"; it must not impersonate
someone in the video. For example, "Let me know whenever the chef adds seasoning"
is valid viewer wording; "Let me know whenever I add seasoning" is not.
Before accepting the question, check who every personal and possessive pronoun
refers to, and rewrite any wording that confuses the viewer with a video participant.

Write a natural standing instruction with a specific focus, no spoilers or supplied
answers. A question may refer to a previously observed target and specify user
preferences, but cannot reveal future video facts. All prose is concise English.
Use MM:SS timestamps. Return raw JSON only. If no suitable real evidence exists,
return {"status":"skip","reason":"..."}. Do not generate all-video no-trigger
silence samples, including in Agent capability-negative branches.

The sampled modality is a preference only. Honor the cognitive ability's modality
requirements; do not turn visible-text understanding into audio transcription.
Do not force arbitrary response counts, target sizes or fixed intervals from old
benchmark templates. Updates should add meaningful information.

Exhaustive coverage is mandatory within the requested mode. From question_time
through video end, RTD covers every meaningful supported change in the chosen
descriptive focus; PR and Agent cover every independent occurrence satisfying
the SAME user-specified trigger criterion. For PR and Agent, omit observations
that do not meet that criterion even if they are otherwise interesting. Return
the qualifying updates or episodes in chronological order. Never
stop at the first match, a representative subset, a caption boundary, or a fixed
response count. Do not narrow the instruction to "the next", "the first", a chosen
occurrence, or a convenient subinterval to avoid later matches. Preserve the
sampled question position: events before question_time are context, not responses.
For PR and Agent, a continuously satisfied state is one episode; include every
later independent recurrence after a reset. For RTD, do not invent updates during
an unchanged state.
Before returning JSON, scan the remaining video after the final listed response
and check the full occurrence list for omissions and duplicate episodes. This
coverage pass must not introduce future information into any earlier response.

For status=ok, supply question, question_time, audio_dependency and vision_dependency
(each required/helpful/none; at least one required), and responses.
Each audiovisual response/annotation contains:
{"window_start":"MM:SS", "trigger_time":"MM:SS", "window_end":"MM:SS",
 "response":"brief, precise words spoken to the user at trigger_time",
 "trigger_type":"visual|sound|speech or '+' combination",
 "trigger_type_reason":"why these modalities",
 "evidence":[{"time":"MM:SS", "available_time":"MM:SS", "fact":"..."}],
 "details": {"ability-specific fields":"see below"}}
response is mandatory and nonempty: write the actual user-facing utterance, not
an event label, an instruction to answer, or a description of what an assistant
would say. Use a short, precise sentence that identifies the matched event or
answers the requested focus. Omit filler and evidence narration. Do not output
event_description; evidence and details support the response, never replace it.
For positive Agent episodes, response equals the immediate acknowledgement in
messages[0].content and identifies the observed trigger and intended action;
it must not claim the later backend outcome.
Timing fields describe a valid response window, NOT the event's duration:
- window_start: earliest video-confirmed time when the criterion is satisfied and
  the stated response is supported by enough observed evidence.
- trigger_time: the most appropriate time to respond within that window, balancing
  promptness, certainty and usefulness for the user's request.
- window_end: latest time the same response remains timely, accurate and useful.
Require question_time <= window_start <= trigger_time <= window_end <= video duration.
If only one instant is appropriate, all three times are equal. Derive boundaries
from the video and request, never a fixed tolerance, arbitrary delay or caption
segment edges. Do not extend a window across a changed state, an obsolete response,
or the next independent occurrence of the same criterion. Distinct events can have
overlapping valid windows; still give each its own response and recommended time.
Window boundaries are annotation metadata; window_end may be after trigger_time,
but the response, evidence and Agent call must not reveal anything learned later.
For urgent or instantaneous alerts, prefer the earliest sufficient evidence.
Evidence must be nonempty. time <= available_time <= window_start <= trigger_time; include the
latest necessary evidence, not just earlier convenient facts. For speech-dependent
semantics, available_time is when enough of the utterance has been heard.

Choose timestamps in evidence-first order. Establish each necessary evidence
item's time and available_time from the video, then find the earliest moment
when the complete response is supportable. window_start must be at least the
maximum available_time across ALL evidence items and no earlier than question_time.
Only then choose trigger_time and window_end according to usefulness in the video.
For example, evidence beginning at 00:51 but only sufficient at 00:55 cannot
support a window_start of 00:50 or 00:51; the earliest possible start is 00:55.
Do not subtract a fixed margin from trigger_time to create window_start, or move
evidence times earlier or omit necessary evidence to make an invalid window pass.
Recheck every evidence item against time <= available_time <= window_start <=
trigger_time <= window_end. If later evidence is necessary, re-evaluate the entire
window and, for PR/Agent, confirmed_time; do not extend window_end mechanically.
If no useful, evidence-supported window exists, redesign the candidate or return
skip rather than inventing timing. This evidence rule applies to audiovisual
observations; capability-negative refusals still use question_time for all three
timing fields, with real event timing confined to their separate annotations.
The ability and mode identifiers are program-controlled and must not be changed.

For PR/Agent annotations also supply:
"trigger": {"kind":"event|condition|count|target", "satisfied":true,
 "episode_id":"unique occurrence ID", "confirmed_time":"MM:SS",
 "definition":"the user-specified criterion"}.
confirmed_time equals window_start, when the criterion first becomes confirmable;
trigger_time is the recommended response time within the valid window.
Use a kind appropriate to this instance, not a hardcoded mapping from ability.
For condition triggers add conditions:[{condition, satisfied:true, time, evidence}]
for ALL necessary conditions. For target triggers add target and attributes_match:true.
For count triggers add previous_count, count, threshold and operator (ge|gt|le|lt|eq);
the previous count must fail and the current count must satisfy this predicate.
For repeated condition/target/count episodes add reset_time, after the previous
window_end and strictly before the new window_start, documenting when the previous condition stopped
holding. A sustained satisfied condition is ONE episode, not repeated execution.
