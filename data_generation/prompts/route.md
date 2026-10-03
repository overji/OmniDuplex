You route one video to the most suitable realtime interaction tasks.

The original video is the only final evidence. The supplied dense captions
are fallible navigation references: use them to locate candidates, then verify
relevance, clarity, timing, and completeness in the video. You may reject,
correct, or ignore any metadata claim. Do not treat metadata presence as proof.

## Candidate task catalog

Evaluate every task in this catalog exactly once:

{{TASK_CATALOG}}

The trigger described by a task request may be deliberately proposed by the
user; it does not need to be a spontaneous or conversationally natural request.
Treat a task as trigger-supported whenever the original video contains a real,
directly verifiable event, object, state change, utterance, or other signal that
could naturally serve as that trigger and is relevant to the task. Do not reject
it merely because the trigger wording is hypothetical, forceful, or externally
specified. Reject it when the corresponding event is absent, only guessed from
metadata, too ambiguous to verify, or unrelated to the task.

Judge the task as a combination of a possible viewer question and the video,
not from the video in isolation. A viewer question may introduce the need for a
lookup, comparison, recording, message, or other Agent work after a trigger;
the video does not need to contain an explicit request for that work or already
show an external system. For Agent tasks, it is sufficient that the video offers
a relevant trigger and at least one concrete parameter that could be extracted
when the viewer's question supplies the goal, destination, recipient, or other
preference. Do not mark an Agent task unsuitable solely because no external
action is currently requested in the footage.

Mark a task unsuitable when the video lacks reliable evidence, targets cannot be
distinguished, a complete timeline cannot be formed, or the task does not fit the
observed content. For suitable tasks, rank by interactive value while requiring
sufficient evidence, clear triggering, and realistic completion. Agent tasks are
suitable when the video supplies a trigger-supported event and at least one core
execution parameter that must be extracted from that event; a request fully
specified by the question alone is not enough.

Return exactly one JSON object. Use the actual catalog task names only as values;
do not copy a task name into a reason or rationale unless needed to identify the
evidence. Do not add question_time, response timelines, tool calls, or external
actions. All prose must be concise English.

{
  "status": "ok",
  "unsuitable_tasks": [{"task": "<task_name>", "reason": "<specific reason>"}],
  "ranked_tasks": [{"task": "<task_name>", "rank": 1, "rationale": "<specific rationale>"}]
}

Every catalog task must appear exactly once across the two arrays. Ranks must be
consecutive starting at 1. If no task is suitable, return {"status":"skip", "reason":"<reason>"}.

This catalog is the complete 12-ability × 3-mode matrix. Judge each combination
independently. RTD requires at least two meaningful time-aligned updates; PR requires
at least one real future trigger relative to a feasible instruction time. Agent
requires a real trigger and a video-derived parameter but follow-up work is NOT
restricted by ability-specific examples. Text-rich understanding requires visible
text. Object counting concerns current objects; action counting concerns cumulative
independent occurrences. Prediction and causal explanation require supporting cues,
not omniscient hindsight. All-video absent-trigger samples are unsuitable.
