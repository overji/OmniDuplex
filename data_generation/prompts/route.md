You route one video to the most suitable ability categories for realtime interaction task generation.

The original video is the only final evidence. The supplied dense captions
are fallible navigation references: use them to locate candidates, then verify
relevance, clarity, timing, and completeness in the video. You may reject,
correct, or ignore any metadata claim. Do not treat metadata presence as proof.

## Candidate ability catalog

Evaluate every one of the 12 ability categories in this catalog exactly once:

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

In every mode, the possible question must come from an outside viewer watching
the video, not from a participant in it. Describe on-screen people in the third
person; do not assign their actions, experiences, possessions, relationships, or
location to the viewer, even for first-person camera footage. First-person wording
may express the viewer's own needs and preferences. Agent follow-up work must
serve that viewer's goal without assuming access to a participant's accounts or
authority to act on their behalf. Do not justify suitability with a question that
requires impersonating a person in the footage.

For Agent feasibility, the possible viewer question must express a natural desired
outcome whose fulfillment inherently needs backend capabilities. It must not name
tools, APIs, or functions, ask to use an Agent or backend, or prescribe a tool-use
sequence. Delegation and tool selection must be inferred from the user's intent.
Keep the goal and any authorization for external operations clear; implicit
delegation does not authorize unrequested actions. A request to call a tool or
Agent is not itself evidence that necessary delegated work exists.

Mark an ability category suitable if the video supports at least one valid task
in that category in any of RTD, PR, or Agent mode. It need not support all three
modes. Mark it unsuitable only when no mode can produce a valid task, for example
because reliable evidence is absent, targets cannot be distinguished, or the
ability does not fit the observed content. Rank suitable categories by interactive
value and explain the supporting video evidence. Do not separately select or rank
mode-specific tasks; downstream generators decide whether each mode can complete
a valid sample and may skip unsupported modes.

Return exactly one JSON object. Use only the catalog ability IDs as task values, without _rtd, _pr, or _agent suffixes;
do not copy a task name into a reason or rationale unless needed to identify the
evidence. Do not add question_time, response timelines, tool calls, or external
actions. All prose must be concise English.

{
  "status": "ok",
  "unsuitable_tasks": [{"task": "<ability_id>", "reason": "<specific reason>"}],
  "ranked_tasks": [{"task": "<ability_id>", "rank": 1, "rationale": "<specific rationale>"}]
}

Every catalog ability category must appear exactly once across the two arrays. Ranks must be
consecutive starting at 1. If no ability category is suitable, return {"status":"skip", "reason":"<reason>"}.

This catalog contains 12 ability categories, not 36 mode-specific tasks.
The following mode requirements only help determine whether at least one mode is
feasible for a category; failure of one mode does not exclude the category. RTD requires at least two meaningful time-aligned updates; PR requires
at least one real future trigger relative to a feasible instruction time. RTD
tracks meaningful evolution within a descriptive focus. PR requires a selective
criterion stated in the viewer's question and silence for nonmatching observations;
generic tracking of every change does not establish PR feasibility. Agent
requires a real trigger and a video-derived parameter but follow-up work is NOT
restricted by ability-specific examples. Text-rich understanding requires visible
text. Object counting concerns current objects; action counting concerns cumulative
independent occurrences. Prediction and causal explanation require supporting cues,
not omniscient hindsight. All-video absent-trigger samples are unsuitable.
