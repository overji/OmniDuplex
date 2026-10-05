Mode: proactive reminder. The viewer's question must specify a selective event,
target state, transition, or count threshold and ask for notification whenever
that criterion is satisfied. The criterion must distinguish what warrants a
reminder from related observations that do not. Broad instructions such as
"track the passenger's emotions" or "report whenever their emotion changes"
are ongoing description requests, not valid PR questions; rewrite them before
generating responses. Adding a trigger object does not make such a question PR.

For emotion recognition, "Let me know whenever the passenger in the video sounds
frightened" is a selective request. Other emotional changes do not warrant a
response unless they independently satisfy that same request. Happiness, relief,
or any other supported emotion can also be selected; do not restrict PR to fear,
negative emotions, or danger. Express uncertainty when inferring emotions.
For other abilities, apply the same distinction: track the evolving object count
in RTD; notify when a user-specified count threshold is crossed in PR. These are
examples, not restrictions on eligible scenarios or trigger kinds.

Derive trigger.definition directly from the question and use the same criterion
for every response. Never invent a selection rule only in annotation fields, or
widen the criterion to include unrelated events. Use condition for a sustained
state and event for a discrete event or specified transition; do not label each
observation of a persistent state as a new event to bypass reset requirements.
Remain silent while the criterion is not met. One sustained matching state is
one episode; a later matching state is a new episode only after a supported reset.

Before accepting the sample, identify a plausible related observation that would
NOT warrant a reminder under the question. This is a semantic self-check, not an
extra output field or a requirement to invent a nonmatching event in the video.
If the question merely asks for all changes within a descriptive focus, rewrite
it into a useful selective request. If no valid request with real matching evidence
is available, return skip. Verify that every response satisfies the question and
that no nonmatching observation is emitted. Do not impose a maximum number of
reminders or force PR to have fewer responses than RTD; their evidence may overlap.

The instruction requests every independent matching occurrence after question_time.
Scan through the end of the video and provide one short, precise response per
occurrence, all of them;
never repeatedly report a sustained condition. There is no forced initial silent
interval. Mark the earliest confirmable moment as window_start and select the most suitable
trigger_time no later than window_end, while the notification is still useful.
At least one real trigger after the instruction is required; otherwise return skip.
Count and compound conditions must actually become satisfied, not merely plausible.
Predictions and causal explanations remain explicitly uncertain in the response.
