Mode: real-time description. Provide at least two meaningful updates at natural
changes or evidence boundaries. Include the current state when useful, then new
information, not paraphrases of the same state. No forced initial silent interval.

The viewer's question must request ongoing description or tracking of a chosen
subject and dimension as it evolves, not just notification when one selected
condition becomes true. Cover all meaningful supported changes within that focus,
including transitions away from a previously reported state. Do not force periodic
reports or invent changes merely to produce multiple responses.
For emotion recognition, a suitable question is "Keep me updated on how the
passenger in the video seems to be feeling." Report the supported evolution of
emotion, rather than reporting only fear or only happiness. Emotions remain
inferences from observable cues, not certain facts.
Before accepting the sample, check that each update adds current information and
that the question asks for an evolving description. A request such as "Tell me
whenever the passenger sounds frightened" belongs to PR and must be rewritten
for RTD. The distinction is the requested behavior, not a required difference in
response count, wording, or timestamps from another sample.

Each response uses the common valid response window: window_start is the earliest
supported update time, trigger_time is the recommended update time, and window_end
is the latest useful update time. These fields do not describe an observation span.
Recommended update times must increase. Valid windows may overlap for distinct
updates, but must not keep an outdated statement eligible after its state changes.
Historical facts may contextualize the current state, but no future evidence is
allowed. Omit trigger objects: this mode continuously describes the chosen focus.
For action counting, fix counting_start across responses; it may precede the
question if the question explicitly asks for the running total from that origin.
