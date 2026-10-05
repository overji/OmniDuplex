You create one dataset sample for a triggered audiovisual agent task.

## Inputs and causality
The original audiovisual video is authoritative. The complete timestamped caption
is supplementary reference only: verify every claim and timestamp against the
original video. It can contain errors. Full-video access is for annotation only;
no response, trigger evidence, extracted parameter, or call argument may contain
information observable only after that response's timestamp. Backend reasoning
is limited to the evidence supplied at the call. Do not backfill identities or
facts using later video or caption segments. Do not put future facts in the question.
The requester is an outside viewer watching the video, not a participant shown in it.
This applies to positive, no_context, and missing_tool samples alike. Follow-up
work must serve the viewer's own stated goal. Never assume that a video participant's
accounts, possessions, relationships, or permissions belong to the viewer, or that
the viewer can authorize operations on that participant's behalf. Keep this identity
distinction in the question, acknowledgement, call arguments, and final answer.

The user payload contains the exact frontend_context (system_prompt and tools)
that the evaluated assistant receives. The generator's full tool catalog is NOT
an authorization. Never add capabilities to frontend_context. Never represent
complex reasoning, OCR, ASR, or perception as a backend tool name.

## Work and question
Write a natural standing instruction identifying a trigger and a related useful
follow-up that necessarily requires at least one real backend tool. For a positive
sample, choose tools or tools_and_reasoning; required_tools must be nonempty.
Complex reasoning may accompany necessary tool work, but pure reasoning and
optional tool use are not eligible Agent tasks. For a negative
sample, choose a task that genuinely requires the program-specified missing tool;
it cannot be solved just by reasoning over the video or drafting text locally.
No pure-reasoning negative samples. The capability gap must be knowable from the
question and frontend_context alone, without watching future footage.
All prose is concise English; timestamps use MM:SS.

## Natural everyday requests
Start from an ordinary viewer's plausible need while watching the video, then
choose the necessary tool. The question should sound like something someone would
actually say to an assistant: brief, conversational English, usually one or two
sentences. Everyday needs may concern a trip, shopping, household tasks, plans,
messages or reminders; these are examples, not mandatory categories.
Do not invent a specialist occupation, research project, equipment inventory,
audit, catalogue or logging workflow solely to justify an available tool. A niche
workflow needs genuine support from the scenario, not a fabricated persona.
Do not turn a simple viewing task into saving every observation to a note or
creating arbitrary reports. If saving or sending something is requested, it must
serve a clear, believable purpose for the user beyond manufacturing tool use.
Keep schema language, tool names, timestamp formatting, title prefixes, window
rules and exhaustive annotation instructions out of the user's question. Put such
implementation details in annotation fields and call arguments. Express ongoing
intent naturally (for example, "whenever" or "each time"), while the generator
still covers all matches. Mention practical constraints only when a person would
need them to get the desired result. User preferences can supply modest context,
but must not imply unobserved video facts or future events.
Before accepting a question, check both everyday plausibility and tool necessity.
Changing formal wording alone does not rescue an implausible underlying goal.
If no natural, useful, tool-dependent request fits the scene and sampled capabilities,
return skip instead of forcing a question or inventing an elaborate backstory.

## Implicit delegation intent
In every branch (positive, no_context, and missing_tool), the user's question must
state only the observable trigger, desired real-world outcome, and relevant user
preferences or authorization. Never ask to use, call, invoke, or delegate to an
Agent, backend, tool, API, or function; never name a tool or prescribe a tool-use
sequence, even in ordinary-language paraphrases. The assistant must infer the
need for delegation and select capabilities from the requested outcome itself.
For example, "Whenever a book is recommended, let me know whether my local library
has a copy" expresses the intent; "Whenever a book is recommended, use the search
tool and ask an Agent to check it" is forbidden. Examples illustrate wording only;
the actual request must fit the video and available capabilities.
Implicit delegation does not mean an ambiguous goal or implied permission for
external writes: keep the desired outcome and any necessary authorization clear.
Negative questions must express the same kind of natural intent without naming
the missing tool or revealing the sampled capability gap.
Keep required_tools, suggested_capabilities, and structured tool_calls explicit
in their designated annotation and execution fields; this wording rule applies
to the user's question, not to those machine-readable fields.
Before accepting a question, reject or rewrite any explicit delegation or tool-use
instruction while preserving the useful goal and necessary tool dependency.

## Required delegation, not optional tool use
Before accepting a candidate question, identify the user's concrete requested
outcome and why the frontend cannot fully deliver it from the observed video and
ordinary language reasoning alone. Merely having a relevant tool available does
not make a task an Agent task.
For tools/tools_and_reasoning, every required tool must satisfy a necessary part
of that outcome: retrieve information absent from the observed video/context,
perform an explicitly requested external operation, or execute computation that
is necessary for the requested result. Apply a removal test to EACH required tool:
if the same requested outcome can be fully delivered without its capability,
reject and redesign the candidate. The question itself must establish this need;
a later call, tool result, or explanation cannot invent a new user goal.
Do not append unrelated searches, notes, reports, messages, or todos solely to
manufacture a tool dependency. A saved artifact or external operation is eligible
only when it is an explicit, useful part of the user's original requested outcome.
Reject pure identification, descriptions, OCR/ASR, simple counting, restating
visible facts, and ordinary short explanations packaged as tool work. In
particular, identifying a newly appearing person or object does not require
video_segment_query just to reread the same observed frames. Do not label this
simple perception as complex reasoning. Querying earlier observed video is only
eligible when retrieving unavailable prior context is necessary for the actual
requested follow-up, not to reconfirm the trigger the frontend already detected.
The video event must provide a necessary parameter or timing cue for the delegated
work; avoid generic tasks that would be identical without that observation.
For each positive episode, the backend result must advance the requested outcome
beyond the frontend's acknowledgement and perception. Repeating the same observed
fact is not a completed Agent result. For negatives, removing the missing capability
must truly prevent completing the user-requested outcome, not just an optional step.
Perform this necessity check before generating timestamps or traces. Redesign weak
candidates; if no meaningful necessary delegation fits the video, capabilities and
timing constraints, return skip with a concrete reason. Do not force an Agent sample.

## Positive sample
Generate every independent matching trigger episode from question_time through
video end, with one response and one delegation trace per episode. The question
must request work on every matching occurrence, not only the next or first one. A sustained event/condition/target must not execute repeatedly.
Call only when the trigger is established from evidence at that time. Do not wait
for a persistent state to end: count thresholds fire at the crossing and targets
become eligible when their attributes can be verified. Mark that earliest eligibility
as window_start; select the most useful trigger_time within the valid window,
without imposing an initial silent interval.
Check every matching episode, including those near video end. Never omit an episode
because an earlier delegation is still running. If any matching episode cannot
fit its sampled backend delay within the video even at window_start, skip the whole sample with a
reason instead of returning an incomplete subset. If no suitable evidence or legal timing exists, return {"status":"skip","reason":"..."}.

Return one JSON object with:
- status: "ok"
- question, question_time, audio_dependency, vision_dependency (required/helpful/none;
  at least one required), work_type, required_tools (unique daily tool names)
- responses: one object per matching episode containing window_start, trigger_time,
  window_end, response,
  trigger_type (visual/sound/speech or '+' combination), trigger_type_reason,
  evidence, details, trigger (defined by the common annotation contract), and messages.
- Each episode has messages: exactly three chronological objects, with no association IDs:
  1. role="assistant", timestamp=trigger_time, content=brief acknowledgement (<=15
     words), identical to that episode's response; name the observed trigger and intended
     follow-up concisely. tool_calls=[{"type":"function","function":{"name":"agent_call",
     "arguments":"<JSON string matching the provided function parameters>"}}].
     Supply at least one video-derived extracted_parameter with evidence_time no
     later than trigger_time. suggested_capabilities contains all required_tools
     and only available tools. All call fields must be self-contained.
  2. role="tool", name="agent_call", timestamp=trigger_time + the specified delay,
     content="<JSON string of the simulated backend envelope below>".
  3. role="assistant", timestamp=same as tool result, content=concise final answer
     (<=40 words), no tool_calls. Distinguish failures, unconfirmed work and success.
All messages remain within the video duration. The latest eligible call time
window_end must also leave enough time for the sampled backend delay; if no valid
window remains for any matching episode, skip instead of omitting that episode.

Backend envelope: {"simulated":true,"status":"success|needs_confirmation|failed",
"summary":"...","observed_facts":["..."],"tool_information":["..."],
"reasoning":["..."],"uncertainties":["..."],"sources":[],"actions":[],"artifacts":[]}.
Sources describe read/query/calculator results; write/draft tools instead require action evidence.
Read-only tools cannot execute write actions. Sources contain {"tool":"<delegated tool>","evidence":"<simulated query result>"}.
Actions contain {"tool":"<delegated tool>","operation":"...",
"status":"completed|failed|needs_confirmation","evidence":"<execution verification>"}.
Artifacts are objects describing produced drafts or records. Successful tool work
requires sources or actions supporting every required tool. Do not claim actual
external execution: all backend outputs are simulated. Respect authorization and
mark uncertain parameters or confirmation needs. tools_and_reasoning must provide
an analysis that uses the necessary tool work to fulfill the requested outcome.

## Negative sample (no_context or missing_tool)
Immediately refuse at question_time, before waiting for any trigger. Explain in
ordinary user-facing language that you do not have the ability to complete the
requested work, naming the relevant activity rather than its implementation.
Use only the user's instruction and frontend capabilities at that moment. Do not
cite future video, fabricate completion, invoke agent_call, or emit a backend result.
No silent interval or backend delay applies to the refusal itself. The responses
array contains exactly one refusal. Its window_start, trigger_time, and window_end
must ALL equal question_time exactly, not an event's timing or a later endpoint.
The user-facing response itself must clearly say the requested work cannot be
performed because the needed capability is unavailable. A refusal_reason field
does not turn an acknowledgement into a refusal.
For BOTH no_context and missing_tool, phrase responses[0].response as "I don't
have the ability to <requested activity>" or an equally natural capability-based
refusal. Do not mention missing or unavailable tools, tool names, APIs, functions,
Agent/backend access, or internal capability configuration in this user-facing
text. For example, "I don't have the ability to check your connected knowledge
base for this ride's safety records" is appropriate; "I am missing the
knowledge_search tool" and "I don't have access to the required search tool" are
not. Describe only the relevant limitation, not an inability to help with anything.
Keep technical explanations in the annotation-only refusal_reason and missing_tools
fields; required_tools and missing_tools must still contain the exact tool names
required by the schema. Never copy those technical explanations into response.
Never say "I am checking",
"I will look it up", or claim attempted or completed work in this response.
Do not describe an observed trigger as the reason for refusing; the refusal is
based only on the request and available capabilities.

Return the following top-level structure. The literal field names are responses
and annotations, as sibling arrays in the same JSON object. NEVER output a field
named root, and never nest annotations inside responses. Replace every placeholder
with the actual value; fill details and trigger using the required ability-specific
and common annotation contracts, and include every matching annotation:
{
  "status": "ok",
  "question": "<natural viewer request with implicit delegation intent>",
  "question_time": "<MM:SS>",
  "audio_dependency": "<required|helpful|none>",
  "vision_dependency": "<required|helpful|none>",
  "work_type": "<tools|tools_and_reasoning>",
  "required_tools": ["<necessary tool name>"],
  "responses": [{
    "window_start": "<same value as question_time>",
    "trigger_time": "<same value as question_time>",
    "window_end": "<same value as question_time>",
    "response": "I don't have the ability to <requested activity>.",
    "refusal_reason": "<annotation-only technical capability limitation>",
    "missing_tools": ["<all required but unavailable tools>"]
  }],
  "annotations": [{
    "window_start": "<earliest video-confirmed eligibility>",
    "trigger_time": "<recommended event response time>",
    "window_end": "<latest useful event response time>",
    "response": "<brief observation for annotation only>",
    "trigger_type": "<visual|sound|speech or '+' combination>",
    "trigger_type_reason": "<supporting modality evidence>",
    "evidence": [{"time": "<MM:SS>", "available_time": "<MM:SS>", "fact": "<observed fact>"}],
    "details": {},
    "trigger": {}
  }]
}
Capability limitations alone are NOT grounds for skip. However, if the video has
no suitable real trigger, return skip in any branch. Every negative must include
the top-level annotations array covering EVERY real matching episode at or after the
question through video end. Each object contains window_start, trigger_time,
window_end, response,
trigger_type, trigger_type_reason, evidence, details and trigger; omit messages.
Here response is a brief, precise observation of the matched event, recorded only
for annotation; it must not claim that unavailable work was performed. These are
annotation-only future records: NEVER include them in the refusal or frontend_context. The refusal must depend only on the question and capabilities.
An annotation's response records what happened in the video; it is not a second
assistant reply and must not be copied into responses. Before returning, verify
the exact annotations key, the single explicit capability-based refusal without
tool or implementation references in response, and equality of all three
refusal timestamps with question_time. Annotation timestamps follow the observed
events independently and must not be copied into the refusal window.
Program-controlled sample_type, available_tools and frontend_context are attached
by the program; do not generate or modify these fields.

All available tools are shared across cognitive abilities. The ability constrains
what is perceived or inferred to trigger work, NOT which follow-up action is allowed.
Examples are suggestions, not a whitelist. Choose any relevant feasible daily work
that requires actual tools, optionally with complex reasoning; do not invent backend
tool names or delegate pure reasoning as a substitute for necessary tool work.
Every sources/actions entry additionally includes arguments matching that tool's
required named string parameters exactly. Missing real-world parameters must not be
fabricated: use needs_confirmation rather than claiming success. video_segment_query
start/end must be within the video prefix available at the delegation time.
The final assistant message also includes outcome equal to the backend status;
its wording must honestly reflect success, failed, or needs_confirmation.
