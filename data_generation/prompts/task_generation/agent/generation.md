You create one dataset sample for a triggered audiovisual agent task.

## Inputs and causality
The original audiovisual video is authoritative. The complete timestamped caption
is supplementary reference only: verify every claim and timestamp against the
original video. It can contain errors. Full-video access is for annotation only;
no response, trigger evidence, extracted parameter, or call argument may contain
information observable only after that response's timestamp. Backend reasoning
is limited to the evidence supplied at the call. Do not backfill identities or
facts using later video or caption segments. Do not put future facts in the question.
The requester is the viewer, not automatically a participant shown in the video.

The user payload contains the exact frontend_context (system_prompt and tools)
that the evaluated assistant receives. The generator's full tool catalog is NOT
an authorization. Never add capabilities to frontend_context. Never represent
complex reasoning, OCR, ASR, or perception as a backend tool name.

## Work and question
Write a natural standing instruction identifying a trigger and a related useful
follow-up. For a positive sample, choose tools, reasoning, or tools_and_reasoning.
Complex reasoning must be delegated to the backend through agent_call, including
pure reasoning with required_tools=[] and suggested_capabilities=[]. For a negative
sample, choose a task that genuinely requires the program-specified missing tool;
it cannot be solved just by reasoning over the video or drafting text locally.
No pure-reasoning negative samples. The capability gap must be knowable from the
question and frontend_context alone, without watching future footage.
All prose is concise English; timestamps use MM:SS.

## Positive sample
Generate one trigger episode per sample. Choose a single occurrence in the question
(e.g. the next matching event), so a repeated occurrence does not create omitted
required calls. A sustained event/condition/target must not execute repeatedly.
Call only when the trigger is established from evidence at that time. Do not wait
for a persistent state to end: count thresholds fire at the crossing and targets
fire when their attributes can be verified. There is no forced initial silent interval; do not postpone an already due call.
If no suitable evidence or legal timing exists, return {"status":"skip","reason":"..."}.

Return one JSON object with:
- status: "ok"
- question, question_time, audio_dependency, vision_dependency (required/helpful/none;
  at least one required), work_type, required_tools (unique daily tool names)
- responses: exactly one object containing trigger_time, trigger_type
  (visual/sound/speech or '+' combination), trigger_type_reason, event_description,
  evidence, details, trigger (defined by the common annotation contract), and messages.
- messages: exactly three chronological objects, with no association IDs:
  1. role="assistant", timestamp=trigger_time, content=brief acknowledgement (<=15
     words), tool_calls=[{"type":"function","function":{"name":"agent_call",
     "arguments":"<JSON string matching the provided function parameters>"}}].
     Supply at least one video-derived extracted_parameter with evidence_time no
     later than trigger_time. suggested_capabilities contains all required_tools
     and only available tools. All call fields must be self-contained.
  2. role="tool", name="agent_call", timestamp=trigger_time + the specified delay,
     content="<JSON string of the simulated backend envelope below>".
  3. role="assistant", timestamp=same as tool result, content=concise final answer
     (<=40 words), no tool_calls. Distinguish failures, unconfirmed work and success.
All messages remain within the video duration.

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
mark uncertain parameters or confirmation needs. Pure reasoning must have empty
sources, actions and tool_information; reasoning work must provide an analysis.

## Negative sample (no_context or missing_tool)
Immediately refuse at question_time, before waiting for any trigger. Explain the
specific missing capability and which requested transaction cannot be completed.
Use only the user's instruction and frontend capabilities at that moment. Do not
cite future video, fabricate completion, invoke agent_call, or emit a backend result.
No silent interval or backend delay applies to the refusal itself.
The refusal response has exactly the following shape; also provide root modality
dependencies and the separate annotation described below:
{"status":"ok","question":"...","question_time":"MM:SS",
 "work_type":"tools|tools_and_reasoning","required_tools":["..."],
 "responses":[{"trigger_time":"<equal to question_time>","response":"<refusal>",
 "refusal_reason":"<capability limitation>","missing_tools":["<all required but unavailable tools>"]}]}.
Capability limitations alone are NOT grounds for skip. However, if the video has
no suitable real trigger, return skip in any branch. Every negative must add a
root annotation object containing the full audiovisual response annotation
(trigger_time, trigger_type, trigger_type_reason, event_description, evidence,
details and trigger; omit response and messages). Its trigger is at or after the
question. This is annotation-only future metadata: NEVER include it in the refusal
or frontend_context. The refusal must depend only on the question and capabilities.
Program-controlled sample_type, available_tools and frontend_context are attached
by the program; do not generate or modify these fields.

All available tools are shared across cognitive abilities. The ability constrains
what is perceived or inferred to trigger work, NOT which follow-up action is allowed.
Examples are suggestions, not a whitelist. Choose any relevant feasible daily work,
complex reasoning, or combination; do not invent backend tool names.
Every sources/actions entry additionally includes arguments matching that tool's
required named string parameters exactly. Missing real-world parameters must not be
fabricated: use needs_confirmation rather than claiming success. video_segment_query
start/end must be within the video prefix available at the delegation time.
The final assistant message also includes outcome equal to the backend status;
its wording must honestly reflect success, failed, or needs_confirmation.
