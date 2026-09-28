You create one dataset sample for the task "Triggered Audiovisual Complex Reasoning".

## Response-time causality and no future leakage

You may inspect the complete video to choose the question and target and to audit
coverage. That full-video access must not leak later information into the
timestamped interaction: each response and message represents only what is
knowable by its own `trigger_time` or message timestamp. Response-level trigger
fields and the first assistant acknowledgement and `agent_call` may use only
information observable or independently confirmed by their timestamps. Every
call argument, including the instruction, observed context, and extracted
parameters, is bound to the call timestamp; each parameter's `evidence_time`
must be no later than that timestamp. A tool result may add only audiovisual
reasoning available from evidence supplied at the call; it must not add later
video evidence and is bound to its tool message timestamp. The final assistant reply may use only video evidence available
by its timestamp plus tool results from earlier or the same timestamp. Later
video evidence must never revise, complete, rename, disambiguate, or otherwise
backfill an earlier response, message, or call; use generic, unknown, or
explicitly uncertain wording when cutoff evidence is insufficient.

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds).
All video-relative timestamps use MM:SS and must remain within the video.

{{QUESTION_TIME_POLICY}}

## Backend capabilities known to the frontend

{{AGENT_CAPABILITIES}}

## Workflow scope

Reason only about the current audiovisual scene. The task answers an explicit
viewer goal by inferring a likely cause, participant intent, or contextual
relationship from evidence completed by the trigger. It does not retrieve,
query, compare, cite, or act on external information.

Every successful result must combine at least two independent, confirmed scene
clues. Prefer evidence from both audio and vision. A single-modality result is
permitted only when it depends on a clear temporal sequence or relationship
between separate confirmed clues. Do not relabel observation as reasoning:
state the inference and distinguish it from its basis. Do not use what happens
later in the video to verify a hypothesis. If multiple explanations remain
plausible or the evidence cannot support a conclusion, retain that uncertainty
or return `needs_confirmation` rather than fabricate certainty.

## Question and evidence-dependent delegation

Write a natural English standing instruction with a relevant viewer objective,
attention scope, and necessary constraints. At least one core reasoning
parameter must be learned from the audiovisual trigger rather than supplied in
the question. If the full concrete reasoning request could be written from the
question alone, reject that candidate. Avoid scripted or contrived questions
that disclose future entities or speech. Select a task grounded in this video,
not a loosely associated analysis.

Prefer a trigger requiring {{PREFERRED_MODALITY}}, falling back to
{{FALLBACK_MODALITY}} only when necessary. Select exactly one trigger at or
after question_time with enough completed evidence to support the inference.
{{AGENT_TIMING_POLICY}}

`extracted_parameters` must contain at least one core parameter derived from
the video. Each entry has non-empty strings `name`, `value`, `evidence`, and an
`evidence_time` no later than `trigger_time`. `observed_context` and
`instruction` must preserve the binding between this evidence and the requested
reasoning. If key parameters are unknowable, return a `needs_confirmation`
result rather than fabricate them.

The requester is the video viewer, the people shown or heard are video
participants, and the assistant answers the viewer. First-person wording in the
question may express the viewer's needs or wish to receive results; it must not
attribute a participant's actions, speech, location, possessions, or experiences
to the viewer. Describe participants in the third person or by an observable
role, even for POV footage. Keep quoted speech attributed to its original
speaker. In assistant replies, "you/your" refers to the viewer, not a video
participant. `agent_call`, `instruction`, `user_goal`, and
`constraints_and_authorization` must preserve these identities.

## Interaction and simulated result contract

At trigger_time acknowledge in at most 15 English words and issue exactly one
`agent_call` in the same assistant message. After exactly the sampled delay,
emit the simulated tool result and an assistant final reply at the same timestamp
(at most 40 words). No result may exceed video duration. No second trigger,
progress update, or extra call is generated. All reasoning happens within the
single backend delegation. Use no id or tool_call_id; associate by message order.

The tool content is a JSON-encoded envelope with `simulated: true`, status
`success|needs_confirmation|failed`, `summary`, `details` (object), and arrays
`sources`, `actions`, and `artifacts`. For this workflow, successful results
must have empty `sources` and `actions`: all evidence is from the video, and no
external action is authorized. Successful `details` must contain non-empty
`reasoning_goal`, `conclusion`, and `basis`, plus an `uncertainties` array.
`basis` must explain the at-least-two confirmed scene clues supporting the
inference. For failed or needs_confirmation results, retain known details and
explain the uncertainty or failure in `summary`. Final replies must be supported
by the tool result and must not overstate certainty.

Return exactly one JSON object in this simplified dataset format. All natural
language must be English. `function.arguments` and tool content are JSON strings.
Each required modality must appear in `trigger_type`.

{
  "status": "ok",
  "question": "<standing instruction specifying the reasoning goal and scope>",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [
    {
      "trigger_time": "MM:SS",
      "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
      "trigger_type_reason": "...",
      "event_description": "...",
      "messages": [
        {
          "timestamp": "MM:SS",
          "role": "assistant",
          "content": "<brief acknowledgement>",
          "tool_calls": [
            {
              "type": "function",
              "function": {
                "name": "agent_call",
                "arguments": "{\"instruction\":\"...\",\"observed_context\":\"...\",\"user_goal\":\"...\",\"constraints_and_authorization\":\"Reason only from supplied audiovisual evidence; no external lookup or actions.\",\"suggested_capabilities\":[\"result_fusion\"],\"expected_output\":\"A scene-grounded reasoning conclusion, basis, and uncertainties.\",\"extracted_parameters\":[{\"name\":\"...\",\"value\":\"...\",\"evidence_time\":\"MM:SS\",\"evidence\":\"...\"}]}"
              }
            }
          ]
        },
        {
          "timestamp": "MM:SS",
          "role": "tool",
          "name": "agent_call",
          "content": "{\"simulated\":true,\"status\":\"success\",\"summary\":\"...\",\"details\":{\"reasoning_goal\":\"...\",\"conclusion\":\"...\",\"basis\":\"...\",\"uncertainties\":[]},\"sources\":[],\"actions\":[],\"artifacts\":[]}"
        },
        {
          "timestamp": "MM:SS",
          "role": "assistant",
          "content": "<concise scene-grounded result>"
        }
      ]
    }
  ]
}

If no qualifying trigger leaves enough return time, return only
`{"status":"skip","reason":"..."}`.

## Completed-event timing and timestamp representation

Before generating any acknowledgement, agent_call, or final reply, wait until
the audiovisual trigger's observable boundary has ended and the trigger is
confirmed. Never act while it is ongoing, from its onset alone, or by relying on
future evidence. `trigger_time` represents the response after that completed
boundary. Represent video times in whole-second buckets by discarding fractional
seconds; never round or carry an event into a later bucket. If completion cannot
be confirmed, remain silent or return `skip`.

## Independent video evidence audit

First inspect the original video to choose candidates; then consult events and
objects only as fallible navigation aids, and return to the video to verify each
claim and onset. Reject false annotations, correct inaccurate times, and include
verified omissions. Events and objects derived from those events are not
independent corroboration. Confidence scores are not proof. Retain and examine
supplied uncertainties; legacy metadata without evidence fields is unverified
reference. Never copy a caption claim or timestamp without checking the original
video. Distinguish visible actions, audible sounds, spoken claims, and on-screen
text: a narrator mentioning an action does not prove that action is visible.
Whole-video inspection is an annotation audit only. Each response and agent_call
may use only evidence observable at or before its own trigger_time or message
timestamp; never use later video evidence. User-visible assistant message content
must not include an MM:SS video timestamp; keep video timestamps only in
structured time fields. Remain silent without a verified trigger; if the task
cannot be grounded, return skip.
