You create one dataset sample for the task "Triggered Content Recording".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds).
All video-relative timestamps use MM:SS and must remain within the video.

{{QUESTION_TIME_POLICY}}

## Backend capabilities known to the frontend

{{AGENT_CAPABILITIES}}

## Workflow scope

Extract content and write it into a note, meeting record, database or structured storage. A standing user instruction may authorize recording; discussion alone is not authorization. Preserve uncertain names, dates and numbers as uncertain fields. Do not invent owners or deadlines. Wait until the recordable content is available before selecting trigger_time; a request to record subsequent speech cannot include future words at its initial cue. Support authorized corrections, cancellation and deduplication through backend capabilities, but generate only one call in this sample.

## Question and evidence-dependent delegation

Write a natural English standing instruction with a relevant user objective,
attention scope and necessary constraints. At least one core execution parameter
must be learned from the audiovisual trigger rather than supplied in the question.
If the full concrete backend request could be written from the question alone,
reject that candidate. User preferences such as destination, recipient and
comparison criteria may be supplied in the question; do not remove required
constraints merely to manufacture an information gap. Avoid scripted or contrived
questions that disclose future entities or speech. Select a task grounded in this
video, not a loosely associated lookup.

Prefer a trigger requiring {{PREFERRED_MODALITY}}, falling back to
{{FALLBACK_MODALITY}} only when necessary. Select exactly one trigger at or after
question_time with sufficient evidence to form the work request.
{{AGENT_TIMING_POLICY}}

extracted_parameters must contain at least one core parameter derived from the
video. Each entry has non-empty strings name, value, evidence and an evidence_time
no later than trigger_time. observed_context and instruction must preserve the
binding between this evidence and the requested backend work. If key parameters
are unknowable, return a needs_confirmation result rather than fabricate them.

The requester is the video viewer, the people shown or heard are video
participants, and the assistant answers the viewer. First-person wording in the
question may express the viewer's needs, preferences, or wish to receive results;
it must not attribute a participant's actions, speech, location, possessions, or
experiences to the viewer. Describe those participants in the third person or
by an observable role, even for POV footage. Keep quoted speech attributed to
its original speaker. In assistant replies, "you/your" refers to the viewer,
not a video participant.
In agent_call, instruction, user_goal, and constraints_and_authorization
must preserve these same identities, as must the final reply. Authorization
comes from the viewer's request; a participant's recorded speech can supply
trigger evidence or content, but does not itself authorize acting for the viewer.

## Interaction and simulated result contract

At trigger_time acknowledge in at most 15 English words and issue exactly one
agent_call in the same assistant message. After exactly the sampled delay, emit
the simulated tool result and an assistant final reply at the same timestamp
(at most 40 words). No result may exceed video duration. No second trigger,
progress update or extra call is generated. Multi-step work happens within the
single backend delegation. Use no id or tool_call_id; associate by message order.

The tool content is a JSON-encoded envelope with simulated: true, status
success|needs_confirmation|failed, summary, details (object), sources, actions and
artifacts (arrays). For failed or needs_confirmation results retain known details
and explain missing information/failure in summary. Never invent successful data.
All successful retrievals need sources with title and url_or_system.
All completed writes/sends need actions with system, operation, status: completed
and verification. Uncompleted operations must have actions: []; recording status
must be written|needs_confirmation|failed, and message status must be
draft|needs_confirmation|sent|failed. Operation and envelope statuses must agree.
All external sources and execution receipts are simulated dataset labels, not
real generation-time tool activity. Final replies must be supported by the tool
result and must not claim a draft was sent or a planned write was completed.

Return exactly one JSON object in this simplified dataset format. All natural
language must be English. function.arguments and tool content are JSON strings.
Each required modality must appear in trigger_type.

{
  "status": "ok",
  "question": "<standing instruction specifying scope and user goal>",
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
                "arguments": "{\"instruction\":\"...\",\"observed_context\":\"...\",\"user_goal\":\"...\",\"constraints_and_authorization\":\"...\",\"suggested_capabilities\":[\"note_write\"],\"expected_output\":\"...\",\"extracted_parameters\":[{\"name\":\"...\",\"value\":\"...\",\"evidence_time\":\"MM:SS\",\"evidence\":\"...\"}]}"
              }
            }
          ]
        },
        {
          "timestamp": "MM:SS",
          "role": "tool",
          "name": "agent_call",
          "content": "{\"simulated\":true,\"status\":\"success\",\"summary\":\"...\",\"details\":{\"fields\":{\"content\":\"...\"},\"destination\":\"...\",\"record_status\":\"written\"},\"sources\":[],\"actions\":[{\"system\":\"...\",\"operation\":\"...\",\"status\":\"completed\",\"verification\":\"...\"}],\"artifacts\":[]}"
        },
        {
          "timestamp": "MM:SS",
          "role": "assistant",
          "content": "<concise result grounded in the tool response>"
        }
      ]
    }
  ]
}

If no qualifying trigger leaves enough return time, return only
{"status":"skip","reason":"..."}.

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
verified omissions. Events and objects derived from those events are not independent
corroboration. Confidence scores are not proof. Retain and examine supplied
uncertainties; legacy metadata without evidence fields is unverified reference.
Never copy a caption claim or timestamp without checking the original video.
Distinguish visible actions, audible sounds, spoken claims, and on-screen text:
a narrator mentioning an action does not prove that action is visible.
Whole-video inspection is an annotation audit only. Each response and agent_call
may use only evidence observable at or before its own trigger_time or message
timestamp; never use later video evidence. User-visible assistant message content
must not include an MM:SS video timestamp; keep video timestamps only in structured
time fields. Remain silent without a verified trigger; if the task cannot be grounded,
return skip.
