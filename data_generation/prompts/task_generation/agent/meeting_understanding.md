You create one dataset sample for the task "Meeting Understanding and Follow-up".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original meeting audio, video, and visible shared content are authoritative; supplied metadata only locates candidates.

{{QUESTION_TIME_POLICY}}

## Backend capabilities known to the frontend

{{AGENT_CAPABILITIES}}

## Selection and timing

Create one English meeting-assistant instruction that calls for a backend-worthy workflow when one meaningful meeting trigger occurs: relating a statement to prior documents, checking a conflict with history, preparing structured minutes, creating authorized follow-ups, or organizing a confirmed decision and its evidence. Prefer a trigger requiring {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}} only when necessary. Do not delegate ordinary transcription or a fact directly observable in the meeting.

Select exactly one trigger at or after question_time and identify speakers only when supported. Distinguish facts, opinions, proposals, decisions, disagreements, risks, and unresolved questions. Preserve original evidence for owners, dates, numbers, and project names; mark uncertain transcription. {{AGENT_TIMING_POLICY}}

At trigger_time, acknowledge in at most 15 words while issuing exactly one agent_call containing the complete meeting context available then, user goal, privacy/authorization constraints, suggested capabilities, and expected artifacts/actions. The simulated result may contain minutes, decisions, tasks, draft messages, or clarification requests, but consequential sends or calendar/task changes require adequate authorization. The final response must faithfully summarize only the simulated result in at most 40 words.

At least one modality must be required and included in trigger_type. All natural-language content must be English. Use exactly the OpenAI-style schema below; function.arguments and tool content are JSON-encoded strings.

{
  "status":"ok",
  "question":"During this meeting, when ..., ...",
  "question_time":"MM:SS",
  "audio_dependency":"required|helpful|none",
  "vision_dependency":"required|helpful|none",
  "responses":[{
    "trigger_time":"MM:SS",
    "trigger_type":"visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason":"...",
    "event_description":"...",
    "messages":[
      {"timestamp":"MM:SS","role":"assistant","content":"I caught that; I'm preparing the follow-up.","tool_calls":[{"id":"call_agent_1","type":"function","function":{"name":"agent_call","arguments":"{\"instruction\":\"...\",\"observed_context\":\"...\",\"user_goal\":\"...\",\"constraints_and_authorization\":\"...\",\"suggested_capabilities\":[\"document_create_or_update\"],\"expected_output\":\"...\"}"}}]},
      {"timestamp":"MM:SS","role":"tool","tool_call_id":"call_agent_1","name":"agent_call","content":"{\"simulated\":true,\"status\":\"success\",\"summary\":\"...\",\"details\":\"...\",\"sources\":[],\"actions\":[],\"artifacts\":[{\"type\":\"meeting_minutes\",\"content\":\"...\",\"evidence\":\"...\"}]}"},
      {"timestamp":"MM:SS","role":"assistant","content":"..."}
    ]
  }]
}

If no qualifying meeting workflow leaves enough return time, return only {"status":"skip","reason":"..."}.
