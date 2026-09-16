You create one dataset sample for the task "Realtime Triggered Tool Action".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; supplied evidence only locates candidates.

{{QUESTION_TIME_POLICY}}

## Backend capabilities known to the frontend

{{AGENT_CAPABILITIES}}

## Selection and timing

Create one English standing instruction binding a specific audio or visual trigger to an external action such as recording a note, creating a task, updating a project/CRM record, sending a message, tracking a shipment, or creating a work order. Prefer a trigger requiring {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}} only when necessary. Distinguish discussion of an action from an actual request. Do not delegate a purely observational response.

Select exactly one verified trigger at or after question_time. {{AGENT_TIMING_POLICY}}

At trigger_time, acknowledge in at most 15 words while making exactly one agent_call. observed_context may include only evidence available at that time. The backend instruction must carry the requested action, extracted parameters, authorization and constraints, suggested capabilities, idempotency expectations, and desired proof of execution. If recipient, amount, time, file, permission, or another consequential field is missing, simulate needs_confirmation rather than unauthorized execution. Never claim success on a simulated failure. The final reply must contain at most 40 words.

At least one modality must be required and included in trigger_type. All natural-language content must be English. Use exactly the OpenAI-style schema below; function.arguments and tool content are JSON-encoded strings.

{
  "status":"ok",
  "question":"When ..., use my tools to ...",
  "question_time":"MM:SS",
  "audio_dependency":"required|helpful|none",
  "vision_dependency":"required|helpful|none",
  "responses":[{
    "trigger_time":"MM:SS",
    "trigger_type":"visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason":"...",
    "event_description":"...",
    "messages":[
      {"timestamp":"MM:SS","role":"assistant","content":"Got it; I'm handling that now.","tool_calls":[{"id":"call_agent_1","type":"function","function":{"name":"agent_call","arguments":"{\"instruction\":\"...\",\"observed_context\":\"...\",\"user_goal\":\"...\",\"constraints_and_authorization\":\"...\",\"suggested_capabilities\":[\"task_manage\"],\"expected_output\":\"...\"}"}}]},
      {"timestamp":"MM:SS","role":"tool","tool_call_id":"call_agent_1","name":"agent_call","content":"{\"simulated\":true,\"status\":\"success\",\"summary\":\"...\",\"details\":\"...\",\"sources\":[],\"actions\":[{\"system\":\"...\",\"operation\":\"...\",\"status\":\"completed\",\"verification\":\"...\"}],\"artifacts\":[]}"},
      {"timestamp":"MM:SS","role":"assistant","content":"..."}
    ]
  }]
}

If no safe, well-grounded trigger leaves enough return time, return only {"status":"skip","reason":"..."}.
