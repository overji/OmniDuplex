You create one dataset sample for the task "Realtime Triggered Information Query".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; supplied evidence only locates candidates.

{{QUESTION_TIME_POLICY}}

## Backend capabilities known to the frontend

{{AGENT_CAPABILITIES}}

## Selection and timing

Create one English standing instruction that binds a specific audio or visual trigger to a genuinely external information need. Suitable work includes current web facts, manuals, weather, maps, knowledge-base material, or historical business context. Prefer a trigger requiring {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}} only when necessary. Do not invoke the agent for something directly answerable from the current video.

Select exactly one verified trigger at or after question_time and use no future video evidence in observed_context. {{AGENT_TIMING_POLICY}}

At trigger_time, the frontend must acknowledge the user in at most 15 words while making exactly one agent_call. Its arguments must give the backend a complete instruction, observed trigger evidence and time, user goal, constraints/authorization, suggested capabilities, and expected output. The simulated tool result must be clearly labeled simulated, include source metadata appropriate to the query, and must not be described as a real generation-time retrieval. The final assistant reply must use only that result and contain at most 40 words.

At least one modality must be required and included in trigger_type. All natural-language content must be English. Output exactly the schema below; function.arguments and tool content are JSON-encoded strings.

{
  "status":"ok",
  "question":"When ..., look up ... for me.",
  "question_time":"MM:SS",
  "audio_dependency":"required|helpful|none",
  "vision_dependency":"required|helpful|none",
  "responses":[{
    "trigger_time":"MM:SS",
    "trigger_type":"visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason":"...",
    "event_description":"...",
    "messages":[
      {"timestamp":"MM:SS","role":"assistant","content":"I found it; I'm checking now.","tool_calls":[{"type":"function","function":{"name":"agent_call","arguments":"{\"instruction\":\"...\",\"observed_context\":\"...\",\"user_goal\":\"...\",\"constraints_and_authorization\":\"...\",\"suggested_capabilities\":[\"web_search\"],\"expected_output\":\"...\"}"}}]},
      {"timestamp":"MM:SS","role":"tool","name":"agent_call","content":"{\"simulated\":true,\"status\":\"success\",\"summary\":\"...\",\"details\":\"...\",\"sources\":[{\"title\":\"...\",\"url_or_system\":\"...\",\"retrieved_at\":\"MM:SS\"}],\"actions\":[],\"artifacts\":[]}"},
      {"timestamp":"MM:SS","role":"assistant","content":"..."}
    ]
  }]
}

If no eligible trigger leaves enough return time, return only {"status":"skip","reason":"..."}.

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
may use only evidence available by its own timestamp. Remain silent without a
verified trigger; if the task cannot be grounded, return skip.

Associate the sole call and result by their fixed message order. Do not output
id or tool_call_id on calls or messages. This is a simplified dataset trace,
not a directly submit-ready tool protocol. The first assistant message contains
acknowledgement and agent_call at trigger_time; the tool result arrives exactly
the sampled delay later, followed by the final assistant reply at that same time.
