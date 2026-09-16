You create one dataset sample for the task "Deduplicated Semantic Object Counting".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; events and items are candidate indexes only.

{{QUESTION_TIME_POLICY}}

Write one natural English instruction asking to count distinct objects, named entities, categories, products, species, or proposals across the stream. Prefer a set whose identity evidence requires {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}} only when needed. Resolve aliases and cross-modal references, merge later appearances of the same target, and keep uncertain identities separate and explicitly qualified. Count a target only at its first directly supported appearance or mention.

First appearances before question_time are historical: combine them in one immediate response at question_time. Each new target at or after question_time gets one response at its first appearance. Each event_time must correspond one-to-one to a new_targets entry. target_id must remain stable and never be counted twice. The running count equals the number of distinct target IDs observed so far.

At least one dependency must be required and included in every trigger_type. All natural-language content must be English.

Return exactly one JSON object:

{
  "status": "ok",
  "question": "Count the different ... that appear.",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [{
    "trigger_time": "MM:SS",
    "event_times": ["MM:SS"],
    "response": "That makes N different targets.",
    "count": 1,
    "new_targets": [{"target_id":"target_001","name":"...","evidence":"...","uncertain":false}],
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If distinct identities cannot be determined reliably, return only {"status":"skip","reason":"..."}.
