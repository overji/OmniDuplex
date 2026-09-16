You create one dataset sample for the task "Triggered Snapshot Counting".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; event descriptions and optional items only identify candidates.

{{QUESTION_TIME_POLICY}}

Write a natural English instruction that names one discrete audio or visual trigger and one visibly countable target, such as "Whenever the whistle sounds, tell me how many players are visible." Prefer a trigger requiring {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}} only when necessary.

Audit the complete video after question_time. For every independently verified trigger onset, count targets in the scene at that exact moment. Count each visible target once, handle brief occlusion conservatively, and never replace the current snapshot count with a cumulative count. Do not report anything when the trigger does not occur. Choose another target or return skip if any complete count is ambiguous.

Vision must be required because the reported quantity comes from the current image. Audio may also be required when the trigger is audible. Include the evidence supporting both the trigger and count. All natural-language content must be English.

Return exactly one JSON object:

{
  "status": "ok",
  "question": "Whenever ..., tell me how many ... are visible.",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required",
  "responses": [{
    "trigger_time": "MM:SS",
    "response": "There are N ... visible now.",
    "count": 0,
    "counting_basis": "Observable targets counted at the trigger frame.",
    "target_locations": [],
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If no reliable trigger/count pair exists, return only {"status":"skip","reason":"..."}.
