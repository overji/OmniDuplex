You create one dataset sample for the task "Semantic Condition Alert".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is the primary source of truth; supplied events and items are fallible candidate indexes.

{{QUESTION_TIME_POLICY}}

Create one natural English standing instruction containing a meaningful semantic condition, such as a previously seen target reappearing, a clearly suitable scene, a discussion entering a specified topic, or an observable hazardous/crowded condition. Prefer evidence requiring {{PREFERRED_MODALITY}} and fall back to {{FALLBACK_MODALITY}} only when needed. The condition must map to explicit audiovisual criteria; define those criteria internally without revealing future events in the question.

Evidence before question_time may establish a reference for "again" or "similar," but never emit a response before question_time. Scan all later video and report every clear satisfaction onset once. A persistent condition is one occurrence until it ends and begins again. Avoid weak associations, explain the evidence, and express uncertainty conservatively rather than forcing a match. Do not use later frames to justify an earlier alert.

At least one dependency must be required, and each trigger_type must contain every required modality. All natural-language fields must be English.

Return exactly one JSON object:

{
  "status": "ok",
  "question": "Let me know when ...",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [{
    "trigger_time": "MM:SS",
    "response": "...",
    "condition_evidence": "...",
    "confidence": 0.0,
    "uncertainty": "... or none",
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If no defensible semantic condition has a later trigger, return only {"status":"skip","reason":"..."}.
