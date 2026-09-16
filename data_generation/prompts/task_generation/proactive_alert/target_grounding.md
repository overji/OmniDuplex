You create one dataset sample for the task "Realtime Target Grounding".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). All timestamps must use MM:SS and remain within the video. The original video is the primary source of truth. The supplied event timeline and optional items are fallible search aids, never proof.

{{QUESTION_TIME_POLICY}}

## Target selection and localization

Create a natural English standing instruction asking to be told when one specific visual target or identifiable sound appears and where it is. Prefer a target requiring {{PREFERRED_MODALITY}}; use {{FALLBACK_MODALITY}} only when the preferred modality has no reliable candidate. Inspect the whole relevant video and choose a target with 1-4 unambiguous onsets at or after question_time. Do not report semantic relatives that are absent.

At each onset, localize using only evidence available then. For a visual target, describe its actual frame location rather than gaze direction and use one frame_region from top-left, top-center, top-right, middle-left, middle-center, middle-right, bottom-left, bottom-center, or bottom-right. Include a normalized [x_min,y_min,x_max,y_max] bounding_box only when supportable. For sound, report the audible source direction and include direction_degrees or distance_estimate only when the recording supports it. Use null for inapplicable optional fields. State uncertainty for occlusion, diffuse sound, or ambiguous location.

Set audio_dependency and vision_dependency to required, helpful, or none; at least one must be required. Every response must include trigger evidence and the required modality in trigger_type. Write all natural-language content in English.

Return exactly one JSON object:

{
  "status": "ok",
  "question": "Tell me when ... and where it is.",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [{
    "trigger_time": "MM:SS",
    "response": "The target is ...",
    "target": "...",
    "location_type": "visual|sound",
    "location": "...",
    "frame_region": "top-left|null",
    "bounding_box": null,
    "direction_degrees": null,
    "distance_estimate": null,
    "confidence": 0.0,
    "uncertainty": "... or none",
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If no target can be fully verified and localized, return only {"status":"skip","reason":"..."}.
