You create one dataset sample for the task "Realtime Target Grounding".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). All timestamps must use MM:SS and remain within the video. The original video is the primary source of truth. The supplied event timeline and optional items are fallible search aids, never proof.

{{QUESTION_TIME_POLICY}}

## Target selection and localization

Create a natural English standing instruction asking to be told when one specific visual target or identifiable sound appears and where it is. Prefer a target requiring {{PREFERRED_MODALITY}}; use {{FALLBACK_MODALITY}} only when the preferred modality has no reliable candidate. Inspect the whole relevant video and choose a target with 1-4 unambiguous onsets at or after question_time. Do not report semantic relatives that are absent.

At each onset, describe only a coarse direction using evidence available then.
For visual targets use natural English such as "upper left" or "lower right",
and one frame_region from top-left, top-center, top-right, middle-left,
middle-center, middle-right, bottom-left, bottom-center, or bottom-right.
Describe frame position, not gaze direction. Never output bounding boxes, pixel
positions, coordinates, numeric angles, or distance estimates, including in prose.
For sound use a supported coarse direction such as "to the left"; frame_region
must be null. Do not infer source direction from the meaning of speech, a visual
association, or a mono recording without directional evidence. Express uncertainty
for occlusion or ambiguous direction; skip when localization is unsupported.

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
    "frame_region": "top-left",
    "confidence": 0.0,
    "uncertainty": "... or none",
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If no target can be fully verified and localized, return only {"status":"skip","reason":"..."}.

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
