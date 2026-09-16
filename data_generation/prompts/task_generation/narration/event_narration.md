You create one dataset sample for the task "Event-Driven Realtime Narration".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; supplied metadata only helps locate candidate changes.

{{QUESTION_TIME_POLICY}}

Write one natural English instruction asking for realtime descriptions of a coherent event class, subject, or scene. Prefer a focus whose important updates require {{PREFERRED_MODALITY}}, using {{FALLBACK_MODALITY}} only if necessary. Select a focus with exactly 3-6 important, directly verifiable updates at or after question_time.

Describe each update at its onset using only evidence available then. Cover significant actions, state changes, entrances, abnormal events, or relevant relationships; omit camera cuts, persistent backgrounds, repetitions, and trivial motion. Keep context consistent. observed_facts must contain only observable facts. interpretation may offer a cautious explanation or be null and must never be presented as fact. Do not predict later events or interrupt for unimportant changes.

At least one dependency must be required. Each response must be at most 40 English words and include the required modalities in trigger_type. All natural-language content must be English.

Return exactly one JSON object:

{
  "status": "ok",
  "question": "Describe the important ... as they happen.",
  "question_time": "MM:SS",
  "audio_dependency": "required|helpful|none",
  "vision_dependency": "required|helpful|none",
  "responses": [{
    "trigger_time": "MM:SS",
    "response": "...",
    "observed_facts": ["..."],
    "interpretation": null,
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If 3-6 meaningful updates cannot be verified, return only {"status":"skip","reason":"..."}.

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
