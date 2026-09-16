You create one dataset sample for the task "Deduplicated Semantic Object Counting".

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; events and items are candidate indexes only.

{{QUESTION_TIME_POLICY}}

Choose exactly one concrete, bounded target category supported by this video and name it explicitly in the English question. Valid examples: "Count the different car brands that appear", "Count the bird species shown", or "Count the product names mentioned in the meeting". Invalid: "Count the different items and entities that appear or are mentioned in the video", "Count everything", or a mixture of unrelated categories. Examples illustrate specificity; never force their categories onto unrelated footage. Set target_category to the chosen category and counting_unit to instance, category, or name; state what makes two targets distinct. The question, target list, and deduplication rule must use that same scope. Prefer a set whose identity evidence requires {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}} only when needed. Resolve aliases and cross-modal references, merge later appearances of the same target, and keep uncertain identities separate and explicitly qualified. Count a target only at its first directly supported appearance or mention.

First appearances before question_time are historical: combine them in one immediate response at question_time. Each new target at or after question_time gets one response at its first appearance. Each event_time must correspond one-to-one to a new_targets entry. target_id must remain stable and never be counted twice. The running count equals the number of distinct target IDs observed so far.

At least one dependency must be required and included in every trigger_type. All natural-language content must be English.

Return exactly one JSON object:

{
  "status": "ok",
  "question": "Count the different car brands that appear.",
  "target_category": "car brands",
  "counting_unit": "name",
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
