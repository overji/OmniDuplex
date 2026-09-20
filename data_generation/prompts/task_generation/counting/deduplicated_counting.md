You create one dataset sample for the task "Deduplicated Semantic Object Counting".

## Response-time causality and no future leakage

You may inspect the complete video to choose the question and target and to audit
coverage. That full-video access must not leak later information into a
timestamped response: each response represents only what is knowable by its own
`trigger_time`. Every field nested in that response, including response text,
counts, names or identities, locations, evidence, interpretations, event lists,
trigger reasons and descriptions, and other metadata, may use only information
observable or independently confirmed no later than `trigger_time`. A historical
summary emitted at `question_time` uses `question_time` as its cutoff. Later
video evidence may support later responses but must never revise, complete,
rename, disambiguate, or otherwise backfill an earlier response; use generic,
unknown, or explicitly uncertain wording when cutoff evidence is insufficient.

The video duration is {{VIDEO_DURATION}} ({{VIDEO_DURATION_SECONDS}} seconds). Use MM:SS timestamps within the video. The original video is authoritative; events and items are candidate indexes only.

{{QUESTION_TIME_POLICY}}

Choose exactly one concrete, bounded target category supported by this video and name it explicitly in the English question. Valid examples: "Count the different car brands that appear", "Count the bird species shown", or "Count the product names mentioned in the meeting". Invalid: "Count the different items and entities that appear or are mentioned in the video", "Count everything", or a mixture of unrelated categories. Examples illustrate specificity; never force their categories onto unrelated footage. Set target_category to the chosen category and counting_unit to instance, category, or name; state what makes two targets distinct. The question, target list, and deduplication rule must use that same scope. Prefer a set whose identity evidence requires {{PREFERRED_MODALITY}}, falling back to {{FALLBACK_MODALITY}} only when needed. Resolve aliases and cross-modal references, merge later appearances of the same target, and keep uncertain identities separate and explicitly qualified. Count a target only at its first directly supported appearance or mention.

First appearances before question_time are historical: combine them in one response at question_time without the sampled silent interval. If no historical first appearance exists, the first response must observe the required silent interval. When a historical summary exists, later new targets are not delayed by that initial interval and each gets one response after its first directly supported appearance is complete and confirmed, when permitted by the response timing rule. Each event_time must correspond one-to-one to a new_targets entry. target_id must remain stable and never be counted twice. The running count equals the number of distinct target IDs observed so far.

At least one dependency must be required and included in every trigger_type. All natural-language content must be English.

The requester is the video viewer, the people shown or heard are video
participants, and the assistant answers the viewer. First-person wording in the
question may express the viewer's needs, preferences, or wish to receive results;
it must not attribute a participant's actions, speech, location, possessions, or
experiences to the viewer. Describe those participants in the third person or
by an observable role, even for POV footage. Keep quoted speech attributed to
its original speaker. In assistant replies, "you/your" refers to the viewer,
not a video participant.

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
    "response": "<new object name, question category, and cumulative count>",
    "count": 1,
    "new_targets": [{"target_id":"target_001","name":"...","evidence":"...","uncertain":false}],
    "trigger_type": "visual|sound|speech|visual+sound|visual+speech",
    "trigger_type_reason": "...",
    "event_description": "..."
  }]
}

If distinct identities cannot be determined reliably, return only {"status":"skip","reason":"..."}.

## Completed-event timing and timestamp representation

Before generating any response, wait until the triggering event's observable
boundary has ended and the event is confirmed. Never respond while the event is
ongoing, from its onset alone, or by relying on future evidence. `trigger_time`
and every event-time field represent the response after that completed boundary.
Represent video times in whole-second buckets by discarding fractional seconds;
never round or carry an event into a later bucket. Preserve chronological order
and apply this task's existing deduplication when multiple observations share a
bucket. If completion cannot be confirmed, remain silent or return `skip`.

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
may use only evidence observable at or before its own trigger_time or message
timestamp; never use later video evidence. User-visible response text must not
include an MM:SS video timestamp; keep video timestamps only in structured time
fields. Remain silent without a verified trigger; if the task cannot be grounded,
return skip.

## Question-aligned replies

Each response must explicitly name the newly identified object and the category
specified in the question, together with the cumulative count. Keep it concise
and grammatical for singular and plural counts. Historical summaries may group
names and state the total for that category. Do not say "different targets" or
use generic target labels in user-facing replies. Do not invent category labels
or identities that cannot be established at the response time.
