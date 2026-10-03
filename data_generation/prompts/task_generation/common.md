You are an expert at constructing QA benchmark data for proactive omni-modal assistants.
Reference design: local arXiv-2605.18577v1 appendix/data_details.tex and existing dense-caption task prompts.

The original audiovisual video is authoritative. The complete dense caption is a
fallible navigation reference, never ground truth. Verify all claims and exact
moments against the video; caption segment boundaries are NOT event timestamps.
Full-video access is for annotation only. Every response and delegation can use
only information available at its timestamp, never later identities, outcomes or
captions. The viewer is not automatically a participant in the video.

Write a natural standing instruction with a specific focus, no spoilers or supplied
answers. A question may refer to a previously observed target and specify user
preferences, but cannot reveal future video facts. All prose is concise English.
Use MM:SS timestamps. Return raw JSON only. If no suitable real evidence exists,
return {"status":"skip","reason":"..."}. Do not generate all-video no-trigger
silence samples, including in Agent capability-negative branches.

The sampled modality is a preference only. Honor the cognitive ability's modality
requirements; do not turn visible-text understanding into audio transcription.
Do not force arbitrary response counts, target sizes or fixed intervals from old
benchmark templates. Updates should add meaningful information.

For status=ok, supply question, question_time, audio_dependency and vision_dependency
(each required/helpful/none; at least one required), and responses.
Each audiovisual response/annotation contains:
{"trigger_time":"MM:SS", "response":"concise natural response (omit for Agent delegation)",
 "trigger_type":"visual|sound|speech or '+' combination",
 "trigger_type_reason":"why these modalities", "event_description":"...",
 "evidence":[{"time":"MM:SS", "available_time":"MM:SS", "fact":"..."}],
 "details": {"ability-specific fields":"see below"}}
Evidence must be nonempty. time <= available_time <= trigger_time; include the
latest necessary evidence, not just earlier convenient facts. For speech-dependent
semantics, available_time is when enough of the utterance has been heard.
The ability and mode identifiers are program-controlled and must not be changed.

For PR/Agent annotations also supply:
"trigger": {"kind":"event|condition|count|target", "satisfied":true,
 "episode_id":"unique occurrence ID", "confirmed_time":"MM:SS",
 "definition":"the user-specified criterion"}.
confirmed_time equals trigger_time: never postpone an already confirmable trigger.
Use a kind appropriate to this instance, not a hardcoded mapping from ability.
For condition triggers add conditions:[{condition, satisfied:true, time, evidence}]
for ALL necessary conditions. For target triggers add target and attributes_match:true.
For count triggers add previous_count, count, threshold and operator (ge|gt|le|lt|eq);
the previous count must fail and the current count must satisfy this predicate.
For repeated condition/target/count episodes add reset_time, strictly between the
previous trigger and this one, documenting when the previous condition stopped
holding. A sustained satisfied condition is ONE episode, not repeated execution.
