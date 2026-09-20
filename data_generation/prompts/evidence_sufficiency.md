You are evaluating whether a video response is supported by the video evidence available up to a specific time.

The video input includes the complete one-second interval labeled {{RESPONSE_TIME}}. You must not use information from the following second or any later footage. Judge only what is visibly or audibly established in the provided video.

Decide whether the response can be directly derived from the video by a careful viewer at {{RESPONSE_TIME}}.

Mark the evidence as sufficient only when:
- the relevant person, object, action, state, or transition is directly visible or audible;
- the response does not require guessing, common-sense assumptions, identity backfilling, or knowledge of later events; and
- the video contains enough evidence for the complete claim in the response, not merely a related partial observation.

Mark the evidence as insufficient when the response is only plausible, the relevant event is ambiguous or occluded, the response depends on later footage, or the response contains details not established by the available video. A response may be factually correct in the full video and still be insufficient at this time.

Return only one valid JSON object with exactly this shape:
{"evidence_sufficient": true/false}

Replace `true/false` with exactly one JSON boolean: `true` when the evidence is sufficient, or `false` when it is insufficient. Do not add Markdown, explanations, confidence scores, response indexes, timestamps, or extra fields.

Question:
{{QUESTION}}

Response:
{{RESPONSE}}
