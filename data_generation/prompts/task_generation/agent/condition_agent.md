## Condition-triggered agent
When all necessary semantic conditions hold within the user's specified window,
prepare information, perform a daily transaction, or reason about a useful suggestion.
Examples: when budget discussion begins and a question remains unanswered, retrieve
relevant documents and suggest a question; when departure preparations coincide
with rain, query a route and reason about departure timing.
Allowed tools: personal note/document search, calendar/weather/route queries and
meeting-record drafts. Require evidence for EVERY necessary condition in the same
applicable window. Imminent, ambiguous, or only partially met conditions do not fire.
Positive trigger: {"kind":"condition","satisfied":true,
"evidence":[{"time":"MM:SS","fact":"..."}],
"conditions":[{"condition":"<each necessary condition>","satisfied":true,
"time":"MM:SS","evidence":"<evidence available at trigger_time>"}]}.
