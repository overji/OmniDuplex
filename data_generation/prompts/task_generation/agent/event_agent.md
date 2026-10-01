## Event-triggered agent
After the next user-specified event, handle a related transaction or complex
reasoning goal. Examples: record an assigned task and reason about scheduling;
match a delivered package with an order and prioritize handling it.
Allowed tool scope: todos, calendar reads, note writes, order/package reads and updates.
Extract assignments, owners, deadlines, or package information cautiously; do not
claim illegible identities. The event must actually happen; a prediction is not a trigger.
Positive trigger: {"kind":"event","satisfied":true,
"evidence":[{"time":"MM:SS","fact":"<observable event evidence>"}]}.
