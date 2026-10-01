## Target-triggered agent
When the specified visual or audio target appears, perform the requested related
transaction or complex reasoning. Verify the actual requested attributes, not a
merely similar target. Distinguish current-station announcements from next-stop
previews and product names from matching sizes/specifications. Same continuous
appearance executes once. Choose the next matching appearance for this sample.
Examples: query the onward route and compare exits when the destination station
is announced; find discounts and reason about buying a matching shopping-list item.
Allowed tools: itinerary, route/station navigation, shopping-list queries,
product/offer search and calculator.
Positive trigger: {"kind":"target","satisfied":true,
"evidence":[{"time":"MM:SS","fact":"<identity and attribute evidence>"}],
"target":"<specific requested target>","attributes_match":true}.
