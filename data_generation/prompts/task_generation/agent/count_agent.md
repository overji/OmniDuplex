## Count-triggered agent
Execute a related task when a count first crosses the requested positive threshold
from below. Explicitly choose current quantity, cumulative independent occurrences,
or distinct identities. Define the observation scope and deduplication rule in the
question. Repeated mentions or reappearance of the same identity must not increase
a distinct count. Continuous presence is not repeated occurrences. Never fire just
because a quantity remains above threshold. No future evidence may revise this count.
Examples: organize three assigned todos and analyze schedule conflicts; update a
shopping list and budget when five matching items are in the cart.
Allowed tools: todo reads/updates, calendar reads, shopping-list reads/updates,
budget/accounting reads/updates and calculator.
Positive trigger: {"kind":"count","satisfied":true,
"evidence":[{"time":"MM:SS","fact":"<count evidence>"}],
"counting_mode":"current|cumulative|distinct","threshold":3,
"previous_count":2,"count":3,"counting_basis":"...","deduplication_rule":"..."}.
Numbers above are examples; require previous_count < threshold <= count.
