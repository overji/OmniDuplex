"""Daily application tools by trigger family; reasoning is not a tool."""

TOOLS = {
    "todo_create": "Create a user-authorized todo with extracted owner and deadline.",
    "todo_query": "Query existing todos and their status.",
    "todo_update": "Update existing todos; do not create unrelated tasks.",
    "calendar_query": "Read scheduled events and availability; cannot edit the calendar.",
    "note_write": "Write user-authorized notes.",
    "order_query": "Query orders and packages awaiting delivery.",
    "order_update": "Update existing order or package records; cannot place purchases.",
    "note_search": "Search personal notes.",
    "document_search": "Search available personal documents.",
    "weather_query": "Query weather for a specified place and time.",
    "route_query": "Query maps and routes, including station exits and walking routes.",
    "meeting_draft": "Prepare a meeting-record draft; cannot send it.",
    "shopping_list_query": "Read the user's shopping list.",
    "shopping_list_update": "Update shopping-list items and quantities.",
    "budget_query": "Read budgets and existing accounting records.",
    "budget_update": "Record expenses or update a user-authorized budget.",
    "calculator": "Calculate values from provided numbers.",
    "itinerary_query": "Read the user's itinerary.",
    "product_search": "Search product specifications, prices, and available offers.",
}

TASK_TOOLS = {
    "event_agent": ("todo_create", "todo_query", "calendar_query", "note_write", "order_query", "order_update"),
    "condition_agent": ("note_search", "document_search", "calendar_query", "weather_query", "route_query", "meeting_draft"),
    "count_agent": ("todo_query", "todo_update", "calendar_query", "shopping_list_query", "shopping_list_update", "budget_query", "budget_update", "calculator"),
    "target_agent": ("itinerary_query", "route_query", "shopping_list_query", "product_search", "calculator"),
}


WRITE_TOOLS = frozenset({
    "todo_create", "todo_update", "note_write", "order_update", "meeting_draft",
    "shopping_list_update", "budget_update",
})
