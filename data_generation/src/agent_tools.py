"""Shared daily application capabilities; reasoning is not a tool."""

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

# Each tuple is the required named string arguments of a simulated invocation.
PARAMETERS = {
    "todo_create": ("title", "owner", "deadline"),
    "todo_query": ("query",), "todo_update": ("todo_id", "changes"),
    "calendar_query": ("time_range",), "note_write": ("title", "content"),
    "order_query": ("query",), "order_update": ("order_id", "changes"),
    "note_search": ("query",), "document_search": ("query",),
    "weather_query": ("place", "time"), "route_query": ("origin", "destination"),
    "meeting_draft": ("title", "content"), "shopping_list_query": ("list_name",),
    "shopping_list_update": ("list_name", "changes"), "budget_query": ("query",),
    "budget_update": ("record",), "calculator": ("expression",),
    "itinerary_query": ("time_range",), "product_search": ("query",),
    "calendar_create": ("title", "start", "end"),
    "calendar_update": ("event_id", "changes"),
    "notification_send": ("recipient", "channel", "message"),
    "web_search": ("query",), "knowledge_search": ("query",),
    "history_query": ("query",), "video_segment_query": ("start", "end", "query"),
    "statistics_compute": ("data", "analysis"),
    "report_create": ("title", "content"),
}
TOOLS.update({
    "calendar_create": "Create a user-authorized calendar event; do not invite unspecified people.",
    "calendar_update": "Modify an identified existing event within the user's requested scope.",
    "notification_send": "Send the authorized message to the specified recipient and channel; require confirmation if recipient or authorization is unresolved.",
    "web_search": "Retrieve public information and source references; cannot execute transactions.",
    "knowledge_search": "Search connected knowledge bases; no access beyond the provided connection.",
    "history_query": "Read connected historical records, including faults or activity logs.",
    "video_segment_query": "Retrieve already observed video context only, with MM:SS start/end no later than delegation time; never retrieve future footage.",
    "statistics_compute": "Calculate requested statistics from supplied data; do not invent missing measurements.",
    "report_create": "Create a report artifact from supplied facts and analysis; cannot publish or send it.",
})
WRITE_TOOLS = frozenset({
    "todo_create", "todo_update", "note_write", "order_update", "meeting_draft",
    "shopping_list_update", "budget_update", "calendar_create", "calendar_update",
    "notification_send", "report_create",
})


def describe_tool(name: str) -> str:
    kind = "write" if name in WRITE_TOOLS else "read/compute"
    return f"{TOOLS[name]} Operation: {kind}. Required string arguments: {', '.join(PARAMETERS[name])}."


def validate_arguments(name: str, arguments: object) -> None:
    if not isinstance(arguments, dict) or set(arguments) != set(PARAMETERS[name]):
        raise ValueError(f"{name}: arguments must match {PARAMETERS[name]}")
    if any(not isinstance(value, str) or not value.strip() for value in arguments.values()):
        raise ValueError(f"{name}: arguments must be non-empty strings")
