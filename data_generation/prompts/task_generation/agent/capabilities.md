The backend agent can plan and combine these authorized capabilities:

- Research and retrieval: web_search, web_open, knowledge_base_search, document_search, email_search, meeting_history_search, project_history_search, and database_query.
- Live information: weather_lookup, maps_search, place_search, route_planning, traffic_lookup, transit_lookup, product_lookup, market_data_lookup, and shipment_tracking.
- Analysis: ocr_extract, document_parse, calculator, code_execute, result_fusion, fact_check, and citation_organize.
- Productivity: note_write, document_create_or_update, file_manage, spreadsheet_update, database_update, crm_update, task_manage, project_manage, calendar_manage, email_send, and message_send.
- Operations: ticket_manage, work_order_manage, approval_workflow, business_workflow, logistics_update, and authorized_device_control.
- Safety and reliability: permission_check, critical_parameter_confirmation, idempotency_check, cancellation_or_modification, error_recovery, and execution_verification.

The frontend cannot call these capabilities directly. Its only callable tool is agent_call, which delegates a complete objective and context to the backend agent. The agent must respect authorization, obtain confirmation before consequential or underspecified actions, avoid duplicate side effects, cite time-sensitive research, and return verifiable status instead of claiming unsupported success.

The frontend has this function definition:

{
  "type": "function",
  "function": {
    "name": "agent_call",
    "description": "Delegate one complex, asynchronous objective to the backend agent.",
    "parameters": {
      "type": "object",
      "properties": {
        "instruction": {"type":"string","description":"Complete, self-contained backend work request."},
        "observed_context": {"type":"string","description":"Timestamped audiovisual facts available at the trigger."},
        "user_goal": {"type":"string","description":"The result the user asked for."},
        "constraints_and_authorization": {"type":"string","description":"Scope, permissions, confirmation needs, and user constraints."},
        "suggested_capabilities": {"type":"array","items":{"type":"string"},"minItems":1},
        "expected_output": {"type":"string","description":"Required result, evidence, sources, actions, and artifacts."}
      },
      "required": ["instruction","observed_context","user_goal","constraints_and_authorization","suggested_capabilities","expected_output"],
      "additionalProperties": false
    }
  }
}
