import copy
import json
import unittest
from dataclasses import replace

from data_generation.src.agent_tools import TASK_TOOLS
from data_generation.src.tasks import _agent
from data_generation.tests.fixtures import agent_sample


class AgentTests(unittest.TestCase):
    def test_all_families_and_work_types(self):
        for task in TASK_TOOLS:
            for work in ("tools", "reasoning", "tools_and_reasoning"):
                with self.subTest(task=task, work=work):
                    sample, context = agent_sample(task, work_type=work)
                    result = _agent.validate_result(sample, context, task)
                    self.assertEqual(result["sample_type"], "positive")
                    self.assertEqual(result["frontend_context"], _agent.frontend_context(context))
            for kind in ("no_context", "missing_tool"):
                sample, context = agent_sample(task, kind)
                result = _agent.validate_result(sample, context, task)
                self.assertEqual(result["responses"][0]["trigger_time"], result["question_time"])
                self.assertEqual(result["sample_type"], kind)

    def test_negative_does_not_require_video_delay(self):
        for kind in ("no_context", "missing_tool"):
            sample, context = agent_sample(sample_type=kind)
            context = replace(context, duration_seconds=1)
            _agent.validate_result(sample, context, "event_agent")
            context = replace(context, duration_seconds=5, question_position="non_start")
            sample["question_time"] = sample["responses"][0]["trigger_time"] = "00:02"
            _agent.validate_result(sample, context, "event_agent")

    def test_negative_invalid_variants(self):
        for kind in ("no_context", "missing_tool"):
            sample, context = agent_sample(sample_type=kind)
            mutations = [
                lambda s: s.update(status="skip", reason="Cannot do it"),
                lambda s: s.update(work_type="reasoning", required_tools=[]),
                lambda s: s.update(available_tools=["todo_create"]),
                lambda s: s.update(frontend_context={"tools": []}),
                lambda s: s["responses"][0].update(trigger_time="00:01"),
                lambda s: s["responses"][0].update(tool_calls=[]),
                lambda s: s["responses"][0].update(messages=[]),
                lambda s: s["responses"][0].update(missing_tools=[]),
                lambda s: s["responses"].append(copy.deepcopy(s["responses"][0])),
            ]
            for mutate in mutations:
                invalid = copy.deepcopy(sample)
                mutate(invalid)
                with self.subTest(kind=kind, invalid=invalid), self.assertRaises(ValueError):
                    _agent.validate_result(invalid, context, "event_agent")

    def test_trigger_constraints(self):
        cases = [
            ("event_agent", {"satisfied": False}),
            ("event_agent", {"evidence": [{"time": "00:21", "fact": "Later evidence"}]}),
            ("condition_agent", {"conditions": [{"condition": "rain", "satisfied": False}]}),
            ("condition_agent", {"conditions": [{"condition": "rain", "satisfied": True, "time": "00:21", "evidence": "rain"}]}),
            ("count_agent", {"count": 2}),
            ("count_agent", {"previous_count": 3}),
            ("count_agent", {"counting_mode": "unknown"}),
            ("count_agent", {"deduplication_rule": ""}),
            ("target_agent", {"attributes_match": False}),
        ]
        for task, change in cases:
            sample, context = agent_sample(task)
            sample["responses"][0]["trigger"].update(change)
            with self.subTest(task=task, change=change), self.assertRaises(ValueError):
                _agent.validate_result(sample, context, task)

    def test_positive_message_and_timing_constraints(self):
        mutations = [
            lambda s: s["responses"].append(copy.deepcopy(s["responses"][0])),
            lambda s: s["responses"][0]["messages"][0].update(timestamp="00:19"),
            lambda s: s["responses"][0]["messages"][1].update(timestamp="00:39"),
            lambda s: s["responses"][0]["messages"][2].update(tool_calls=[{}]),
            lambda s: s["responses"][0]["messages"][0]["tool_calls"][0].update(id="unexpected"),
            lambda s: s["responses"][0]["messages"][0]["tool_calls"].append({}),
        ]
        for mutate in mutations:
            sample, context = agent_sample()
            mutate(sample)
            with self.assertRaises(ValueError):
                _agent.validate_result(sample, context, "event_agent")
        sample, context = agent_sample()
        for context in (replace(context, duration_seconds=30), replace(context, agent_delay_seconds=None),
                        replace(context, initial_response_delay_seconds=30)):
            with self.assertRaises(ValueError):
                _agent.validate_result(sample, context, "event_agent")

    def test_delegation_cannot_invent_tools_or_future_parameters(self):
        for change in ({"suggested_capabilities": ["web_search"]}, {"suggested_capabilities": []},
                       {"extracted_parameters": [{"name": "item", "value": "parcel", "evidence": "Later", "evidence_time": "00:21"}]}):
            sample, context = agent_sample()
            function = sample["responses"][0]["messages"][0]["tool_calls"][0]["function"]
            args = json.loads(function["arguments"])
            args.update(change)
            function["arguments"] = json.dumps(args)
            with self.assertRaises(ValueError):
                _agent.validate_result(sample, context, "event_agent")

    def test_backend_evidence_and_pure_reasoning(self):
        changes = [
            ("tools", {"simulated": False}),
            ("tools", {"actions": [], "sources": []}),
            ("tools", {"actions": [{"tool": "email_send", "operation": "send", "status": "completed", "evidence": "sent"}]}),
            ("reasoning", {"tool_information": ["Weather queried."]}),
            ("reasoning", {"reasoning": []}),
        ]
        for work, change in changes:
            sample, context = agent_sample(work_type=work)
            message = sample["responses"][0]["messages"][1]
            envelope = json.loads(message["content"])
            envelope.update(change)
            message["content"] = json.dumps(envelope)
            with self.assertRaises(ValueError):
                _agent.validate_result(sample, context, "event_agent")
