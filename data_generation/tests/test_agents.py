import copy
import json
import unittest
from dataclasses import replace

from data_generation.src.agent_tools import TOOLS, PARAMETERS
from data_generation.src.catalog import ABILITIES
from data_generation.src.registry import get_task
from data_generation.src.tasks import _agent
from data_generation.tests.fixtures import agent_sample


def validate(sample, context, task="event_understanding_agent"):
    return get_task(task).validate_result(sample, context)


class AgentTests(unittest.TestCase):
    def test_all_abilities_work_types_and_negative_branches(self):
        for ability in ABILITIES:
            task = f"{ability}_agent"
            for work in ("tools", "reasoning", "tools_and_reasoning"):
                with self.subTest(task=task, work=work):
                    sample, context = agent_sample(task, work_type=work)
                    result = validate(sample, context, task)
                    self.assertEqual(result["sample_type"], "positive")
                    self.assertEqual(result["frontend_context"], _agent.frontend_context(context))
            for kind in ("no_context", "missing_tool"):
                sample, context = agent_sample(task, kind)
                result = validate(sample, context, task)
                self.assertEqual(result["responses"][0]["trigger_time"], result["question_time"])
                self.assertGreater(result["annotation"]["trigger_time"], result["question_time"])
                self.assertNotIn("annotation", result["frontend_context"])

    def test_negative_invalid_variants_and_skip(self):
        for kind in ("no_context", "missing_tool"):
            sample, context = agent_sample(sample_type=kind)
            self.assertEqual(validate({"status": "skip", "reason": "No suitable real trigger."}, context)["status"], "skip")
            mutations = [
                lambda s: s.update(work_type="reasoning", required_tools=[]),
                lambda s: s.update(available_tools=["todo_create"]),
                lambda s: s.update(frontend_context={"tools": []}),
                lambda s: s["responses"][0].update(trigger_time="00:01"),
                lambda s: s["responses"][0].update(tool_calls=[]),
                lambda s: s["responses"][0].update(messages=[]),
                lambda s: s["responses"][0].update(missing_tools=[]),
                lambda s: s["responses"].append(copy.deepcopy(s["responses"][0])),
                lambda s: s.pop("annotation"),
                lambda s: s["annotation"]["evidence"][0].update(available_time="00:21"),
            ]
            for mutate in mutations:
                invalid = copy.deepcopy(sample)
                mutate(invalid)
                with self.subTest(kind=kind, invalid=invalid), self.assertRaises(ValueError):
                    validate(invalid, context)

    def test_positive_message_and_timing_constraints(self):
        mutations = [
            lambda s: s["responses"].append(copy.deepcopy(s["responses"][0])),
            lambda s: s["responses"][0]["messages"][0].update(timestamp="00:19"),
            lambda s: s["responses"][0]["messages"][1].update(timestamp="00:39"),
            lambda s: s["responses"][0]["messages"][2].update(tool_calls=[{}]),
            lambda s: s["responses"][0]["messages"][0]["tool_calls"][0].update(id="unexpected"),
            lambda s: s["responses"][0]["messages"][0]["tool_calls"].append({}),
            lambda s: s["responses"][0]["messages"][2].update(outcome="failed"),
        ]
        for mutate in mutations:
            sample, context = agent_sample()
            mutate(sample)
            with self.assertRaises(ValueError):
                validate(sample, context)
        sample, context = agent_sample()
        for bad in (replace(context, duration_seconds=30), replace(context, agent_delay_seconds=None),
                    replace(context, initial_response_delay_seconds=30)):
            with self.assertRaises(ValueError):
                validate(sample, bad)

    def test_delegation_cannot_invent_tools_or_future_parameters(self):
        for change in ({"suggested_capabilities": ["invented_tool"]}, {"suggested_capabilities": []},
                       {"extracted_parameters": [{"name": "item", "value": "parcel", "evidence": "Later", "evidence_time": "00:21"}]}):
            sample, context = agent_sample()
            function = sample["responses"][0]["messages"][0]["tool_calls"][0]["function"]
            args = json.loads(function["arguments"])
            args.update(change)
            function["arguments"] = json.dumps(args)
            with self.assertRaises(ValueError):
                validate(sample, context)

    def test_backend_evidence_and_pure_reasoning(self):
        changes = [
            ("tools", {"simulated": False}),
            ("tools", {"actions": [], "sources": []}),
            ("tools", {"actions": [{"tool": "invented", "operation": "send", "status": "completed", "evidence": "sent"}]}),
            ("reasoning", {"tool_information": ["Weather queried."]}),
            ("reasoning", {"reasoning": []}),
            ("tools", {"actions": [{"tool": "todo_create", "operation": "create", "status": "completed", "evidence": "record", "arguments": {}}]}),
        ]
        for work, change in changes:
            sample, context = agent_sample(work_type=work)
            message = sample["responses"][0]["messages"][1]
            envelope = json.loads(message["content"])
            envelope.update(change)
            message["content"] = json.dumps(envelope)
            with self.assertRaises(ValueError):
                validate(sample, context)

    def test_tools_are_shared_and_explicit_and_video_query_is_prefix_only(self):
        self.assertEqual(set(TOOLS), set(PARAMETERS))
        sample, context = agent_sample("emotion_recognition_agent")
        sample["required_tools"] = ["video_segment_query"]
        function = sample["responses"][0]["messages"][0]["tool_calls"][0]["function"]
        arguments = json.loads(function["arguments"])
        arguments["suggested_capabilities"] = ["video_segment_query"]
        function["arguments"] = json.dumps(arguments)
        message = sample["responses"][0]["messages"][1]
        envelope = json.loads(message["content"])
        envelope["actions"] = []
        envelope["sources"] = [{"tool": "video_segment_query", "arguments": {"start": "00:00", "end": "00:20", "query": "context"}, "evidence": "Retrieved prefix"}]
        message["content"] = json.dumps(envelope)
        validate(sample, context, "emotion_recognition_agent")
        envelope["sources"][0]["arguments"]["end"] = "00:21"
        message["content"] = json.dumps(envelope)
        with self.assertRaises(ValueError):
            validate(sample, context, "emotion_recognition_agent")

    def test_failure_and_confirmation_outcomes(self):
        for status, action_status in (("failed", "failed"), ("needs_confirmation", "needs_confirmation")):
            sample, context = agent_sample()
            messages = sample["responses"][0]["messages"]
            envelope = json.loads(messages[1]["content"])
            envelope["status"] = status
            envelope["actions"][0]["status"] = action_status
            envelope["summary"] = "Could not finish the action."
            messages[1]["content"] = json.dumps(envelope)
            messages[2].update(outcome=status, content="The action has not been completed.")
            validate(sample, context)
            envelope["status"] = "success"
            messages[1]["content"] = json.dumps(envelope)
            messages[2]["outcome"] = "success"
            with self.assertRaises(ValueError):
                validate(sample, context)

    def test_extra_call_arguments_and_context_tools_are_rejected(self):
        sample, context = agent_sample()
        function = sample["responses"][0]["messages"][0]["tool_calls"][0]["function"]
        arguments = json.loads(function["arguments"])
        arguments["future_caption"] = "Something later."
        function["arguments"] = json.dumps(arguments)
        with self.assertRaises(ValueError):
            validate(sample, context)
        sample, context = agent_sample(sample_type="missing_tool")
        with self.assertRaises(ValueError):
            validate(sample, replace(context, available_tools=("note_write",)))

    def test_read_tools_cannot_claim_write_execution(self):
        sample, context = agent_sample()
        sample["required_tools"] = ["calendar_query"]
        messages = sample["responses"][0]["messages"]
        function = messages[0]["tool_calls"][0]["function"]
        arguments = json.loads(function["arguments"])
        arguments["suggested_capabilities"] = ["calendar_query"]
        function["arguments"] = json.dumps(arguments)
        envelope = json.loads(messages[1]["content"])
        envelope["actions"][0].update(tool="calendar_query", arguments={"time_range": "today"})
        messages[1]["content"] = json.dumps(envelope)
        with self.assertRaisesRegex(ValueError, "read-only"):
            validate(sample, context)
