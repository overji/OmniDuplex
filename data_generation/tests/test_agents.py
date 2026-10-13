import copy
import json
import unittest
from dataclasses import replace

from data_generation.src.domain.agent_tools import TOOLS, PARAMETERS
from data_generation.src.domain.catalog import ABILITIES
from data_generation.src.application.registry import get_task
from data_generation.src.domain.tasks import _agent
from data_generation.tests.fixtures import agent_sample, observation
from data_generation.src.domain.qa import adapt_responses


def validate(sample, context, task="event_understanding_agent"):
    return get_task(task).validate_result(sample, context)


class AgentTests(unittest.TestCase):
    def test_all_abilities_work_types_and_negative_branches(self):
        for ability in ABILITIES:
            task = f"{ability}_agent"
            for work in ("tools", "tools_and_reasoning"):
                with self.subTest(task=task, work=work):
                    sample, context = agent_sample(task, work_type=work)
                    result = validate(sample, context, task)
                    self.assertEqual(result["sample_type"], "positive")
                    self.assertEqual(result["frontend_context"], _agent.frontend_context(context, get_task(task).capabilities_template))
            for kind in ("no_context", "missing_tool"):
                sample, context = agent_sample(task, kind)
                result = validate(sample, context, task)
                self.assertEqual(result["responses"][0]["trigger_time"], result["question_time"])
                self.assertGreater(result["annotations"][0]["trigger_time"], result["question_time"])
                self.assertNotIn("annotations", result["frontend_context"])

    def test_all_agent_abilities_require_actual_tools(self):
        for ability in ABILITIES:
            task = f"{ability}_agent"
            sample, context = agent_sample(task, work_type="reasoning")
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "pure reasoning"):
                validate(sample, context, task)
            for work in ("tools", "tools_and_reasoning"):
                sample, context = agent_sample(task, work_type=work)
                sample["required_tools"] = []
                with self.subTest(task=task, work=work), self.assertRaisesRegex(ValueError, "at least one necessary"):
                    validate(sample, context, task)

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
                lambda s: s.pop("annotations"),
                lambda s: s["annotations"][0]["evidence"][0].update(available_time="00:21"),
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

    def test_every_agent_episode_is_validated_and_merged_in_time_order(self):
        sample, context = agent_sample()
        second = json.loads(json.dumps(sample["responses"][0]).replace("00:20", "00:30")
                            .replace("00:40", "00:50").replace("00:41", "00:51").replace("00:48", "00:58"))
        second["trigger"]["episode_id"] = "event-30"
        sample["responses"].append(second)
        result = validate(sample, context)
        merged = adapt_responses(result["responses"])
        self.assertEqual([r["trigger_time"] for r in merged],
                         ["00:20", "00:30", "00:40", "00:41", "00:50", "00:51"])
        self.assertEqual([r["source_trigger_time"] for r in merged if "source_trigger_time" in r],
                         ["00:20", "00:30"])
        for mutate in (
            lambda r: r.pop("response"),
            lambda r: r.update(response="A different acknowledgement."),
            lambda r: r["messages"][1].update(timestamp="00:49"),
            lambda r: r["messages"][2].update(outcome="failed"),
        ):
            invalid = copy.deepcopy(sample)
            mutate(invalid["responses"][1])
            with self.assertRaises(ValueError):
                validate(invalid, context)
        invalid = copy.deepcopy(sample)
        envelope = json.loads(invalid["responses"][1]["messages"][1]["content"])
        envelope["actions"] = []
        invalid["responses"][1]["messages"][1]["content"] = json.dumps(envelope)
        with self.assertRaisesRegex(ValueError, "every required tool"):
            validate(invalid, context)

    def test_negative_annotations_cover_multiple_episodes_without_extra_refusals(self):
        for kind in ("no_context", "missing_tool"):
            sample, context = agent_sample(sample_type=kind)
            sample["annotations"].append(observation("event_understanding", "agent", 40))
            result = validate(sample, context)
            self.assertEqual(len(result["annotations"]), 2)
            self.assertEqual(len(result["responses"]), 1)
            sample["annotations"][1].pop("response")
            with self.assertRaises(ValueError):
                validate(sample, context)

    def test_agent_window_and_delay_use_recommended_call_time(self):
        sample, context = agent_sample()
        response = sample["responses"][0]
        response.update(trigger_time="00:22", window_end="00:25")
        response["messages"][0]["timestamp"] = "00:22"
        response["messages"][1]["timestamp"] = "00:42"
        response["messages"][2].update(timestamp="00:43", window_start="00:43", window_end="00:48")
        validate(sample, context)
        response["window_end"] = "01:41"
        with self.assertRaisesRegex(ValueError, "latest eligible call"):
            validate(sample, context)
        sample, context = agent_sample(sample_type="missing_tool")
        sample["responses"][0]["window_end"] = "00:01"
        with self.assertRaisesRegex(ValueError, "refusal window"):
            validate(sample, context)

    def test_completion_windows_are_required_and_natural_not_fixed_three_seconds(self):
        sample, context = agent_sample()
        final = sample["responses"][0]["messages"][2]
        final.update(window_end="01:00", timestamp="00:48")
        validate(sample, context)
        merged = adapt_responses(sample["responses"])
        self.assertEqual(merged[2]["window_end"], "01:00")
        self.assertEqual(merged[2]["agent_result_time"], "00:40")
        for fields in ({"timestamp": "00:40"}, {"window_start": "00:40"},
                       {"window_end": "02:01"}):
            broken = copy.deepcopy(sample)
            broken["responses"][0]["messages"][2].update(fields)
            with self.assertRaises(ValueError):
                validate(broken, context)
        final.pop("window_start")
        with self.assertRaises(ValueError):
            validate(sample, context)

    def test_negative_window_is_exactly_three_seconds_and_fits_video(self):
        sample, context = agent_sample(sample_type="no_context")
        validate(sample, context)
        for end in ["00:00", "00:02", "00:04"]:
            broken = copy.deepcopy(sample)
            broken["responses"][0]["window_end"] = end
            with self.assertRaises(ValueError):
                validate(broken, context)
        with self.assertRaises(ValueError):
            validate(sample, replace(context, duration_seconds=2.9))

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

    def test_backend_evidence_and_required_reasoning(self):
        changes = [
            ("tools", {"simulated": False}),
            ("tools", {"actions": [], "sources": []}),
            ("tools", {"actions": [{"tool": "invented", "operation": "send", "status": "completed", "evidence": "sent"}]}),
            ("tools_and_reasoning", {"reasoning": []}),
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
