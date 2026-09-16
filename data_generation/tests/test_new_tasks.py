from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_generation.src import config
from data_generation.src.cli import main, parse_args
from data_generation.src.tasks import (
    _shared,
    cumulative_counting,
    deduplicated_counting,
    event_alert,
    event_narration,
    meeting_understanding,
    semantic_condition_alert,
    snapshot_counting,
    target_grounding,
    triggered_information_query,
    triggered_tool_action,
)
from data_generation.src.tasks.event_alert import GenerationContext as EventContext
from data_generation.src.tasks.event_alert import GenerationRequest as EventRequest


class RequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.events = root / "events.json"
        self.objects = root / "objects.json"
        self.events.write_text(
            json.dumps(
                {
                    "audio_events": [{"event_id": "A1"}],
                    "visual_events": [{"event_id": "V1"}],
                    "video_summary": {"must_not": "be copied"},
                }
            ),
            encoding="utf-8",
        )
        self.objects.write_text(
            json.dumps({"items": [{"item_id": "I1", "name": "bell"}]}),
            encoding="utf-8",
        )

    def test_all_new_prompts_render_and_objects_are_optional(self) -> None:
        modules = (
            target_grounding,
            snapshot_counting,
            semantic_condition_alert,
            event_narration,
            cumulative_counting,
            deduplicated_counting,
            triggered_information_query,
            triggered_tool_action,
            meeting_understanding,
        )
        for module in modules:
            with (
                self.subTest(module=module.__name__),
                patch.object(_shared, "probe_video_duration", return_value=90),
                patch.object(_shared, "choose_preferred_modality", return_value="audio"),
                patch.object(_shared, "choose_question_position", return_value="start"),
                patch.object(_shared, "choose_agent_delay", return_value=20),
            ):
                request = module.build_request(
                    self.events, Path("video.mp4"), objects_path=self.objects
                )
                self.assertNotIn("{{", request.system_prompt)
                self.assertIn("01:30", request.system_prompt)
                evidence = json.loads(request.user_prompt)
                self.assertEqual(
                    set(evidence), {"audio_events", "visual_events", "items"}
                )
                self.assertNotIn("video_summary", evidence)

        with patch.object(_shared, "probe_video_duration", return_value=90):
            request = target_grounding.build_request(
                self.events, Path("video.mp4")
            )
        self.assertNotIn("items", json.loads(request.user_prompt))

    def test_objects_input_is_validated(self) -> None:
        self.objects.write_text('{"items":["bad"]}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "items array"):
            target_grounding.build_request(
                self.events, Path("video.mp4"), objects_path=self.objects
            )

    def test_agent_prompt_contains_catalog_and_exact_delay(self) -> None:
        with (
            patch.object(_shared, "probe_video_duration", return_value=90),
            patch.object(_shared, "choose_agent_delay", return_value=31),
            patch.object(_shared, "choose_question_position", return_value="start"),
        ):
            request = triggered_information_query.build_request(
                self.events, Path("video.mp4")
            )
        self.assertIn("web_search", request.system_prompt)
        self.assertIn("permission_check", request.system_prompt)
        self.assertIn("exactly 31 seconds", request.system_prompt)
        self.assertEqual(request.context.agent_delay_seconds, 31)


class DelayTests(unittest.TestCase):
    def test_delay_boundaries(self) -> None:
        self.assertIsNone(config.choose_agent_delay(14.99))
        with patch.object(config.random, "randint", return_value=15) as draw:
            self.assertEqual(config.choose_agent_delay(15), 15)
            draw.assert_called_once_with(15, 15)
        with patch.object(config.random, "randint", return_value=45) as draw:
            self.assertEqual(config.choose_agent_delay(100), 45)
            draw.assert_called_once_with(15, 45)


class CliTests(unittest.TestCase):
    def test_objects_option_is_parsed(self) -> None:
        args = parse_args(
            [
                "--task",
                "target_grounding",
                "--video",
                "video.mp4",
                "events.json",
                "--objects",
                "objects.json",
            ]
        )
        self.assertEqual(args.objects, Path("objects.json"))

    def test_successful_output_gets_video_filename(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "result.json"
        request = EventRequest(
            system_prompt="system",
            user_prompt="user",
            context=EventContext(60, "audio", "start"),
            video_path=Path("/private/path/source.mp4"),
        )
        validated = {
            "status": "ok",
            "question": "Watch for the bell.",
            "question_time": "00:00",
            "responses": [],
        }
        with (
            patch.object(event_alert, "build_request", return_value=request),
            patch.object(event_alert, "validate_result", return_value=validated),
            patch("data_generation.src.cli.chat_completion", return_value="{}"),
        ):
            status = main(
                [
                    "--task",
                    "event_alert",
                    "--video",
                    "ignored.mp4",
                    "ignored.json",
                    "--api-key",
                    "key",
                    "--model",
                    "model",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(status, 0)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["video"], "source.mp4")
        self.assertNotIn("private", result["video"])


class DirectValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = _shared.GenerationContext(60, "vision", "start")
        self.base = {
            "status": "ok",
            "question": "Tell me when it happens.",
            "question_time": "00:00",
            "audio_dependency": "none",
            "vision_dependency": "required",
            "responses": [
                {
                    "trigger_time": "00:10",
                    "response": "It is happening now.",
                    "trigger_type": "visual",
                    "trigger_type_reason": "The change is visible.",
                    "event_description": "A visible event begins.",
                }
            ],
        }

    def test_target_grounding(self) -> None:
        result = copy.deepcopy(self.base)
        result["responses"][0].update(
            {
                "target": "exit sign",
                "location_type": "visual",
                "location": "upper right",
                "frame_region": "top-right",
                "confidence": 0.9,
            }
        )
        target_grounding.validate_result(result, self.context)
        result["responses"][0]["frame_region"] = "outside"
        with self.assertRaisesRegex(ValueError, "frame_region"):
            target_grounding.validate_result(result, self.context)

    def test_snapshot_counting(self) -> None:
        result = copy.deepcopy(self.base)
        result["responses"][0].update(
            {"count": 3, "counting_basis": "Three visible players."}
        )
        snapshot_counting.validate_result(result, self.context)
        result["responses"][0]["count"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            snapshot_counting.validate_result(result, self.context)

    def test_semantic_condition(self) -> None:
        result = copy.deepcopy(self.base)
        result["responses"][0].update(
            {"condition_evidence": "The same marked car returns.", "confidence": 0.8}
        )
        semantic_condition_alert.validate_result(result, self.context)

    def test_narration_requires_three_updates(self) -> None:
        result = copy.deepcopy(self.base)
        result["responses"] = []
        for second in (10, 20, 30):
            response = copy.deepcopy(self.base["responses"][0])
            response["trigger_time"] = f"00:{second:02d}"
            response["observed_facts"] = ["A visible state changes."]
            response["interpretation"] = None
            result["responses"].append(response)
        event_narration.validate_result(result, self.context)

    def test_cumulative_and_deduplicated_history(self) -> None:
        context = _shared.GenerationContext(60, "audio", "non_start")
        result = {
            "status": "ok",
            "question": "Keep count.",
            "question_time": "00:10",
            "audio_dependency": "required",
            "vision_dependency": "none",
            "responses": [
                {
                    "trigger_time": "00:10",
                    "event_times": ["00:02", "00:06"],
                    "response": "The running count is two.",
                    "count": 2,
                    "trigger_type": "sound",
                    "trigger_type_reason": "Two bells were audible.",
                    "event_description": "Historical bell occurrences.",
                },
                {
                    "trigger_time": "00:20",
                    "event_times": ["00:20"],
                    "response": "The running count is three.",
                    "count": 3,
                    "trigger_type": "sound",
                    "trigger_type_reason": "Another bell is audible.",
                    "event_description": "A bell rings.",
                },
            ],
        }
        cumulative_counting.validate_result(copy.deepcopy(result), context)
        dedup = copy.deepcopy(result)
        dedup["responses"][0]["new_targets"] = [
            {"target_id": "a", "name": "Alpha", "evidence": "Spoken."},
            {"target_id": "b", "name": "Beta", "evidence": "Spoken."},
        ]
        dedup["responses"][1]["new_targets"] = [
            {"target_id": "c", "name": "Gamma", "evidence": "Spoken."}
        ]
        deduplicated_counting.validate_result(dedup, context)


class AgentValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = _shared.GenerationContext(60, "vision", "start", 20)
        arguments = {
            "instruction": "Find the displayed device manual.",
            "observed_context": "At 00:10 model X is visible.",
            "user_goal": "Retrieve the manual.",
            "constraints_and_authorization": "Read-only research is authorized.",
            "suggested_capabilities": ["web_search", "document_search"],
            "expected_output": "A sourced manual summary.",
        }
        envelope = {
            "simulated": True,
            "status": "success",
            "summary": "The manual was found.",
            "details": "A matching manual is available.",
            "sources": [{"title": "Manual"}],
            "actions": [],
            "artifacts": [],
        }
        self.result = {
            "status": "ok",
            "question": "When model X appears, find its manual.",
            "question_time": "00:00",
            "audio_dependency": "none",
            "vision_dependency": "required",
            "responses": [
                {
                    "trigger_time": "00:10",
                    "trigger_type": "visual",
                    "trigger_type_reason": "The model label becomes visible.",
                    "event_description": "Model X appears.",
                    "messages": [
                        {
                            "timestamp": "00:10",
                            "role": "assistant",
                            "content": "I see it; I'm checking now.",
                            "tool_calls": [
                                {
                                    "id": "call_agent_1",
                                    "type": "function",
                                    "function": {
                                        "name": "agent_call",
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                        {
                            "timestamp": "00:30",
                            "role": "tool",
                            "tool_call_id": "call_agent_1",
                            "name": "agent_call",
                            "content": json.dumps(envelope),
                        },
                        {
                            "timestamp": "00:30",
                            "role": "assistant",
                            "content": "I found the matching manual.",
                        },
                    ],
                }
            ],
        }

    def test_valid_openai_style_trace(self) -> None:
        triggered_information_query.validate_result(self.result, self.context)

    def test_call_arguments_must_be_json(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"][0]["messages"][0]["tool_calls"][0]["function"][
            "arguments"
        ] = "not json"
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            triggered_information_query.validate_result(result, self.context)

    def test_result_time_must_match_delay_and_video(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"][0]["messages"][1]["timestamp"] = "00:31"
        with self.assertRaisesRegex(ValueError, "sampled delay"):
            triggered_information_query.validate_result(result, self.context)

        short_context = _shared.GenerationContext(14, "vision", "start", None)
        with self.assertRaisesRegex(ValueError, "shorter"):
            triggered_information_query.validate_result(result, short_context)
        self.assertEqual(
            triggered_information_query.validate_result(
                {"status": "skip", "reason": "Too short."}, short_context
            ),
            {"status": "skip", "reason": "Too short."},
        )


if __name__ == "__main__":
    unittest.main()
