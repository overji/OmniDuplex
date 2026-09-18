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
    triggered_message_notification,
    semantic_condition_alert,
    snapshot_counting,
    state_monitoring,
    target_grounding,
    triggered_realtime_query,
    triggered_multi_source_synthesis,
    triggered_knowledge_retrieval,
    triggered_content_recording,
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
            triggered_knowledge_retrieval,
            triggered_content_recording,
            triggered_message_notification,
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

    def test_uncertainties_survive_evidence_loading(self) -> None:
        self.events.write_text(json.dumps({
            "audio_events": [], "visual_events": [],
            "uncertainties": ["Unclear spoken name"],
        }), encoding="utf-8")
        self.objects.write_text(json.dumps({
            "items": [], "uncertainties": ["Possible sign"],
        }), encoding="utf-8")
        evidence = _shared.load_evidence(self.events, self.objects)
        self.assertEqual(evidence["event_uncertainties"], ["Unclear spoken name"])
        self.assertEqual(evidence["object_uncertainties"], ["Possible sign"])

    def test_all_response_prompts_enforce_temporal_contract(self) -> None:
        context = _shared.GenerationContext(
            90,
            "audio",
            "start",
            agent_delay_seconds=20,
            initial_response_delay_seconds=10,
        )
        agent_modules = (
            triggered_knowledge_retrieval,
            triggered_realtime_query,
            triggered_multi_source_synthesis,
            triggered_content_recording,
            triggered_message_notification,
        )
        shared_modules = (
            target_grounding,
            snapshot_counting,
            semantic_condition_alert,
            event_narration,
            cumulative_counting,
            deduplicated_counting,
            *agent_modules,
        )
        prompts = [
            _shared.render_prompt(
                module.PROMPT_PATH,
                context,
                agent=module in agent_modules,
            )
            for module in shared_modules
        ]
        prompts.extend(
            (
                event_alert._render_system_prompt(
                    event_alert.GenerationContext(
                        90,
                        "audio",
                        "start",
                        initial_response_delay_seconds=10,
                    )
                ),
                state_monitoring._render_system_prompt(
                    state_monitoring.GenerationContext(
                        90,
                        "start",
                        initial_response_delay_seconds=10,
                    )
                ),
            )
        )

        self.assertEqual(len(prompts), 13)
        for prompt in prompts:
            normalized = " ".join(prompt.split())
            with self.subTest(prompt=prompt.splitlines()[0]):
                self.assertIn(
                    "may use only evidence observable at or before its own "
                    "trigger_time or message timestamp; never use later video evidence",
                    normalized,
                )
                self.assertIn("must not include an MM:SS video timestamp", normalized)
                self.assertIn("observable boundary has ended", normalized)
                self.assertIn("confirmed", normalized)
                self.assertIn("discarding fractional seconds", normalized)

    def test_event_narration_waits_for_completed_updates(self) -> None:
        prompt = _shared.render_prompt(
            event_narration.PROMPT_PATH,
            _shared.GenerationContext(
                90,
                "audio",
                "start",
                initial_response_delay_seconds=10,
            ),
        )
        normalized = " ".join(prompt.split())

        self.assertIn(
            "earliest moment when the completed update is observable and confirmable",
            normalized,
        )
        self.assertIn("For a sustained event, wait until it ends", normalized)
        self.assertNotIn("Describe each update at its onset", normalized)

    def test_target_grounding_has_no_fixed_four_response_cap(self) -> None:
        prompt = _shared.render_prompt(
            target_grounding.PROMPT_PATH,
            _shared.GenerationContext(90, "vision", "start"),
        )
        normalized = " ".join(prompt.split())
        self.assertNotIn("1-4 unambiguous onsets", normalized)
        self.assertIn("every independent, unambiguous, directly verified occurrence", normalized)
        self.assertIn("each independently verified occurrence, create exactly one separate response", normalized)

        context = _shared.GenerationContext(60, "vision", "start")
        base_response = {
            "trigger_time": "00:10",
            "response": "The target is upper left.",
            "target": "exit sign",
            "location_type": "visual",
            "location": "upper left",
            "frame_region": "top-left",
            "confidence": 0.9,
            "trigger_type": "visual",
            "trigger_type_reason": "The target is visible.",
            "event_description": "The target appears.",
        }
        result = {
            "status": "ok",
            "question": "Tell me when the exit sign appears.",
            "question_time": "00:00",
            "audio_dependency": "none",
            "vision_dependency": "required",
            "responses": [base_response],
        }
        responses = []
        regions = ("top-left", "top-center", "top-right", "middle-left", "middle-center")
        for index, region in enumerate(regions, 10):
            response = copy.deepcopy(base_response)
            response.update(
                trigger_time=f"00:{index:02d}",
                target="exit sign",
                location_type="visual",
                location=region.replace("-", " "),
                frame_region=region,
                confidence=0.9,
            )
            responses.append(response)
        result["responses"] = responses
        target_grounding.validate_result(result, context)

    def test_agent_timing_waits_for_completed_trigger(self) -> None:
        context = _shared.GenerationContext(
            90,
            "audio",
            "start",
            agent_delay_seconds=20,
            initial_response_delay_seconds=10,
        )
        for module in (
            triggered_knowledge_retrieval,
            triggered_realtime_query,
            triggered_multi_source_synthesis,
            triggered_content_recording,
            triggered_message_notification,
        ):
            prompt = _shared.render_prompt(module.PROMPT_PATH, context, agent=True)
            normalized = " ".join(prompt.split())
            with self.subTest(task=module.__name__):
                self.assertIn(
                    "Before generating any acknowledgement, agent_call, or final reply",
                    normalized,
                )
                self.assertIn("must have ended and be confirmed before trigger_time", normalized)
                self.assertIn("never round or carry an event into a later bucket", normalized)

    def test_agent_prompt_contains_catalog_and_exact_delay(self) -> None:
        with (
            patch.object(_shared, "probe_video_duration", return_value=90),
            patch.object(_shared, "choose_agent_delay", return_value=31),
            patch.object(_shared, "choose_question_position", return_value="start"),
        ):
            request = triggered_knowledge_retrieval.build_request(
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

    def test_initial_response_delay_uses_ten_seconds_and_half_duration_bounds(self) -> None:
        self.assertIsNone(config.choose_initial_response_delay(19.99))
        with patch.object(config.random, "randint", return_value=10) as draw:
            self.assertEqual(config.choose_initial_response_delay(20), 10)
            draw.assert_called_once_with(10, 10)
        with patch.object(config.random, "randint", return_value=60) as draw:
            self.assertEqual(config.choose_initial_response_delay(120), 60)
            draw.assert_called_once_with(10, 60)
        with patch.object(config.random, "randint", return_value=180) as draw:
            self.assertEqual(config.choose_initial_response_delay(500), 180)
            draw.assert_called_once_with(10, 180)


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
        for key in ("bounding_box", "direction_degrees", "distance_estimate"):
            invalid = copy.deepcopy(result)
            invalid["responses"][0][key] = None
            with self.assertRaisesRegex(ValueError, "numeric locations"):
                target_grounding.validate_result(invalid, self.context)
        result["responses"][0]["frame_region"] = "outside"
        with self.assertRaisesRegex(ValueError, "frame_region"):
            target_grounding.validate_result(result, self.context)

    def test_snapshot_counting(self) -> None:
        result = copy.deepcopy(self.base)
        result["trigger_times"] = ["00:10"]
        result["responses"][0].update(
            {"count": 3, "counting_basis": "Three visible players.",
             "target_locations": ["upper left", "center", "lower right"]}
        )
        snapshot_counting.validate_result(result, self.context)
        for locations in ([[10, 20]], ["at 10, 20"], [{"box": [0, 0, 1, 1]}]):
            invalid = copy.deepcopy(result)
            invalid["responses"][0]["target_locations"] = locations
            with self.assertRaises(ValueError):
                snapshot_counting.validate_result(invalid, self.context)
        result["responses"][0]["count"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            snapshot_counting.validate_result(result, self.context)

    def test_semantic_condition(self) -> None:
        result = copy.deepcopy(self.base)
        result["semantic_condition"] = "The environment becomes chaotic"
        result["observable_criteria"] = ["Conflicting flows", "Repeated obstructions"]
        result["responses"][0].update(
            {"condition_evidence": "People moving against each other repeatedly obstruct the passage.", "confidence": 0.8}
        )
        semantic_condition_alert.validate_result(result, self.context)
        for key in ("semantic_condition", "observable_criteria"):
            invalid = copy.deepcopy(result)
            del invalid[key]
            with self.assertRaises(ValueError):
                semantic_condition_alert.validate_result(invalid, self.context)

    def test_snapshot_trigger_audit_must_match_every_response(self) -> None:
        result = copy.deepcopy(self.base)
        result["responses"][0].update(
            count=1, counting_basis="One separated target.", target_locations=["center"]
        )
        for times in ([], ["00:10", "00:20"], ["00:10", "00:10"], ["00:20", "00:10"]):
            result["trigger_times"] = times
            with self.subTest(times=times), self.assertRaises(ValueError):
                snapshot_counting.validate_result(result, self.context)

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
        dedup["target_category"] = "product names"
        dedup["counting_unit"] = "name"
        dedup["responses"][0]["new_targets"] = [
            {"target_id": "a", "name": "Alpha", "evidence": "Spoken."},
            {"target_id": "b", "name": "Beta", "evidence": "Spoken."},
        ]
        dedup["responses"][1]["new_targets"] = [
            {"target_id": "c", "name": "Gamma", "evidence": "Spoken."}
        ]
        deduplicated_counting.validate_result(dedup, context)
        for key in ("target_category", "counting_unit"):
            invalid = copy.deepcopy(dedup)
            del invalid[key]
            with self.assertRaises(ValueError):
                deduplicated_counting.validate_result(invalid, context)


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
            "extracted_parameters": [{"name": "model", "value": "X", "evidence_time": "00:10", "evidence": "Model label visible."}],
        }
        envelope = {
            "simulated": True,
            "status": "success",
            "summary": "The manual was found.",
            "details": {"query_subject": "Model X", "knowledge": "A matching manual is available."},
            "sources": [{"title": "Manual", "url_or_system": "manual-system"}],
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

    def test_valid_id_free_trace(self) -> None:
        triggered_knowledge_retrieval.validate_result(self.result, self.context)

    def test_parameters_must_have_non_future_evidence(self) -> None:
        for parameters in ([], [{"name": "model", "value": "X",
                                 "evidence_time": "00:11", "evidence": "Visible."}]):
            result = copy.deepcopy(self.result)
            function = result["responses"][0]["messages"][0]["tool_calls"][0]["function"]
            arguments = json.loads(function["arguments"])
            arguments["extracted_parameters"] = parameters
            function["arguments"] = json.dumps(arguments)
            with self.assertRaises(ValueError):
                triggered_knowledge_retrieval.validate_result(result, self.context)

    def test_workflow_success_contracts(self) -> None:
        cases = (
            (triggered_knowledge_retrieval,
             {"query_subject": "X", "knowledge": "Documented specification."},
             [{"title": "Manual", "url_or_system": "manual"}], [], "knowledge"),
            (triggered_realtime_query,
             {"query_subject": "X", "data_source": "live service",
              "queried_at": "2026-09-16T12:00Z", "valid_at": "2026-09-16T12:00Z",
              "information_type": "current", "result": "Available"},
             [{"title": "Live source", "url_or_system": "service"}], [], "valid_at"),
            (triggered_multi_source_synthesis,
             {"decision_goal": "Compare", "conclusion": "Prefer X",
              "basis": "Source comparison", "conflicts": [], "uncertainties": []},
             [{"title": "A", "url_or_system": "source-a"},
              {"title": "B", "url_or_system": "source-b"}], [], "basis"),
            (triggered_content_recording,
             {"fields": {"content": "Observed content"}, "destination": "notes",
              "record_status": "written"},
             [], [{"system": "notes", "operation": "write", "status": "completed",
                   "verification": "Stored record"}], "destination"),
            (triggered_message_notification,
             {"recipient": "Team", "content": "Confirmed decision",
              "send_time": "now", "message_status": "sent"},
             [], [{"system": "mail", "operation": "send", "status": "completed",
                   "verification": "Delivery receipt"}], "recipient"),
        )
        for module, details, sources, actions, required_key in cases:
            with self.subTest(workflow=module.__name__):
                result = copy.deepcopy(self.result)
                tool = result["responses"][0]["messages"][1]
                envelope = json.loads(tool["content"])
                envelope.update(details=details, sources=sources, actions=actions)
                tool["content"] = json.dumps(envelope)
                module.validate_result(result, self.context)
                del details[required_key]
                tool["content"] = json.dumps(envelope)
                with self.assertRaises(ValueError):
                    module.validate_result(result, self.context)

    def test_duplicate_sources_and_false_sent_status_are_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        tool = result["responses"][0]["messages"][1]
        envelope = json.loads(tool["content"])
        envelope["sources"] *= 2
        envelope["details"] = {
            "decision_goal": "Compare", "conclusion": "Unclear", "basis": "Same source",
            "conflicts": [], "uncertainties": [],
        }
        tool["content"] = json.dumps(envelope)
        with self.assertRaises(ValueError):
            triggered_multi_source_synthesis.validate_result(result, self.context)
        envelope.update(status="needs_confirmation", actions=[], details={
            "message_status": "sent",
        })
        tool["content"] = json.dumps(envelope)
        with self.assertRaises(ValueError):
            triggered_message_notification.validate_result(result, self.context)
        envelope.update(status="success", details={
            "message_status": "draft", "recipient": "Team", "content": "Draft",
            "send_time": "after confirmation",
        })
        tool["content"] = json.dumps(envelope)
        triggered_message_notification.validate_result(result, self.context)

    def test_ids_order_and_multiple_calls_are_rejected(self) -> None:
        for mutation in ("call_id", "result_id", "order", "multiple"):
            result = copy.deepcopy(self.result)
            messages = result["responses"][0]["messages"]
            if mutation == "call_id":
                messages[0]["tool_calls"][0]["id"] = "old_id"
            elif mutation == "result_id":
                messages[1]["tool_call_id"] = "old_id"
            elif mutation == "order":
                messages[1], messages[2] = messages[2], messages[1]
            else:
                messages[0]["tool_calls"] *= 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                triggered_knowledge_retrieval.validate_result(result, self.context)

    def test_call_arguments_must_be_json(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"][0]["messages"][0]["tool_calls"][0]["function"][
            "arguments"
        ] = "not json"
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            triggered_knowledge_retrieval.validate_result(result, self.context)

    def test_result_time_must_match_delay_and_video(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"][0]["messages"][1]["timestamp"] = "00:31"
        with self.assertRaisesRegex(ValueError, "sampled delay"):
            triggered_knowledge_retrieval.validate_result(result, self.context)

        short_context = _shared.GenerationContext(14, "vision", "start", None)
        with self.assertRaisesRegex(ValueError, "shorter"):
            triggered_knowledge_retrieval.validate_result(result, short_context)
        self.assertEqual(
            triggered_knowledge_retrieval.validate_result(
                {"status": "skip", "reason": "Too short."}, short_context
            ),
            {"status": "skip", "reason": "Too short."},
        )


if __name__ == "__main__":
    unittest.main()
