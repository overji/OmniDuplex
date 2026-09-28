from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from data_generation.src import config
from data_generation.src.cli import main
from data_generation.src.common import parse_duration
from data_generation.src.registry import get_task, task_names
from data_generation.src.tasks import event_alert


class ConfigTests(unittest.TestCase):
    def test_probability_boundaries_and_split(self) -> None:
        cases = (
            (1.0, 0.999999, "audio"),
            (0.0, 0.0, "vision"),
            (0.5, 0.499999, "audio"),
            (0.5, 0.5, "vision"),
        )
        for probability, draw, expected in cases:
            with self.subTest(probability=probability, draw=draw):
                with patch.object(config.random, "random", return_value=draw):
                    self.assertEqual(
                        config.choose_preferred_modality(probability), expected
                    )

    def test_start_position_sampling_and_short_video_fallback(self) -> None:
        with patch.object(config.random, "random", return_value=0.49):
            self.assertEqual(config.choose_question_position(30, 0.5), "start")
        with patch.object(config.random, "random", return_value=0.5):
            self.assertEqual(config.choose_question_position(30, 0.5), "non_start")
        with patch.object(config.random, "random") as draw:
            self.assertEqual(config.choose_question_position(0.9, 0.0), "start")
            draw.assert_not_called()

    def test_invalid_probability(self) -> None:
        for value in (-0.01, 1.01, True, "0.5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    config.validate_probability("TEST", value)  # type: ignore[arg-type]

    def test_duration_supports_long_mmss_and_rejects_invalid_values(self) -> None:
        self.assertEqual(parse_duration("75:00"), 4500)
        self.assertEqual(parse_duration("01:15:00"), 4500)
        for value in ("01:60:00", "01:00:60", float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_duration(value)


class PromptTests(unittest.TestCase):
    def _source_file(self, duration: str = "00:01:00.000"):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "input.json"
        path.write_text(
            json.dumps(
                {
                    "video_summary": {
                        "total_duration": duration,
                        "main_content_summary": "must not enter the user prompt",
                    },
                    "audio_events": [{"event_id": "A1", "start_seconds": 5}],
                    "visual_events": [{"event_id": "V1", "start_seconds": 10}],
                }
            ),
            encoding="utf-8",
        )
        return temporary, path

    def test_dynamic_controls_are_in_system_prompt_only(self) -> None:
        temporary, path = self._source_file()
        self.addCleanup(temporary.cleanup)
        with (
            patch.object(event_alert, "choose_preferred_modality", return_value="audio"),
            patch.object(event_alert, "choose_question_position", return_value="non_start"),
            patch.object(event_alert, "probe_video_duration", return_value=60),
        ):
            request = event_alert.build_request(path, Path("video.mp4"))

        self.assertIn("video duration is 01:00", request.system_prompt)
        self.assertIn("audio_dependency is `required`", request.system_prompt)
        self.assertIn("strictly after 00:00", request.system_prompt)
        self.assertNotIn("{{", request.system_prompt)
        user_evidence = json.loads(request.user_prompt)
        self.assertEqual(set(user_evidence), {"audio_events", "visual_events"})
        self.assertNotIn("Generation", request.user_prompt)
        self.assertNotIn("main_content_summary", request.user_prompt)

    def test_start_policy_is_rendered(self) -> None:
        context = event_alert.GenerationContext(60, "vision", "start")
        prompt = event_alert._render_system_prompt(context)
        self.assertIn("question_time MUST be 00:00", prompt)
        self.assertIn("vision_dependency is `required`", prompt)

    def test_prompt_prioritizes_video_and_audits_occurrence_count(self) -> None:
        context = event_alert.GenerationContext(60, "vision", "start")
        prompt = event_alert._render_system_prompt(context)
        normalized = " ".join(prompt.split())

        self.assertIn("The original video is the primary source of truth", normalized)
        self.assertIn("only as fallible references", normalized)
        self.assertIn("Exclude a referenced occurrence", normalized)
        self.assertIn("event descriptions omit it", normalized)
        self.assertIn("one uninterrupted occurrence only once", normalized)
        self.assertIn("clearly ended and began again", normalized)
        self.assertIn("total number of timestamps across", normalized)
        self.assertNotIn("Use them as the sole evidence", normalized)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = event_alert.GenerationContext(60, "audio", "non_start")
        self.result = {
            "status": "ok",
            "question": "Let me know whenever the bell rings.",
            "question_time": "00:15",
            "audio_dependency": "required",
            "vision_dependency": "none",
            "responses": [
                self._response("00:15", ["00:05", "00:10"]),
                self._response("00:20", ["00:20"]),
                self._response("00:40", ["00:40"]),
            ],
        }

    @staticmethod
    def _response(trigger_time: str, event_times: list[str]) -> dict[str, object]:
        return {
            "trigger_time": trigger_time,
            "event_times": event_times,
            "response": "The bell has rung.",
            "trigger_type": "sound",
            "trigger_type_reason": "A discrete bell sound is annotated.",
            "event_description": "The bell rings.",
        }

    def test_historical_summary_and_future_responses(self) -> None:
        validated = event_alert.validate_result(copy.deepcopy(self.result), self.context)
        self.assertEqual(len(validated["responses"]), 3)

    def test_only_historical_or_only_future_is_valid(self) -> None:
        historical = copy.deepcopy(self.result)
        historical["responses"] = historical["responses"][:1]
        event_alert.validate_result(historical, self.context)

        future = copy.deepcopy(self.result)
        future["responses"] = future["responses"][1:]
        event_alert.validate_result(future, self.context)

    def test_start_sample_accepts_event_at_zero(self) -> None:
        result = copy.deepcopy(self.result)
        result["question_time"] = "00:00"
        result["responses"] = [self._response("00:00", ["00:00"])]
        context = event_alert.GenerationContext(60, "audio", "start")
        event_alert.validate_result(result, context)

    def test_skip_is_valid(self) -> None:
        self.assertEqual(
            event_alert.validate_result(
                {"status": "skip", "reason": "No discrete event."}, self.context
            ),
            {"status": "skip", "reason": "No discrete event."},
        )

    def test_question_position_must_match_sampled_mode(self) -> None:
        result = copy.deepcopy(self.result)
        result["question_time"] = "00:00"
        with self.assertRaisesRegex(ValueError, "non-start"):
            event_alert.validate_result(result, self.context)

    def test_historical_summary_bypasses_silent_interval(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"][0]["trigger_time"] = "00:16"
        context = replace(self.context, initial_response_delay_seconds=10)
        with self.assertRaisesRegex(ValueError, "question_time"):
            event_alert.validate_result(result, context)

        result["responses"][0]["trigger_time"] = "00:15"
        event_alert.validate_result(result, context)

    def test_future_trigger_must_not_precede_event_time(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"][1]["trigger_time"] = "00:21"
        event_alert.validate_result(result, self.context)

        result["responses"][1]["trigger_time"] = "00:19"
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            event_alert.validate_result(result, self.context)

    def test_nonhistorical_response_still_observes_silent_interval(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"] = [result["responses"][1]]
        result["responses"][0]["trigger_time"] = "00:24"
        context = replace(self.context, initial_response_delay_seconds=10)
        with self.assertRaisesRegex(ValueError, "sampled silent interval"):
            event_alert.validate_result(result, context)

    def test_response_must_not_contain_video_timestamp(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"][0]["response"] = "The event occurred at 00:12."
        with self.assertRaisesRegex(ValueError, "video timestamp"):
            event_alert.validate_result(result, self.context)

    def test_response_cannot_mix_past_and_future(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"] = [self._response("00:15", ["00:10", "00:20"])]
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            event_alert.validate_result(result, self.context)

    def test_non_start_question_cannot_equal_event_onset(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"] = [self._response("00:15", ["00:15"])]
        with self.assertRaisesRegex(ValueError, "cannot equal"):
            event_alert.validate_result(result, self.context)

    def test_historical_occurrences_must_share_one_response(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"] = [
            self._response("00:15", ["00:05"]),
            self._response("00:15", ["00:10"]),
        ]
        with self.assertRaises(ValueError):
            event_alert.validate_result(result, self.context)

    def test_exactly_one_dependency_is_required(self) -> None:
        for dependencies in (("required", "required"), ("helpful", "none")):
            with self.subTest(dependencies=dependencies):
                result = copy.deepcopy(self.result)
                result["audio_dependency"], result["vision_dependency"] = dependencies
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    event_alert.validate_result(result, self.context)

    def test_trigger_type_must_include_required_modality(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"][0]["trigger_type"] = "visual"
        with self.assertRaisesRegex(ValueError, "required audio"):
            event_alert.validate_result(result, self.context)


class EntrypointTests(unittest.TestCase):
    def test_registry(self) -> None:
        self.assertEqual(
            task_names(),
            (
                "cumulative_counting",
                "deduplicated_counting",
                "event_alert",
                "event_narration",
                "route",
                "semantic_condition_alert",
                "snapshot_counting",
                "state_monitoring",
                "target_grounding",
                "triggered_complex_reasoning",
                "triggered_content_recording",
                "triggered_knowledge_retrieval",
                "triggered_message_notification",
                "triggered_realtime_query",
            ),
        )
        self.assertIs(get_task("event_alert"), event_alert)
        with self.assertRaisesRegex(ValueError, "unknown task"):
            get_task("missing")

    def test_print_prompt_entrypoint(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "events.json"
        source.write_text(
            json.dumps({"audio_events": [], "visual_events": []}),
            encoding="utf-8",
        )
        video = Path(__file__).resolve().parents[1] / "videos" / "test.mp4"
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "--task",
                    "event_alert",
                    "--video",
                    str(video),
                    str(source),
                    "--print-prompt",
                ]
            )
        self.assertEqual(status, 0)
        rendered = output.getvalue()
        self.assertIn("The video duration is 02:00", rendered)
        self.assertIn(f"[video_url: {video.resolve()}]", rendered)
        self.assertIn('"audio_events"', rendered)
        self.assertIn('"visual_events"', rendered)
        self.assertNotIn('"video_metadata"', rendered)


if __name__ == "__main__":
    unittest.main()
