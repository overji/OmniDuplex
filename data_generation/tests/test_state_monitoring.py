from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tasks import state_monitoring


class PromptTests(unittest.TestCase):
    def _input_file(self, items: object) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "objects.json"
        path.write_text(json.dumps({"items": items}), encoding="utf-8")
        return temporary, path

    def test_request_uses_only_items_and_measured_video_duration(self) -> None:
        temporary, path = self._input_file(
            [{"item_id": "item_001", "name": "Main subject", "is_main_item": True}]
        )
        self.addCleanup(temporary.cleanup)
        with (
            patch.object(state_monitoring, "probe_video_duration", return_value=65.2),
            patch.object(
                state_monitoring,
                "choose_question_position",
                return_value="non_start",
            ),
        ):
            request = state_monitoring.build_request(path, Path("video.mp4"))

        self.assertEqual(set(json.loads(request.user_prompt)), {"items"})
        self.assertIn("video duration is 01:05 (65 seconds)", request.system_prompt)
        self.assertIn("strictly after 00:00", request.system_prompt)
        self.assertIn("Only monitor transitions strictly after", request.system_prompt)
        self.assertNotIn("{{", request.system_prompt)

    def test_invalid_items_are_rejected(self) -> None:
        for items in (None, [], ["not-an-object"]):
            with self.subTest(items=items):
                temporary, path = self._input_file(items)
                try:
                    with self.assertRaises(ValueError):
                        state_monitoring._load_evidence(path)
                finally:
                    temporary.cleanup()

    def test_start_policy_is_rendered(self) -> None:
        prompt = state_monitoring._render_system_prompt(
            state_monitoring.GenerationContext(60, "start")
        )
        self.assertIn("question_time MUST be 00:00", prompt)
        self.assertIn("Mandatory audio-first scan", prompt)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = state_monitoring.GenerationContext(60, "non_start")
        self.result = {
            "status": "ok",
            "audio_scan": "Checked speakers and music; speaker turns are discrete.",
            "question": "Tell me whenever the speaker changes.",
            "question_time": "00:10",
            "audio_dependency": "required",
            "responses": [
                self._response("00:20", "The speaker changes from Alice to Bob."),
                self._response("00:40", "The speaker changes from Bob to Alice."),
            ],
        }

    @staticmethod
    def _response(trigger_time: str, response: str) -> dict[str, str]:
        return {
            "trigger_time": trigger_time,
            "response": response,
            "trigger_type": "speech",
            "trigger_type_reason": "A different visible speaker begins talking.",
            "event_description": "The active speaker changes.",
        }

    def test_valid_result(self) -> None:
        self.assertEqual(
            state_monitoring.validate_result(copy.deepcopy(self.result), self.context),
            self.result,
        )

    def test_skip_is_valid(self) -> None:
        self.assertEqual(
            state_monitoring.validate_result(
                {"status": "skip", "reason": "No countable dimension."},
                self.context,
            ),
            {"status": "skip", "reason": "No countable dimension."},
        )

    def test_requires_two_to_five_transitions(self) -> None:
        for count in (1, 6):
            result = copy.deepcopy(self.result)
            result["responses"] = [
                self._response(
                    f"00:{20 + index * 5:02d}",
                    f"The state changes from value{index} to value{index + 1}.",
                )
                for index in range(count)
            ]
            with self.assertRaisesRegex(ValueError, "2-5"):
                state_monitoring.validate_result(result, self.context)

    def test_transition_must_be_after_question_and_chronological(self) -> None:
        for times in (("00:10", "00:40"), ("00:40", "00:20")):
            result = copy.deepcopy(self.result)
            for response, trigger_time in zip(result["responses"], times):
                response["trigger_time"] = trigger_time
            with self.assertRaises(ValueError):
                state_monitoring.validate_result(result, self.context)

    def test_response_requires_from_to_and_word_limit(self) -> None:
        cases = (
            "The speaker changes.",
            "The speaker changes from Alice to Bob with many unnecessary words that make this response far too long today.",
        )
        for text in cases:
            result = copy.deepcopy(self.result)
            result["responses"][0]["response"] = text
            with self.assertRaises(ValueError):
                state_monitoring.validate_result(result, self.context)

    def test_audio_dependency_must_match_trigger_type(self) -> None:
        result = copy.deepcopy(self.result)
        result["responses"][0]["trigger_type"] = "visual"
        with self.assertRaisesRegex(ValueError, "required audio"):
            state_monitoring.validate_result(result, self.context)

        visual = copy.deepcopy(self.result)
        visual["audio_dependency"] = "none"
        for response in visual["responses"]:
            response["trigger_type"] = "visual"
        state_monitoring.validate_result(visual, self.context)

    def test_question_position_must_match_context(self) -> None:
        result = copy.deepcopy(self.result)
        result["question_time"] = "00:00"
        with self.assertRaisesRegex(ValueError, "non-start"):
            state_monitoring.validate_result(result, self.context)


if __name__ == "__main__":
    unittest.main()
