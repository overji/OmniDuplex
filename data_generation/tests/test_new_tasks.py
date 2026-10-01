import copy
import unittest
from unittest.mock import patch
from data_generation.src import config
from data_generation.src.tasks import (
    _shared, cumulative_counting, deduplicated_counting, event_narration,
    semantic_condition_alert, snapshot_counting, target_grounding,
)


class PreservedPromptTests(unittest.TestCase):
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



class DelayTests(unittest.TestCase):
    def test_delay_boundaries(self) -> None:
        self.assertIsNone(config.choose_agent_delay(9.99))
        with patch.object(config.random, "randint", return_value=15) as draw:
            self.assertEqual(config.choose_agent_delay(30), 15)
            draw.assert_called_once_with(5, 15)
        with patch.object(config.random, "randint", return_value=25) as draw:
            self.assertEqual(config.choose_agent_delay(100), 25)
            draw.assert_called_once_with(10, 50)

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
