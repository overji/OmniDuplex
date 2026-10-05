import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from data_generation.src.cli import generate as cli
from data_generation.src.application import generation, requests
from data_generation.src.domain.catalog import TASK_CATALOG, ABILITIES
from data_generation.src.domain.common import SAMPLE_SCHEMA
from data_generation.src.application.merge_qa import build_bundle
from data_generation.src.application.registry import TASKS, get_task
from data_generation.src.domain.tasks import interaction
from data_generation.tests.fixtures import caption, task_sample, observation


class MatrixTests(unittest.TestCase):
    def test_all_36_render_validate_and_merge(self):
        self.assertEqual(len(TASK_CATALOG), 36)
        self.assertEqual(len(TASKS), 37)
        with tempfile.TemporaryDirectory() as directory, patch.object(requests, "probe_video_duration", return_value=120):
            folder = Path(directory) / "video"
            folder.mkdir()
            path = folder / "caption.json"
            path.write_text(json.dumps(caption()))
            for name in TASK_CATALOG:
                with self.subTest(task=name):
                    task = get_task(name)
                    request = requests.build_task_request(name, path, Path("video.mp4"))
                    self.assertEqual(json.loads(request.user_prompt)["caption"], caption())
                    self.assertIn(name, request.system_prompt)
                    self.assertNotIn("{{", request.system_prompt)
                    self.assertEqual(request.context.initial_response_delay_seconds, 0)
                    result, context = task_sample(name)
                    result = task.validate_result(result, context)
                    result.update(schema=SAMPLE_SCHEMA, task=name, video="video.mp4", generation_context=asdict(context))
                    (folder / f"{name}.json").write_text(json.dumps(result))
            self.assertEqual(len(build_bundle(Path(directory), Path(directory))), 36)
        for old in ("event_agent", "event_alert", "target_grounding", "cumulative_counting"):
            with self.assertRaises(ValueError):
                get_task(old)

    def test_invalid_evidence_and_metadata_all_abilities(self):
        for ability in ABILITIES:
            task = get_task(f"{ability}_pr")
            sample, context = task_sample(task.name)
            for mutation in (
                lambda s: s["responses"][0]["evidence"][0].update(available_time="00:21"),
                lambda s: s["responses"][0]["evidence"][0].update(time="00:21"),
                lambda s: s["responses"][0]["trigger"].update(confirmed_time="00:19"),
                lambda s: s["responses"][0]["trigger"].update(satisfied=False),
                lambda s: s.update(ability="invented"),
                lambda s: s.update(responses=[]),
            ):
                bad = copy.deepcopy(sample)
                mutation(bad)
                with self.subTest(ability=ability), self.assertRaises(ValueError):
                    task.validate_result(bad, context)

    def test_actual_response_required_for_every_task(self):
        for name in TASK_CATALOG:
            sample, context = task_sample(name)
            self.assertNotIn("event_description", sample["responses"][0])
            get_task(name).validate_result(sample, context)
            for value in (None, "", "   "):
                invalid = copy.deepcopy(sample)
                item = invalid["responses"][0]
                item["event_description"] = "A visible event occurs."
                if value is None:
                    item.pop("response")
                else:
                    item["response"] = value
                with self.subTest(task=name, value=value), self.assertRaises(ValueError):
                    get_task(name).validate_result(invalid, context)

    def test_rtd_windows_and_minimum_updates(self):
        task = get_task("attribute_perception_rtd")
        sample, context = task_sample(task.name)
        for change in (lambda s: s["responses"].pop(),
                       lambda s: s["responses"][1].update(window_start="00:19"),
                       lambda s: s["responses"][1].update(response=s["responses"][0]["response"]),
                       lambda s: s["responses"][0].update(window_end="00:19")):
            bad = copy.deepcopy(sample)
            change(bad)
            with self.assertRaises(ValueError):
                task.validate_result(bad, context)
        with self.assertRaises(ValueError):
            task.validate_result(sample, replace(context, initial_response_delay_seconds=10))

    def test_trigger_windows_allow_interior_recommended_time(self):
        sample, context = task_sample("event_understanding_pr")
        item = sample["responses"][0]
        item.update(trigger_time="00:22", window_end="00:25")
        get_task("event_understanding_pr").validate_result(sample, context)
        for change in ({"window_start": "00:23"}, {"window_end": "00:21"},
                       {"window_start": "00:19"}, {"window_end": "02:01"}):
            invalid = copy.deepcopy(sample)
            invalid["responses"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                get_task("event_understanding_pr").validate_result(invalid, context)
        invalid = copy.deepcopy(sample)
        invalid["responses"][0]["trigger"]["confirmed_time"] = "00:22"
        with self.assertRaises(ValueError):
            get_task("event_understanding_pr").validate_result(invalid, context)

    def test_distinct_rtd_updates_can_have_overlapping_valid_windows(self):
        sample, context = task_sample("attribute_perception_rtd")
        sample["responses"][0]["window_end"] = "00:45"
        sample["responses"][1]["window_end"] = "00:50"
        get_task("attribute_perception_rtd").validate_result(sample, context)

    def test_repeated_triggers_need_reset_and_unique_episodes(self):
        task = get_task("object_perception_pr")
        sample, context = task_sample(task.name)
        sample["responses"].append(observation("object_perception", "pr", 40))
        for item in sample["responses"]:
            item["trigger"].update(kind="target", target="parcel", attributes_match=True)
        with self.assertRaises(ValueError):
            task.validate_result(sample, context)
        sample["responses"][1]["trigger"]["reset_time"] = "00:30"
        task.validate_result(sample, context)
        sample["responses"][1]["trigger"]["episode_id"] = "event-20"
        with self.assertRaises(ValueError):
            task.validate_result(sample, context)

    def test_grounding_future_span_and_counting(self):
        task = get_task("grounding_pr")
        sample, context = task_sample(task.name)
        details = sample["responses"][0]["details"]
        details.update(localization="span", start="00:10", end="00:20")
        task.validate_result(sample, context)
        details["end"] = "00:21"
        with self.assertRaises(ValueError):
            task.validate_result(sample, context)
        for name in ("object_counting_rtd", "action_counting_rtd"):
            sample, context = task_sample(name)
            sample["responses"][1]["details"]["count"] += 1
            with self.assertRaises(ValueError):
                get_task(name).validate_result(sample, context)
        sample, context = task_sample("object_counting_pr")
        sample["responses"][0]["trigger"].update(kind="count", previous_count=1, count=2, threshold=2, operator="ge")
        get_task("object_counting_pr").validate_result(sample, context)
        sample["responses"][0]["trigger"]["previous_count"] = 2
        with self.assertRaises(ValueError):
            get_task("object_counting_pr").validate_result(sample, context)

    def test_visual_and_inference_contracts(self):
        for name in ("text_rich_understanding_pr", "object_counting_pr", "spatial_understanding_pr"):
            sample, context = task_sample(name)
            sample.update(audio_dependency="required", vision_dependency="none")
            sample["responses"][0]["trigger_type"] = "speech"
            with self.assertRaises(ValueError):
                get_task(name).validate_result(sample, context)
        for ability in ("emotion_recognition", "causal_reasoning", "prospective_reasoning"):
            sample, context = task_sample(f"{ability}_pr")
            sample["responses"][0]["details"]["claim_type"] = "fact"
            with self.assertRaises(ValueError):
                get_task(f"{ability}_pr").validate_result(sample, context)

    def test_cli_dry_run_and_generation(self):
        name = "attribute_perception_rtd"
        sample, context = task_sample(name)
        with tempfile.TemporaryDirectory() as directory, patch.object(requests, "probe_video_duration", return_value=120), patch.object(interaction.config, "choose_question_position", return_value="start"):
            root = Path(directory)
            source = root / "caption.json"
            source.write_text(json.dumps(caption()))
            args = ["--task", name, "--video", "video.mp4", str(source)]
            with patch.object(generation, "chat_completion", return_value=json.dumps(sample)) as call, redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args + ["--print-prompt"]), 0)
                call.assert_not_called()
                self.assertEqual(cli.main(args + ["--api-key", "test", "--model", "test", "--output", str(root / "sample.json")]), 0)
                call.assert_called_once()
            self.assertEqual(json.loads((root / "sample.json").read_text())["schema"], SAMPLE_SCHEMA)

    def test_compound_condition_and_threshold_directions(self):
        task = get_task("event_understanding_pr")
        sample, context = task_sample(task.name)
        trigger = sample["responses"][0]["trigger"]
        trigger.update(kind="condition", conditions=[{"condition": "ready", "satisfied": True, "time": "00:20", "evidence": "ready signal"}])
        task.validate_result(sample, context)
        for change in ({"satisfied": False}, {"time": "00:21"}):
            bad = copy.deepcopy(sample)
            bad["responses"][0]["trigger"]["conditions"][0].update(change)
            with self.assertRaises(ValueError):
                task.validate_result(bad, context)
        task = get_task("object_counting_pr")
        for operator, before, threshold in (("ge", 1, 2), ("gt", 1, 1), ("le", 3, 2), ("lt", 3, 3), ("eq", 3, 2)):
            sample, context = task_sample(task.name)
            sample["responses"][0]["trigger"].update(kind="count", previous_count=before, count=2, threshold=threshold, operator=operator)
            task.validate_result(sample, context)

    def test_non_start_and_invalid_context(self):
        task = get_task("event_understanding_pr")
        sample, context = task_sample(task.name)
        context = replace(context, question_position="non_start")
        sample["question_time"] = "00:10"
        task.validate_result(sample, context)
        for timestamp in ("00:00", "00:21", "02:00"):
            bad = copy.deepcopy(sample)
            bad["question_time"] = timestamp
            with self.assertRaises(ValueError):
                task.validate_result(bad, context)
        for bad_context in (replace(context, duration_seconds=float("nan")), replace(context, preferred_modality="unknown"),
                            replace(context, available_tools={}), replace(context, agent_delay_seconds=20)):
            with self.assertRaises(ValueError):
                task.validate_result(sample, bad_context)

    def test_cumulative_history_and_identity_deduplication(self):
        task = get_task("action_counting_rtd")
        sample, context = task_sample(task.name)
        mutations = (
            lambda s: s["responses"][1]["details"]["occurrences"][0].update(id="different"),
            lambda s: s["responses"][1]["details"]["occurrences"][1].update(id="a"),
            lambda s: s["responses"][1]["details"]["occurrences"][1].update(time="00:41"),
            lambda s: s["responses"][1]["details"].update(counting_start="00:01"),
        )
        for mutation in mutations:
            bad = copy.deepcopy(sample)
            mutation(bad)
            with self.assertRaises(ValueError):
                task.validate_result(bad, context)
        sample, context = task_sample("object_counting_pr")
        sample["responses"][0]["details"]["target_ids"] = ["a", "a"]
        with self.assertRaises(ValueError):
            get_task("object_counting_pr").validate_result(sample, context)

    def test_direct_responses_cannot_contain_agent_calls(self):
        sample, context = task_sample("grounding_pr")
        sample["responses"][0]["tool_calls"] = []
        with self.assertRaises(ValueError):
            get_task("grounding_pr").validate_result(sample, context)
