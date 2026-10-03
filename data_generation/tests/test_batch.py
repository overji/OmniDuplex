import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from data_generation.src import batch, cli, video_caption
from data_generation.src.catalog import TASK_CATALOG
from data_generation.src.tasks import interaction, route
from data_generation.tests.fixtures import caption, task_sample, agent_sample


class BatchTests(unittest.TestCase):
    def setup_run(self, root, tasks):
        videos = root / "videos"
        videos.mkdir()
        (videos / "example.mp4").write_bytes(b"source video")
        args = batch.parse_args(["--videos-dir", str(videos), "--output-dir", str(root / "output"),
                                 "--tasks", *tasks, "--api-key", "test", "--model", "test", "--seed", "42"])
        return args

    def api(self, selected, fail_task=None):
        def respond(system, user, **kwargs):
            if kwargs["json_array"]:
                return json.dumps(caption())
            if "Task: " not in system:
                return json.dumps({"status": "ok", "ranked_tasks": [
                    {"task": name, "rank": index, "rationale": "Clear changes."} for index, name in enumerate(selected, 1)],
                    "unsuitable_tasks": [{"task": name, "reason": "Not enough evidence."} for name in TASK_CATALOG if name not in selected]})
            task = system.split("\nTask: ")[1].splitlines()[0]
            if task == fail_task:
                raise RuntimeError("simulated API failure")
            if task.endswith("_agent"):
                context = json.loads(system.split("Sampled constraints (immutable): ")[1].splitlines()[0])
                branch = context["sample_type"]
                sample, _ = agent_sample(task, branch)
                if branch != "positive":
                    sample["required_tools"] = [context["required_missing_tool"]]
                    sample["responses"][0]["missing_tools"] = sample["required_tools"]
            else:
                sample, _ = task_sample(task)
            return json.dumps(sample)
        return respond

    def patches(self, api):
        stack = ExitStack()
        for module in (interaction, route, video_caption):
            stack.enter_context(patch.object(module, "probe_video_duration", return_value=120))
        stack.enter_context(patch.object(interaction.config, "choose_question_position", return_value="start"))
        stack.enter_context(patch.object(interaction.config, "choose_agent_delay", return_value=20))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        call = stack.enter_context(patch.object(cli, "chat_completion", side_effect=api))
        return stack, call

    def test_full_matrix_batch_and_resume_no_api(self):
        tasks = list(TASK_CATALOG)
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            stack, call = self.patches(self.api(tasks))
            with stack:
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(call.call_count, 38)
                args.resume = True
                call.reset_mock()
                self.assertEqual(batch.run(args), 0)
                call.assert_not_called()
            bundle = json.loads((args.output_dir / "all_qa.json").read_text())
            report = json.loads((args.output_dir / "coverage.json").read_text())
            self.assertEqual(len(bundle), 36)
            self.assertEqual(report["accepted"], 36)
            self.assertEqual(set(report["tasks"]), set(TASK_CATALOG))
            self.assertTrue(all(stage["reused"] for stage in report["stages"]))
            self.assertNotIn("api_key", (args.output_dir / "manifest.json").read_text())

    def test_resume_invalidates_tampered_artifact_video_and_config(self):
        tasks = ["object_counting_rtd"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            stack, call = self.patches(self.api(tasks))
            with stack:
                self.assertEqual(batch.run(args), 0)
                args.resume = True
                path = next(args.output_dir.glob("*/object_counting_rtd.*.json"))
                path.write_text('{"corrupted":true}')
                call.reset_mock()
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(call.call_count, 1)
                (args.videos_dir / "example.mp4").write_bytes(b"changed source")
                call.reset_mock()
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(call.call_count, 3)
                args.model = "other-model"
                call.reset_mock()
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(call.call_count, 3)
                with patch.object(batch, "implementation_digest", return_value="new-prompt"):
                    call.reset_mock()
                    self.assertEqual(batch.run(args), 0)
                    self.assertEqual(call.call_count, 3)

    def test_failure_continues_recovers_and_does_not_merge_stale_files(self):
        tasks = ["attribute_perception_rtd", "object_counting_pr"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            stack, call = self.patches(self.api(tasks, fail_task=tasks[0]))
            with stack:
                self.assertEqual(batch.run(args), 1)
                bundle = json.loads((args.output_dir / "all_qa.json").read_text())
                self.assertEqual([item["task"] for item in bundle], [tasks[1]])
                args.resume = True
                call.side_effect = self.api(tasks)
                call.reset_mock()
                self.assertEqual(batch.run(args), 0)
                self.assertEqual(call.call_count, 1)
                # A new failed attempt must not silently reuse an old successful file.
                args.resume = False
                call.side_effect = self.api(tasks, fail_task=tasks[0])
                self.assertEqual(batch.run(args), 1)
            bundle = json.loads((args.output_dir / "all_qa.json").read_text())
            self.assertEqual([item["task"] for item in bundle], [tasks[1]])
            report = json.loads((args.output_dir / "coverage.json").read_text())
            self.assertEqual(report["tasks"][tasks[0]]["failed"], 1)

    def test_skip_and_unroutable_are_not_accepted(self):
        tasks = ["attribute_perception_rtd", "object_counting_pr"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            normal = self.api([tasks[0]])

            def response(system, user, **kwargs):
                if "\nTask:" in system:
                    return '{"status":"skip","reason":"No suitable updates."}'
                return normal(system, user, **kwargs)

            stack, _ = self.patches(response)
            with stack:
                self.assertEqual(batch.run(args), 0)
            report = json.loads((args.output_dir / "coverage.json").read_text())
            self.assertEqual(report["tasks"][tasks[0]]["skip"], 1)
            self.assertEqual(report["tasks"][tasks[1]]["unsuitable"], 1)
            self.assertEqual(report["accepted"], 0)
