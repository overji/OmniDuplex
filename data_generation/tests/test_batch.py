import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from data_generation.src.cli import batch
from data_generation.src.application import generation, requests, video_caption
from data_generation.src.application import batch as batch_service
from data_generation.src.domain.catalog import ABILITIES, TASK_CATALOG, split_task
from data_generation.src.domain.tasks import interaction
from data_generation.tests.fixtures import caption, task_sample, agent_sample


class BatchTests(unittest.TestCase):
    def setup_run(self, root, tasks):
        videos = root / "videos"
        videos.mkdir()
        (videos / "example.mp4").write_bytes(b"source video")
        args = batch.parse_args(["--videos-dir", str(videos), "--output-dir", str(root / "output"),
                                 "--tasks", *tasks, "--api-key", "test", "--model", "test", "--seed", "42"])
        return args

    def run_batch(self, args):
        if not args.resume:
            parent = getattr(args, "output_parent", args.output_dir)
            args.output_parent = parent
            args.output_dir = parent
        result = batch.run(args)
        if not args.resume:
            args.output_dir = max(args.output_parent.iterdir())
        return result

    def api(self, selected, fail_task=None):
        selected_abilities = list(dict.fromkeys(split_task(name)[0] for name in selected))
        def respond(system, user, **kwargs):
            if kwargs["json_array"]:
                return json.dumps(caption())
            if "Task: " not in system:
                return json.dumps({"status": "ok", "ranked_tasks": [
                    {"task": name, "rank": index, "rationale": "Clear changes."} for index, name in enumerate(selected_abilities, 1)],
                    "unsuitable_tasks": [{"task": name, "reason": "Not enough evidence."} for name in ABILITIES if name not in selected_abilities]})
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
        for module in (requests, video_caption):
            stack.enter_context(patch.object(module, "probe_video_duration", return_value=120))
        stack.enter_context(patch.object(interaction.config, "choose_question_position", return_value="start"))
        stack.enter_context(patch.object(interaction.config, "choose_agent_delay", return_value=20))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        call = stack.enter_context(patch.object(generation, "chat_completion", side_effect=api))
        return stack, call

    def test_full_matrix_batch_and_resume_no_api(self):
        tasks = list(TASK_CATALOG)
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            stack, call = self.patches(self.api(tasks))
            with stack:
                self.assertEqual(self.run_batch(args), 0)
                self.assertEqual(call.call_count, 38)
                args.resume = True
                call.reset_mock()
                self.assertEqual(self.run_batch(args), 0)
                call.assert_not_called()
            bundle = json.loads((args.output_dir / "all_qa.json").read_text())
            report = json.loads((args.output_dir / "coverage.json").read_text())
            self.assertEqual(len(bundle), 36)
            self.assertEqual(report["accepted"], 36)
            self.assertEqual(set(report["tasks"]), set(TASK_CATALOG))
            self.assertTrue(all(stage["reused"] for stage in report["stages"]))
            self.assertNotIn("api_key", (args.output_dir / ".state" / "manifest.json").read_text())

    def test_resume_invalidates_tampered_artifact_video_and_config(self):
        tasks = ["object_counting_rtd"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            stack, call = self.patches(self.api(tasks))
            with stack:
                self.assertEqual(self.run_batch(args), 0)
                args.resume = True
                path = next(args.output_dir.glob("*/qa/object_counting_rtd.0000.json"))
                path.write_text('{"corrupted":true}')
                call.reset_mock()
                self.assertEqual(self.run_batch(args), 0)
                self.assertEqual(call.call_count, 1)
                (args.videos_dir / "example.mp4").write_bytes(b"changed source")
                call.reset_mock()
                self.assertEqual(self.run_batch(args), 0)
                self.assertEqual(call.call_count, 3)
                args.model = "other-model"
                call.reset_mock()
                self.assertEqual(self.run_batch(args), 0)
                self.assertEqual(call.call_count, 3)
                with patch.object(batch_service, "implementation_digest", return_value="new-prompt"):
                    call.reset_mock()
                    self.assertEqual(self.run_batch(args), 0)
                    self.assertEqual(call.call_count, 3)

    def test_failure_continues_recovers_and_does_not_merge_stale_files(self):
        tasks = ["attribute_perception_rtd", "object_counting_pr"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            stack, call = self.patches(self.api(tasks, fail_task=tasks[0]))
            with stack:
                self.assertEqual(self.run_batch(args), 1)
                bundle = json.loads((args.output_dir / "all_qa.json").read_text())
                self.assertEqual([item["task"] for item in bundle], [tasks[1]])
                args.resume = True
                call.side_effect = self.api(tasks)
                call.reset_mock()
                self.assertEqual(self.run_batch(args), 0)
                self.assertEqual(call.call_count, 1)
                # A new failed attempt must not silently reuse an old successful file.
                args.resume = False
                call.side_effect = self.api(tasks, fail_task=tasks[0])
                self.assertEqual(self.run_batch(args), 1)
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
                self.assertEqual(self.run_batch(args), 0)
            report = json.loads((args.output_dir / "coverage.json").read_text())
            self.assertEqual(report["tasks"][tasks[0]]["skip"], 1)
            self.assertEqual(report["tasks"][tasks[1]]["unsuitable"], 1)
            self.assertEqual(report["accepted"], 0)

    def test_suitable_ability_attempts_all_modes_and_rejects_other_ability(self):
        modes = ["attribute_perception_rtd", "attribute_perception_pr", "attribute_perception_agent"]
        rejected = "object_counting_pr"
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), modes + [rejected])
            # One mode in this fixture selects its entire ability for routing.
            stack, call = self.patches(self.api([modes[0]]))
            with stack:
                self.assertEqual(self.run_batch(args), 0)
                self.assertEqual(call.call_count, 5)
            report = json.loads((args.output_dir / "coverage.json").read_text())
            self.assertTrue(all(report["tasks"][task]["ok"] == 1 for task in modes))
            self.assertEqual(report["tasks"][rejected]["unsuitable"], 1)
            route_result = json.loads(next(args.output_dir.glob("*/.state/route.json")).read_text())
            self.assertEqual(route_result["ranked_tasks"][0]["task"], "attribute_perception")

    def test_task_selection_limits_modes_of_suitable_ability(self):
        task = "attribute_perception_pr"
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), [task])
            stack, call = self.patches(self.api([task]))
            with stack:
                self.assertEqual(self.run_batch(args), 0)
                self.assertEqual(call.call_count, 3)
            bundle = json.loads((args.output_dir / "all_qa.json").read_text())
            self.assertEqual([sample["task"] for sample in bundle], [task])

    def test_route_skip_prevents_all_generation(self):
        tasks = ["attribute_perception_rtd", "object_counting_pr"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            normal = self.api(tasks)

            def response(system, user, **kwargs):
                if not kwargs["json_array"]:
                    return '{"status":"skip","reason":"No suitable ability."}'
                return normal(system, user, **kwargs)

            stack, call = self.patches(response)
            with stack:
                self.assertEqual(self.run_batch(args), 0)
                self.assertEqual(call.call_count, 2)
            report = json.loads((args.output_dir / "coverage.json").read_text())
            self.assertEqual(report["accepted"], 0)
            self.assertTrue(all(report["tasks"][task]["unsuitable"] == 1 for task in tasks))

    def test_timestamped_runs_per_video_totals_and_cleanup(self):
        tasks = ["object_counting_rtd"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            (args.videos_dir / "nested").mkdir()
            (args.videos_dir / "nested" / "example.mp4").write_bytes(b"another video")
            stack, _ = self.patches(self.api(tasks))
            with stack:
                self.assertEqual(self.run_batch(args), 0)
                first = args.output_dir
                self.assertRegex(first.name, r"^\d{8}_\d{6}_\d{6}$")
                folders = sorted(first.glob("example-*"))
                self.assertEqual(len(folders), 2)
                combined = []
                for folder in folders:
                    self.assertTrue((folder / "caption.json").is_file())
                    self.assertTrue((folder / ".state" / "route.json").is_file())
                    self.assertTrue((folder / "qa" / "object_counting_rtd.0000.json").is_file())
                    video_bundle = json.loads((folder / "all_qa.json").read_text())
                    self.assertEqual(video_bundle[0]["video_folder"], folder.name)
                    combined.extend(video_bundle)
                bundle = json.loads((first / "all_qa.json").read_text())
                self.assertEqual(sorted(combined, key=lambda x: x["video"]), sorted(bundle, key=lambda x: x["video"]))
                self.assertFalse(list(first.rglob("*.usage.json")))
                self.assertFalse(list(first.rglob("*.pending")))
                report = json.loads((first / "coverage.json").read_text())
                self.assertEqual(report["api_usage"]["calls"], 6)
                self.assertEqual(self.run_batch(args), 0)
                self.assertNotEqual(first, args.output_dir)
                self.assertTrue((first / "all_qa.json").is_file())

    def test_cleanup_preserves_usage_on_resume(self):
        tasks = ["object_counting_rtd"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            normal = self.api(tasks)

            def measured(*values, **kwargs):
                kwargs["usage_callback"]({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
                return normal(*values, **kwargs)

            stack, call = self.patches(measured)
            with stack:
                self.assertEqual(self.run_batch(args), 0)
                before = json.loads((args.output_dir / "coverage.json").read_text())["api_usage"]
                self.assertEqual(before["total_tokens"], 30)
                self.assertFalse(list(args.output_dir.rglob("*.usage.json")))
                args.resume = True
                call.reset_mock()
                self.assertEqual(self.run_batch(args), 0)
                call.assert_not_called()
                after = json.loads((args.output_dir / "coverage.json").read_text())["api_usage"]
                self.assertEqual(before, after)

    def test_invalid_result_keeps_usage_and_empty_video_bundle(self):
        tasks = ["object_counting_rtd"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            stack, _ = self.patches(lambda *args, **kwargs: "{}")
            with stack:
                self.assertEqual(self.run_batch(args), 1)
            self.assertEqual(len(list(args.output_dir.rglob("*.usage.json"))), 1)
            self.assertEqual(json.loads(next(args.output_dir.glob("*/all_qa.json")).read_text()), [])
            report = json.loads((args.output_dir / "coverage.json").read_text())
            self.assertEqual(report["api_usage"]["calls"], 1)
            self.assertEqual(report["tasks"][tasks[0]]["blocked"], 1)

    def test_cleanup_failure_is_reported_without_losing_results(self):
        tasks = ["object_counting_rtd"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            stack, _ = self.patches(self.api(tasks))
            with stack, patch.object(Path, "unlink", side_effect=OSError("cannot delete")):
                self.assertEqual(self.run_batch(args), 1)
            self.assertEqual(len(json.loads((args.output_dir / "all_qa.json").read_text())), 1)
            report = json.loads((args.output_dir / "coverage.json").read_text())
            self.assertEqual(len(report["errors"]), 3)
            self.assertTrue(all("cleanup failed" in error["error"] for error in report["errors"]))

    def test_summary_write_failure_does_not_clean_sidecars(self):
        tasks = ["object_counting_rtd"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            original = batch_service.write_json

            def write(path, value):
                if path.name == "all_qa.json":
                    raise OSError("disk full")
                return original(path, value)

            stack, _ = self.patches(self.api(tasks))
            with stack, patch.object(batch_service, "write_json", side_effect=write):
                with self.assertRaisesRegex(OSError, "disk full"):
                    self.run_batch(args)
            self.assertEqual(len(list(args.output_dir.rglob("*.usage.json"))), 3)

    def test_resume_requires_explicit_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), ["object_counting_rtd"])
            args.resume = True
            with self.assertRaisesRegex(ValueError, "pointing to a run"):
                batch.run(args)

    def test_legacy_resume_retains_layout_and_persists_sidecar_usage(self):
        tasks = ["object_counting_rtd"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            stack, call = self.patches(self.api(tasks))
            with stack:
                self.assertEqual(self.run_batch(args), 0)
                manifest_path = args.output_dir / ".state" / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                del manifest["layout_version"]
                artifacts = {}
                for key, artifact in manifest["artifacts"].items():
                    new_key = key.replace("/qa/", "/").replace("/.state/", "/")
                    source = args.output_dir / key
                    destination = args.output_dir / new_key
                    if source != destination:
                        source.rename(destination)
                    destination.with_suffix(".usage.json").write_text(json.dumps(artifact.pop("usage")))
                    artifacts[new_key] = artifact
                manifest["artifacts"] = artifacts
                (args.output_dir / "manifest.json").write_text(json.dumps(manifest))
                manifest_path.unlink()
                for folder in args.output_dir.glob("*/qa"):
                    folder.rmdir()
                args.resume = True
                call.reset_mock()
                self.assertEqual(self.run_batch(args), 0)
                call.assert_not_called()
                self.assertTrue(next(args.output_dir.glob("*/route.json")).is_file())
                self.assertFalse(list(args.output_dir.rglob("*.usage.json")))
                self.assertEqual(json.loads((args.output_dir / "coverage.json").read_text())["api_usage"]["calls"], 3)
                self.assertEqual(self.run_batch(args), 0)
                call.assert_not_called()
                self.assertEqual(json.loads((args.output_dir / "coverage.json").read_text())["api_usage"]["calls"], 3)

    def test_video_checkpoint_precedes_next_video_and_preserves_unowned_files(self):
        tasks = ["object_counting_rtd"]
        with tempfile.TemporaryDirectory() as directory:
            args = self.setup_run(Path(directory), tasks)
            (args.videos_dir / "second.mp4").write_bytes(b"second")
            normal = self.api(tasks)
            captions = 0
            unowned = None

            def respond(*values, **kwargs):
                nonlocal captions, unowned
                if kwargs["json_array"]:
                    captions += 1
                    run_dir = next(args.output_dir.iterdir())
                    if captions == 1:
                        unowned = run_dir / "user.usage.json"
                        unowned.write_text('{}')
                    if captions == 2:
                        self.assertEqual(len(json.loads((run_dir / "all_qa.json").read_text())), 1)
                        self.assertEqual(json.loads((run_dir / "coverage.json").read_text())["accepted"], 1)
                        self.assertEqual(list(run_dir.rglob("*.usage.json")), [unowned])
                return normal(*values, **kwargs)

            stack, _ = self.patches(respond)
            with stack:
                self.assertEqual(self.run_batch(args), 0)
            self.assertTrue(unowned.is_file())
