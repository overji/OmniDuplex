import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS

from data_generation.src.cli.pipeline import parse_args as pipeline_args
from data_generation.src.cli.options import application_options
from data_generation.src.application.options import PipelineOptions
from data_generation.src.application import pipeline
from data_generation.src.domain.selection import select_tasks
from data_generation.src.domain.common import SAMPLE_SCHEMA
from data_generation.src.infrastructure import gemini
from data_generation.tests import test_batch


def parse_args(argv):
    # These tests mock Qwen's automatic local-video upload path.
    return pipeline_args(argv if '--provider' in argv else [*argv, '--provider', 'qwen'])


class PipelineTests(unittest.TestCase):
    def test_sampling_weights_boundaries_and_repeatability(self):
        ranked = [{"task": "object_perception", "rank": 1}, {"task": "action_perception", "rank": 2}]
        kwargs = dict(seed=42, pr_probability=1, agent_probability=1, agent_positive_probability=1)
        decision = select_tasks(ranked, {"object_perception_pr": 9}, **kwargs)
        self.assertEqual(decision, select_tasks(ranked, {"object_perception_pr": 9}, **kwargs))
        self.assertEqual(decision["ability_draws"][0]["weights"], [0.1, 0.5])
        self.assertEqual(decision["tasks"][1]["branch"], "positive")
        kwargs.update(pr_probability=0, agent_probability=0)
        result = select_tasks(ranked, {}, **kwargs)
        self.assertEqual(len(result['tasks']), len(result['abilities']))
        self.assertTrue(all(item['task'].endswith('_rtd') for item in result['tasks']))
        kwargs.update(agent_probability=1, agent_positive_probability=0)
        self.assertIn(select_tasks(ranked, {}, **kwargs)["tasks"][1]["branch"], {"no_context", "missing_tool"})

    def test_history_counts_only_canonical_successful_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            nested = run / "video"
            nested.mkdir(parents=True)
            (run / "manifest.json").write_text('{"artifacts":{}}')
            sample = {"schema": SAMPLE_SCHEMA, "status": "ok", "task": "object_perception_agent", "responses": [1, 2, 3]}
            bundle = [sample, dict(sample, status="skip"), dict(sample, schema="old")]
            (run / "all_qa.json").write_text(json.dumps(bundle))
            (nested / "all_qa.json").write_text(json.dumps(bundle))
            copied = root / "merged-copy"
            copied.mkdir()
            (copied / "all_qa.json").write_text(json.dumps(bundle))
            self.assertEqual(pipeline.historical_counts(root), {"object_perception_agent": 1})
            self.assertEqual(pipeline.historical_counts(root, exclude=run.resolve()), {})

    def test_pipeline_resume_and_no_resampling_after_history_changes(self):
        helper = test_batch.BatchTests()
        tasks = ["object_perception_pr", "object_perception_agent"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"source")
            # A sibling video must never be processed by the single-video entrypoint.
            (root / "sibling.mp4").write_bytes(b"other")
            parsed = parse_args(["--video", str(video), "--output-dir", str(root / "out"),
                                 "--api-key", "test", "--model", "test", "--pr-probability", "1",
                                 "--agent-probability", "1", "--agent-positive-probability", "1"])
            args = application_options(PipelineOptions, parsed)
            stack, call = helper.patches(helper.api(tasks))
            with stack:
                self.assertEqual(pipeline.run(args), 0)
                self.assertEqual(call.call_count, 4)
                run = next(args.output_dir.iterdir())
                manifest = json.loads((run / ".state/manifest.json").read_text())
                history = args.output_dir / "another-run"
                history.mkdir()
                (history / "manifest.json").write_text('{"artifacts":{}}')
                (history / "all_qa.json").write_bytes((run / "all_qa.json").read_bytes())
                args.resume, args.output_dir = True, run
                call.reset_mock()
                self.assertEqual(pipeline.run(args), 0)
                call.assert_not_called()
                self.assertEqual(json.loads((run / ".state/manifest.json").read_text())["selections"], manifest["selections"])
                args.agent_probability = 0
                with self.assertRaisesRegex(ValueError, "settings"):
                    pipeline.run(args)

    def test_skip_is_not_replaced(self):
        helper = test_batch.BatchTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"source")
            args = application_options(PipelineOptions, parse_args([
                "--video", str(video), "--output-dir", str(root / "out"), "--api-key", "test",
                "--model", "test", "--agent-probability", "0"]))
            original = helper.api(["object_perception_pr"])
            def api(system, user, **kwargs):
                return json.dumps({"status": "skip", "reason": "No suitable episode."}) if "\nTask: " in system else original(system, user, **kwargs)
            stack, call = helper.patches(api)
            with stack:
                self.assertEqual(pipeline.run(args), 0)
                self.assertEqual(call.call_count, 3)
            run = next(args.output_dir.iterdir())
            self.assertEqual(json.loads((run / "all_qa.json").read_text()), [])

    def test_agent_failure_resume_only_calls_incomplete_stage(self):
        helper = test_batch.BatchTests()
        tasks = ["object_perception_pr", "object_perception_agent"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"source")
            args = application_options(PipelineOptions, parse_args([
                "--video", str(video), "--output-dir", str(root / "out"), "--api-key", "test",
                "--model", "test", "--pr-probability", "1", "--agent-probability", "1",
                "--agent-positive-probability", "1"]))
            stack, call = helper.patches(helper.api(tasks, fail_task="object_perception_agent"))
            with stack:
                self.assertEqual(pipeline.run(args), 1)
                run = next(args.output_dir.iterdir())
                self.assertEqual(len(json.loads((run / "all_qa.json").read_text())), 1)
                args.resume, args.output_dir = True, run
                call.side_effect = helper.api(tasks)
                call.reset_mock()
                self.assertEqual(pipeline.run(args), 0)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(len(json.loads((run / "all_qa.json").read_text())), 2)

    def test_negative_agent_context_and_route_skip(self):
        helper = test_batch.BatchTests()
        tasks = ["object_perception_pr", "object_perception_agent"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"source")
            args = application_options(PipelineOptions, parse_args([
                "--video", str(video), "--output-dir", str(root / "out"), "--api-key", "test",
                "--model", "test", "--pr-probability", "1", "--agent-probability", "1",
                "--agent-positive-probability", "0"]))
            stack, call = helper.patches(helper.api(tasks))
            with stack:
                self.assertEqual(pipeline.run(args), 0)
            run = next(args.output_dir.iterdir())
            bundle = json.loads((run / "all_qa.json").read_text())
            agent = next(sample for sample in bundle if sample["mode"] == "agent")
            self.assertIn(agent["sample_type"], {"no_context", "missing_tool"})
            self.assertEqual(agent["sample_type"], agent["generation_context"]["sample_type"])
            original = helper.api(tasks)
            def api(system, user, **kwargs):
                return original(system, user, **kwargs) if kwargs["json_array"] else json.dumps({"status": "skip", "reason": "No supported ability."})
            args.output_dir = root / "skip-out"
            stack, call = helper.patches(api)
            with stack:
                self.assertEqual(pipeline.run(args), 0)
                self.assertEqual(call.call_count, 2)
            report = json.loads((next(args.output_dir.iterdir()) / "coverage.json").read_text())
            self.assertEqual(report["routing_status"], "skip")
            self.assertEqual(report["selection"]["tasks"], [])
            self.assertEqual(report['average_selected_abilities'], 0)
            self.assertEqual(report['average_attempted_abilities'], 0)
            self.assertEqual(report['average_successful_abilities'], 0)


class GeminiTests(unittest.TestCase):
    def fake_sdk(self):
        from unittest.mock import Mock
        uploaded = NS(name="files/test", uri="file-uri", mime_type="video/mp4", state=NS(name="ACTIVE"))
        usage = NS(candidates_token_count=3, thoughts_token_count=2, prompt_token_count=7,
                   total_token_count=12, cached_content_token_count=1, model_dump=lambda **kw: {"total_token_count": 12})
        client = Mock()
        client.files.upload.return_value = uploaded
        client.models.generate_content.return_value = NS(text='{"status":"skip"}',
            candidates=[NS(finish_reason=NS(name="STOP"))], usage_metadata=usage)
        genai = NS(Client=Mock(return_value=client))
        types = NS(HttpOptions=lambda **kw: kw, HttpRetryOptions=lambda **kw: kw,
                   GenerateContentConfig=lambda **kw: kw, Part=NS(from_uri=lambda **kw: kw))
        return genai, types, client

    def test_upload_reused_usage_and_cleanup(self):
        genai, types, client = self.fake_sdk()
        usage = []
        with patch.object(gemini, "_sdk", return_value=(genai, types)), gemini.video_session():
            for _ in range(3):
                gemini.completion("system", "text", video_path=Path("test.mp4"), api_key="test",
                                  base_url="https://example.test", model="test", timeout=10, retries=2,
                                  usage_callback=usage.append)
            client.files.upload.assert_called_once()
            self.assertEqual(client.models.generate_content.call_count, 3)
            self.assertEqual(usage[0]["completion_tokens"], 5)
        client.files.delete.assert_called_once_with(name="files/test")
        client.close.assert_called_once()

    def test_failed_processing_timeout_and_truncation(self):
        genai, types, client = self.fake_sdk()
        uploaded = client.files.upload.return_value
        uploaded.state.name = "FAILED"
        with self.assertRaisesRegex(RuntimeError, "processing failed"):
            gemini._active_file(client, uploaded, 10)
        uploaded.state.name = "PROCESSING"
        with patch.object(gemini.time, "monotonic", side_effect=[0, 11]), self.assertRaisesRegex(RuntimeError, "timed out"):
            gemini._active_file(client, uploaded, 10)
        uploaded.state.name = "ACTIVE"
        client.models.generate_content.return_value.candidates[0].finish_reason.name = "MAX_TOKENS"
        with patch.object(gemini, "_sdk", return_value=(genai, types)), self.assertRaisesRegex(RuntimeError, "complete text"):
            gemini.completion("s", "u", video_path=Path("test.mp4"), api_key="test", base_url="test", model="test", timeout=10, retries=0)
        client.files.delete.assert_called_once()

    def test_gemini_environment_precedence(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text("GEMINI_API_KEY=file-key\nGEMINI_MODEL=file-model\n")
            with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True):
                args = parse_args(["--video", "test.mp4", "--output-dir", "out", "--provider", "gemini", "--env-file", str(env), "--model", "cli-model"])
            self.assertEqual((args.api_key, args.model), ("env-key", "cli-model"))

    def test_upload_failure_and_processing_poll(self):
        genai, types, client = self.fake_sdk()
        uploaded = client.files.upload.return_value
        processing = NS(name=uploaded.name, state=NS(name="PROCESSING"))
        client.files.get.return_value = uploaded
        with patch.object(gemini.time, "sleep"):
            self.assertIs(gemini._active_file(client, processing, 10), uploaded)
        client.files.upload.side_effect = OSError("upload unavailable")
        with patch.object(gemini, "_sdk", return_value=(genai, types)), self.assertRaisesRegex(RuntimeError, "upload failed"):
            gemini.completion("s", "u", video_path=Path("test.mp4"), api_key="test", base_url="test", model="test", timeout=10, retries=0)
        client.close.assert_called_once()
        client.files.delete.assert_not_called()
