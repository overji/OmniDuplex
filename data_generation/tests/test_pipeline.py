import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from data_generation.src import cli, client, config, video_caption
from data_generation.src.common import SAMPLE_SCHEMA, extract_json_array
from data_generation.src.registry import TASKS
from data_generation.src.tasks import _agent, _shared, event_alert, route, state_monitoring
from data_generation.tests.fixtures import caption


class CaptionTests(unittest.TestCase):
    def test_valid_caption_and_rounded_end(self):
        self.assertEqual(video_caption.validate_caption(caption(60), 59.6), caption(60))
        self.assertEqual(video_caption.validate_caption(caption(60), 60.4), caption(60))
        self.assertEqual(extract_json_array('```json\n' + json.dumps(caption(60)) + '\n```'), caption(60))

    def test_invalid_caption(self):
        invalid = [[], {}, {"audio_events": [], "visual_events": []}, ["bad"]]
        for field, value in (("start", "00:01"), ("end", "00:00"), ("end", "00:46"),
                             ("end", "00:60"), ("caption", ""), ("audio", None)):
            item = copy.deepcopy(caption(60))
            item[0][field] = value
            invalid.append(item)
        for field, value in (("start", "00:29"), ("start", "00:31"), ("end", "00:59"), ("end", "01:01")):
            item = copy.deepcopy(caption(60))
            item[1][field] = value
            invalid.append(item)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                video_caption.validate_caption(value, 60)
        with self.assertRaises(ValueError):
            extract_json_array('{}')

    def test_paper_prompt_and_single_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(video_caption, "probe_video_duration", return_value=120):
            bundle = video_caption.build_video_prompts(Path("video.mp4"))
            self.assertIn("approximately\n  8 segments", bundle.caption_prompt)
            self.assertIn("02:00", bundle.caption_prompt)
            self.assertIn("5-30 seconds", bundle.caption_prompt)
            self.assertNotIn("{duration", bundle.caption_prompt)
            video_caption._write_prompts(bundle, Path(directory))
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {"caption.prompt.md", "video-input.md"})

    def test_caption_cli_one_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "caption.json"
            with patch.object(video_caption, "probe_video_duration", return_value=60), patch.object(
                cli, "chat_completion", return_value=json.dumps(caption(60))
            ) as call:
                status = cli.main(["--task", "caption", "--video", "test.mp4", "--model", "model",
                                   "--api-key", "test", "--output", str(out)])
            self.assertEqual(status, 0)
            call.assert_called_once()
            self.assertTrue(call.call_args.kwargs["json_array"])
            self.assertEqual(json.loads(out.read_text()), caption(60))

    def test_api_array_does_not_force_object_format(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return b'{"choices":[{"message":{"content":"[]"}}]}'
        with patch.object(client, "build_user_content", return_value=[]), patch.object(
            client.urllib.request, "urlopen", return_value=Response()
        ) as call:
            for array in (True, False):
                client.chat_completion("system", "user", video_path=Path("v.mp4"), api_key="test",
                                       base_url="https://example.invalid/v1", model="m", timeout=1,
                                       retries=0, json_array=array)
                payload = json.loads(call.call_args.args[0].data)
                self.assertEqual(payload["temperature"], 0.1)
                self.assertEqual("response_format" in payload, not array)


class RequestTests(unittest.TestCase):
    def test_all_tasks_receive_full_caption(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory) / "caption.json"
            path.write_text(json.dumps(caption()))
            for module in (_agent, _shared, event_alert, state_monitoring, route):
                stack.enter_context(patch.object(module, "probe_video_duration", return_value=120))
            for name, module in TASKS.items():
                with self.subTest(task=name):
                    request = module.build_request(path, Path("video.mp4"))
                    user = json.loads(request.user_prompt)
                    self.assertEqual(user["caption"], caption())
                    self.assertEqual(request.video_path.name, "video.mp4")
                    self.assertNotIn("{{", request.system_prompt)
                    self.assertNotIn("objects", user)
                    self.assertIn("caption", request.system_prompt.lower())
                    with self.assertRaises(TypeError):
                        module.build_request(path, Path("video.mp4"), objects_path=path)

    def test_legacy_inputs_and_cli_fail(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(_shared, "probe_video_duration", return_value=120):
            path = Path(directory) / "events.json"
            path.write_text('{"audio_events":[],"visual_events":[]}')
            with self.assertRaises(ValueError):
                TASKS["target_grounding"].build_request(path, Path("video.mp4"))
        for args in (["--task", "triggered_complex_reasoning", "--video", "v.mp4"],
                     ["--task", "event_alert", "--video", "v.mp4"],
                     ["--task", "event_alert", "--video", "v.mp4", "caption.json", "--objects", "o.json"],
                     ["--task", "caption", "--video", "v.mp4", "caption.json"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(args)

    def test_sampling_boundaries(self):
        for draws, expected in (([0.49], "positive"), ([0.5, 0.49], "no_context"), ([0.5, 0.5], "missing_tool")):
            with patch.object(config.random, "random", side_effect=draws):
                self.assertEqual(config.choose_agent_sample_type(), expected)

    def test_negative_context_not_polluted_by_generator_catalog(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(_agent, "probe_video_duration", return_value=5):
            path = Path(directory) / "caption.json"
            path.write_text(json.dumps(caption(5)))
            for sample_type in ("no_context", "missing_tool"):
                with patch.object(config, "choose_agent_sample_type", return_value=sample_type):
                    request = _agent.build_request(path, Path("v.mp4"), "event_agent")
                frontend = json.loads(request.user_prompt)["frontend_context"]
                self.assertNotIn(request.context.required_missing_tool, frontend["system_prompt"])
                self.assertIsNone(request.context.agent_delay_seconds)
                self.assertIsNone(request.context.initial_response_delay_seconds)
                self.assertIn("Refuse immediately", request.system_prompt)
                if sample_type == "no_context":
                    self.assertEqual(frontend["tools"], [])
                    self.assertNotIn("agent_call", frontend["system_prompt"])
                else:
                    self.assertEqual(frontend["tools"][0]["function"]["name"], "agent_call")

    def test_agent_cli_persists_context(self):
        from data_generation.tests.fixtures import agent_sample
        sample, context = agent_sample(sample_type="no_context")
        request = _shared.GenerationRequest("system", "user", context, Path("test.mp4"))
        output = io.StringIO()
        with patch.object(TASKS["event_agent"], "build_request", return_value=request), patch.object(
            cli, "chat_completion", return_value=json.dumps(sample)
        ), redirect_stdout(output):
            self.assertEqual(cli.main(["--task", "event_agent", "--video", "test.mp4", "caption.json",
                                       "--model", "m", "--api-key", "test"]), 0)
        saved = json.loads(output.getvalue())
        self.assertEqual(saved["schema"], SAMPLE_SCHEMA)
        self.assertEqual(saved["frontend_context"]["tools"], [])
        self.assertEqual(saved["task"], "event_agent")
        self.assertEqual(saved["sample_type"], "no_context")
        self.assertEqual(saved["generation_context"]["sample_type"], "no_context")
