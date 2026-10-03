import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_generation.src import cli, client, video_caption
from data_generation.src.common import extract_json_array
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
