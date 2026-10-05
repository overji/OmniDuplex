import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_generation.src.infrastructure import client


def stream(*events):
    return io.BytesIO(b"".join(
        b"data: " + (event if isinstance(event, str) else json.dumps(event)).encode() + b"\n\n"
        for event in events
    ))


def delta(content, **fields):
    return {"choices": [{"index": 0, "delta": {"content": content}, **fields}]}


class QwenClientTests(unittest.TestCase):
    def call(self, **kwargs):
        options = dict(video_path=Path("unused.mp4"), video_url="https://media.invalid/video.mp4",
                       api_key="test", base_url="https://qwen.invalid/v1", model="qwen3.8-omni-flash",
                       timeout=120, retries=0, provider="qwen")
        options.update(kwargs)
        return client.chat_completion("Return JSON", "Describe the video", **options)

    def test_stream_ignores_reasoning_and_usage_and_preserves_input(self):
        response = stream({"choices": [{"delta": {"reasoning_content": "private reasoning"}}]},
                          delta('{"ok":'), delta('true}', finish_reason="stop"),
                          {"choices": [], "usage": {"total_tokens": 1}}, "[DONE]")
        with patch.object(client.urllib.request, "urlopen", return_value=response) as call:
            self.assertEqual(self.call(), '{"ok":true}')
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://qwen.invalid/v1/chat/completions")
        self.assertEqual(body["temperature"], 0.1)
        self.assertTrue(body["stream"])
        self.assertEqual(body["modalities"], ["text"])
        self.assertNotIn("response_format", body)
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["messages"][1]["content"][0]["video_url"]["url"], "https://media.invalid/video.mp4")

    def test_local_video_is_uploaded_and_oss_header_is_set(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            video.write(b"original-video")
            video.flush()
            with patch.object(client, "upload_video", return_value="oss://temporary/video.mp4") as upload, patch.object(
                client.urllib.request, "urlopen", return_value=stream(delta("[]", finish_reason="stop"))
            ) as call:
                self.assertEqual(self.call(video_path=Path(video.name), video_url=None, json_array=True), "[]")
            content = json.loads(call.call_args.args[0].data)["messages"][1]["content"]
            self.assertEqual(content[0]["video_url"]["url"], "oss://temporary/video.mp4")
            upload.assert_called_once()
            self.assertEqual(upload.call_args.args[0], Path(video.name))
            self.assertEqual(upload.call_args.kwargs["model"], "qwen3.8-omni-flash")
            self.assertEqual(call.call_args.args[0].get_header("X-dashscope-ossresourceresolve"), "enable")

    def test_large_video_is_referenced_without_reading_bytes(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            video.truncate(7_500_000)
            with patch.object(client, "upload_video", return_value="oss://temporary/large.mp4"), patch.object(
                Path, "read_bytes", side_effect=AssertionError("must not read video into request")
            ), patch.object(
                client.urllib.request, "urlopen", return_value=stream(delta("{}", finish_reason="stop"))
            ) as call:
                self.assertEqual(self.call(video_path=Path(video.name), video_url=None), "{}")
            content = json.loads(call.call_args.args[0].data)["messages"][1]["content"]
            self.assertEqual(content[0]["video_url"]["url"], "oss://temporary/large.mp4")

    def test_explicit_oss_uri_skips_upload(self):
        with patch.object(client, "upload_video") as upload, patch.object(
            client.urllib.request, "urlopen", return_value=stream(delta("{}", finish_reason="stop"))
        ) as call:
            self.assertEqual(self.call(video_url="oss://temporary/shared.mp4"), "{}")
            upload.assert_not_called()
        content = json.loads(call.call_args.args[0].data)["messages"][1]["content"]
        self.assertEqual(content[0]["video_url"]["url"], "oss://temporary/shared.mp4")

    def test_invalid_remote_url_rejected_before_network(self):
        with patch.object(client.urllib.request, "urlopen") as call:
            with self.assertRaisesRegex(ValueError, "remote"):
                self.call(video_url="data:video/mp4;base64,invalid")
            with self.assertRaisesRegex(ValueError, "remote"):
                self.call(video_url="file:///tmp/video.mp4")
            call.assert_not_called()

    def test_incomplete_or_error_streams_are_not_accepted(self):
        for response in (stream(delta('{"ok":')),
                         stream(delta("{}", finish_reason="length")),
                         stream({"error": {"code": "invalid_parameter"}}),
                         stream("[DONE]")):
            with self.subTest(response=response), patch.object(client.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(RuntimeError):
                    self.call()

    def test_retry_discards_partial_text(self):
        with patch.object(client.urllib.request, "urlopen", side_effect=[
            stream(delta("partial")), stream(delta("{}", finish_reason="stop"))
        ]) as call, patch.object(client.time, "sleep"):
            self.assertEqual(self.call(retries=1), "{}")
            self.assertEqual(call.call_count, 2)
