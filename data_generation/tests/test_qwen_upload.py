import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from data_generation.src.infrastructure import qwen_upload


class Response(io.BytesIO):
    status = 200


POLICY = {
    "upload_host": "https://upload.example.invalid",
    "upload_dir": "dashscope-instant/test",
    "oss_access_key_id": "temporary-key",
    "policy": "signed-policy",
    "signature": "signature",
    "x_oss_object_acl": "private",
    "x_oss_forbid_overwrite": "true",
    "max_file_size_mb": "256",
}


class UploadTests(unittest.TestCase):
    def setUp(self):
        qwen_upload._UPLOAD_CACHE.clear()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(qwen_upload._UPLOAD_CACHE.clear)
        self.path = Path(self.directory.name) / "video.mp4"
        self.path.write_bytes(b"original-video-bytes")
        self.requests = []
        self.forms = []

    def response(self, request, **kwargs):
        self.requests.append(request)
        if request.get_method() == "GET":
            return Response(json.dumps({"data": POLICY}).encode())
        self.forms.append(b"".join(request.data))
        self.assertEqual(int(request.get_header("Content-length")), len(self.forms[-1]))
        self.assertNotIn("Authorization", request.headers)
        return Response(b"")

    def upload(self, **kwargs):
        options = dict(api_key="test-key", model="qwen3.8-omni-flash",
                       base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=120, retries=0)
        options.update(kwargs)
        return qwen_upload.upload_video(self.path, **options)

    def test_original_bytes_streamed_file_last_and_url_reused(self):
        with patch.object(qwen_upload.urllib.request, "urlopen", side_effect=self.response), patch.object(
            Path, "read_bytes", side_effect=AssertionError("must stream file")
        ):
            first = self.upload()
            self.assertEqual(self.upload(), first)
        self.assertEqual(len(self.requests), 2)
        self.assertIn("action=getPolicy&model=qwen3.8-omni-flash", self.requests[0].full_url)
        self.assertEqual(self.requests[0].get_header("Authorization"), "Bearer test-key")
        self.assertTrue(first.startswith("oss://dashscope-instant/test/"))
        self.assertTrue(first.endswith(".mp4"))
        form = self.forms[0]
        self.assertIn(b"original-video-bytes", form)
        self.assertIn(b"private", form)
        self.assertGreater(form.index(b'name="file"'), form.index(b'name="success_action_status"'))

    def test_cache_invalidated_by_file_model_credential_and_expiry(self):
        with patch.object(qwen_upload.urllib.request, "urlopen", side_effect=self.response):
            first = self.upload()
            self.path.write_bytes(b"changed-video")
            self.assertNotEqual(self.upload(), first)
            self.upload(model="different-model")
            self.upload(api_key="different-key")
            with patch.object(qwen_upload.time, "monotonic", return_value=time.monotonic() + 49 * 3600):
                self.upload()
        self.assertEqual(len(self.requests), 10)

    def test_policy_size_limit_blocks_upload(self):
        policy = {**POLICY, "max_file_size_mb": "0"}
        with patch.object(qwen_upload.urllib.request, "urlopen", return_value=Response(json.dumps({"data": policy}).encode())) as call:
            with self.assertRaisesRegex(RuntimeError, "file limit"):
                self.upload()
            self.assertEqual(call.call_count, 1)
        self.assertFalse(qwen_upload._UPLOAD_CACHE)

    def test_upload_failure_is_not_cached(self):
        with patch.object(qwen_upload.urllib.request, "urlopen", side_effect=[
            Response(json.dumps({"data": POLICY}).encode()), OSError("upload interrupted")
        ]):
            with self.assertRaisesRegex(RuntimeError, "upload interrupted"):
                self.upload()
        self.assertFalse(qwen_upload._UPLOAD_CACHE)

    def test_custom_and_regional_upload_endpoint(self):
        self.assertEqual(qwen_upload.upload_policy_url("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
                         "https://dashscope-intl.aliyuncs.com/api/v1/uploads")
        with self.assertRaisesRegex(ValueError, "QWEN_UPLOAD_URL"):
            qwen_upload.upload_policy_url("https://custom.invalid/v1")
        with patch.object(qwen_upload.urllib.request, "urlopen", side_effect=self.response):
            self.upload(base_url="https://custom.invalid/v1", upload_url="https://policy.invalid/api/v1/uploads")
        self.assertTrue(self.requests[0].full_url.startswith("https://policy.invalid/"))
