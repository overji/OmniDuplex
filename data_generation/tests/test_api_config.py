import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_generation.src.cli import batch, generate as cli
from data_generation.src.cli.api_config import QWEN_BASE_URL, QWEN_MODEL


class APIConfigTests(unittest.TestCase):
    def test_qwen_dotenv_and_explicit_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text('QWEN_API_KEY="file-key"\nQWEN_URL=https://file.invalid/v1\nQWEN_MODEL=file-model\n')
            args = ["--task", "caption", "--video", "video.mp4", "--provider", "qwen", "--env-file", str(env)]
            with patch.dict(os.environ, {}, clear=True):
                parsed = cli.parse_args(args)
                self.assertEqual((parsed.api_key, parsed.base_url, parsed.model),
                                 ("file-key", "https://file.invalid/v1", "file-model"))
                self.assertNotIn("QWEN_API_KEY", os.environ)
            with patch.dict(os.environ, {"QWEN_API_KEY": "env-key", "QWEN_URL": "https://env.invalid/v1"}, clear=True):
                parsed = cli.parse_args(args)
                self.assertEqual((parsed.api_key, parsed.base_url), ("env-key", "https://env.invalid/v1"))
                explicit = cli.parse_args(args + ["--api-key", "arg-key", "--model", "arg-model", "--base-url", "https://arg.invalid/v1"])
                self.assertEqual((explicit.api_key, explicit.model, explicit.base_url),
                                 ("arg-key", "arg-model", "https://arg.invalid/v1"))

    def test_qwen_defaults_and_batch_match_single_task(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            env = Path(directory) / ".env"
            env.write_text("QWEN_API_KEY=file-key\n")
            shared = ["--provider", "qwen", "--env-file", str(env)]
            single = cli.parse_args(["--task", "caption", "--video", "video.mp4", *shared])
            multiple = batch.parse_args(["--videos-dir", directory, "--output-dir", directory, *shared])
            self.assertEqual(single.model, QWEN_MODEL)
            self.assertEqual(single.base_url, QWEN_BASE_URL)
            self.assertEqual((single.timeout, single.retries), (120.0, 2))
            for field in ("provider", "model", "base_url", "api_key", "timeout", "retries"):
                self.assertEqual(getattr(single, field), getattr(multiple, field))

    def test_openai_defaults_and_credentials_remain_separate(self):
        with patch.dict(os.environ, {"QWEN_API_KEY": "qwen-key"}, clear=True):
            parsed = cli.parse_args(["--task", "caption", "--video", "v.mp4"])
            self.assertEqual(parsed.provider, "openai")
            self.assertEqual(parsed.base_url, "https://api.openai.com/v1")
            self.assertIsNone(parsed.api_key)
            self.assertIsNone(parsed.model)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key", "OPENAI_MODEL": "existing-model",
                                    "OPENAI_BASE_URL": "https://existing.invalid/v1"}, clear=True):
            parsed = cli.parse_args(["--task", "caption", "--video", "v.mp4"])
            self.assertEqual((parsed.api_key, parsed.model, parsed.base_url),
                             ("openai-key", "existing-model", "https://existing.invalid/v1"))

    def test_qwen_does_not_fall_back_to_openai_key(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_API_KEY": "other-key"}, clear=True):
            args = cli.parse_args(["--task", "caption", "--video", "v.mp4", "--provider", "qwen",
                                   "--env-file", str(Path(directory) / "missing.env")])
            self.assertIsNone(args.api_key)
