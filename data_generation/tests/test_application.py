"""Application contracts independent of command-line argument objects."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from data_generation.src.application import generation
from data_generation.src.application.options import GenerationOptions
from data_generation.src.domain.caption import VideoPromptBundle
from data_generation.src.infrastructure.paths import DEFAULT_ENV_FILE, PROMPT_ROOT, ROOT
from data_generation.tests.fixtures import caption


class GenerationServiceTests(unittest.TestCase):
    def options(self, output):
        return GenerationOptions(
            provider="openai", model="test", base_url="https://example.invalid/v1",
            api_key="test", upload_url=None, timeout=120, retries=2,
            task="caption", video=Path("video.mp4"), usage_output=output,
        )

    def test_returns_sample_and_usage_without_mutating_inputs_or_printing(self):
        with tempfile.TemporaryDirectory() as directory:
            options = self.options(Path(directory) / "caption.usage.json")
            original = copy.deepcopy(options)
            request = VideoPromptBundle(options.video, 60, "01:00", "caption prompt")
            usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

            def respond(*args, **kwargs):
                kwargs["usage_callback"](usage)
                return json.dumps(caption(60))

            output = io.StringIO()
            with patch.object(generation, "chat_completion", side_effect=respond), redirect_stdout(output):
                result = generation.generate(options, request)
            self.assertEqual(result.sample, caption(60))
            self.assertEqual(result.usage_record["total_tokens"], 10)
            self.assertEqual(json.loads(options.usage_output.read_text()), result.usage_record)
            self.assertEqual(options, original)
            self.assertFalse(hasattr(options, "usage_record"))
            self.assertEqual(output.getvalue(), "")

    def test_usage_is_saved_even_when_model_output_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            options = self.options(Path(directory) / "caption.usage.json")
            request = VideoPromptBundle(options.video, 60, "01:00", "caption prompt")
            with patch.object(generation, "chat_completion", return_value="{}"):
                with self.assertRaises(ValueError):
                    generation.generate(options, request)
            self.assertEqual(json.loads(options.usage_output.read_text())["status"], "unavailable")

    def test_default_resources_still_point_to_original_locations(self):
        project = Path(__file__).resolve().parents[1]
        self.assertEqual(ROOT, project)
        self.assertEqual(PROMPT_ROOT, project / "prompts")
        self.assertEqual(DEFAULT_ENV_FILE, project.parent / ".env")
