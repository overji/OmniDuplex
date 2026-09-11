from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src import video_caption


class DurationTests(unittest.TestCase):
    def test_format_video_duration_rounds_and_carries(self) -> None:
        self.assertEqual(video_caption.format_video_duration(0), "00:00:00.000")
        self.assertEqual(
            video_caption.format_video_duration(3723.4564),
            "01:02:03.456",
        )
        self.assertEqual(
            video_caption.format_video_duration(59.9996),
            "00:01:00.000",
        )

    def test_probe_prefers_container_duration(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            result = unittest.mock.Mock(
                stdout=json.dumps(
                    {
                        "format": {"duration": "12.345"},
                        "streams": [{"duration": "12.000"}],
                    }
                )
            )
            with patch("src.video_caption.subprocess.run", return_value=result):
                duration = video_caption.probe_video_duration(Path(handle.name))
        self.assertEqual(duration, 12.345)

    def test_probe_uses_longest_stream_duration_as_fallback(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            result = unittest.mock.Mock(
                stdout=json.dumps(
                    {
                        "format": {},
                        "streams": [
                            {"duration": "4.0"},
                            {"duration": "5.25"},
                        ],
                    }
                )
            )
            with patch("src.video_caption.subprocess.run", return_value=result):
                duration = video_caption.probe_video_duration(Path(handle.name))
        self.assertEqual(duration, 5.25)

    def test_video_to_data_url_uses_detected_mime_type(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            handle.write(b"video-bytes")
            handle.flush()
            data_url = video_caption.video_to_data_url(Path(handle.name))
        self.assertEqual(data_url, "data:video/mp4;base64,dmlkZW8tYnl0ZXM=")


class PromptTests(unittest.TestCase):
    def test_build_renders_duration_into_both_prompts(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            with patch(
                "src.video_caption.probe_video_duration",
                return_value=65.125,
            ):
                bundle = video_caption.build_video_prompts(Path(handle.name))

        self.assertEqual(bundle.duration, "00:01:05.125")
        for prompt in (bundle.events_prompt, bundle.objects_prompt):
            self.assertIn("00:01:05.125", prompt)
            self.assertIn("65.125 seconds", prompt)
            self.assertNotIn("{{VIDEO_DURATION", prompt)

    def test_cli_writes_both_prompt_files(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_handle:
            with tempfile.TemporaryDirectory() as output_dir:
                with patch(
                    "src.video_caption.probe_video_duration",
                    return_value=8.5,
                ):
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        status = video_caption.main(
                            [video_handle.name, "-o", output_dir]
                        )
                events = (Path(output_dir) / "events.prompt.md").read_text(
                    encoding="utf-8"
                )
                objects = (Path(output_dir) / "objects.prompt.md").read_text(
                    encoding="utf-8"
                )

        self.assertEqual(status, 0)
        self.assertIn("00:00:08.500", events)
        self.assertIn("00:00:08.500", objects)
        self.assertIn("video duration: 00:00:08.500", stdout.getvalue())

    def test_objects_prompt_injects_event_metadata(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_handle:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as events_handle:
                events_handle.write(json.dumps({
                    "video_summary": {"main_content_summary": "workout"},
                    "audio_events": [{"event_id": "A1"}],
                    "visual_events": [{"event_id": "V1"}],
                }))
                events_handle.flush()
                with patch("src.video_caption.probe_video_duration", return_value=8.5):
                    bundle = video_caption.build_video_prompts(
                        Path(video_handle.name), events_json=Path(events_handle.name)
                    )
        self.assertIn('"main_content_summary": "workout"', bundle.objects_prompt)
        self.assertIn('"event_id": "A1"', bundle.objects_prompt)
        self.assertIn('"event_id": "V1"', bundle.objects_prompt)
        self.assertNotIn("{{VIDEO_", bundle.objects_prompt)

    def test_events_json_requires_fields(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video_handle:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as events_handle:
                events_handle.write("{}")
                events_handle.flush()
                with patch("src.video_caption.probe_video_duration", return_value=1.0):
                    with self.assertRaisesRegex(ValueError, "missing required fields"):
                        video_caption.build_video_prompts(
                            Path(video_handle.name), events_json=Path(events_handle.name)
                        )

    def test_cli_reports_missing_video(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = video_caption.main(["missing-video.mp4"])
        self.assertEqual(status, 1)
        self.assertIn("video file does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
