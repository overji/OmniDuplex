from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import video_caption


class DurationTests(unittest.TestCase):
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
