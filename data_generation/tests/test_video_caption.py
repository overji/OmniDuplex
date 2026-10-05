from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_generation.src.infrastructure import video as video_caption


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
            with patch("data_generation.src.infrastructure.video.subprocess.run", return_value=result):
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
            with patch("data_generation.src.infrastructure.video.subprocess.run", return_value=result):
                duration = video_caption.probe_video_duration(Path(handle.name))
        self.assertEqual(duration, 5.25)
