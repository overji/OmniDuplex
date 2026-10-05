from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_generation.src.infrastructure.client import build_user_content


class UserContentTests(unittest.TestCase):
    def test_build_user_content_contains_video_url_and_text(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            handle.write(b"sample")
            handle.flush()
            content = build_user_content("evidence", Path(handle.name), video_url="https://media.example/video.mp4")

        self.assertEqual(content[0]["type"], "video_url")
        video_url = content[0]["video_url"]
        self.assertIsInstance(video_url, dict)
        self.assertEqual(video_url["url"], "https://media.example/video.mp4")
        self.assertEqual(content[1], {"type": "text", "text": "evidence"})


if __name__ == "__main__":
    unittest.main()
