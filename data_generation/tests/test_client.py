from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.client import build_user_content


class UserContentTests(unittest.TestCase):
    def test_build_user_content_contains_video_url_and_text(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            handle.write(b"sample")
            handle.flush()
            content = build_user_content("evidence", Path(handle.name))

        self.assertEqual(content[0]["type"], "video_url")
        video_url = content[0]["video_url"]
        self.assertIsInstance(video_url, dict)
        self.assertTrue(video_url["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual(content[1], {"type": "text", "text": "evidence"})


if __name__ == "__main__":
    unittest.main()
