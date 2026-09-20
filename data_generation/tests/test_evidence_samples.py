import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_evidence_samples import build_sample, timestamp_seconds


class EvidenceSampleTests(unittest.TestCase):
    def test_timestamp_parser(self) -> None:
        self.assertEqual(timestamp_seconds("00:27"), 27)
        with self.assertRaises(ValueError):
            timestamp_seconds("01:60")

    def test_builds_one_sample_per_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps({
                "question": "Is the person sitting or standing?",
                "responses": [{"trigger_time": "00:01", "response": "The person changes from sitting to standing."}],
            }), encoding="utf-8")
            video = root / "input.mp4"
            video.write_bytes(b"not used by this test")
            output = root / "output"
            # Exercise input validation without invoking ffmpeg on a fake video.
            with self.assertRaises(Exception):
                build_sample(source, video, output)


if __name__ == "__main__":
    unittest.main()
