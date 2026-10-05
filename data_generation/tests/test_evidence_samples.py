import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_evidence_samples import build_sample, timestamp_seconds
from data_generation.tests.fixtures import saved_agent
from data_generation.src.domain.qa import adapt_responses


class EvidenceSampleTests(unittest.TestCase):
    def test_timestamp_parser(self):
        self.assertEqual(timestamp_seconds("00:27"), 27)
        with self.assertRaises(ValueError):
            timestamp_seconds("01:60")

    def test_builds_one_sample_per_factual_response(self):
        with tempfile.TemporaryDirectory() as directory, patch("scripts.build_evidence_samples.subprocess.run") as run:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps({
                "question": "Is the person sitting or standing?",
                "responses": [{"trigger_time": "00:01", "response": "The person changes from sitting to standing."}],
            }))
            build_sample(source, root / "input.mp4", root / "out")
            run.assert_called_once()
            self.assertIn("2", run.call_args.args[0])
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(len(manifest["responses"]), 1)
            self.assertEqual(manifest["responses"][0]["clip_end_exclusive"], "00:02")

    def test_refusals_and_agent_results_are_excluded_before_ffmpeg(self):
        samples = [saved_agent(sample_type=kind) for kind in ("positive", "no_context", "missing_tool")]
        merged = saved_agent()
        merged["responses"] = adapt_responses(merged["responses"])
        samples.append(merged)
        with tempfile.TemporaryDirectory() as directory, patch("scripts.build_evidence_samples.subprocess.run") as run:
            root = Path(directory)
            for sample in samples:
                source = root / "sample.json"
                source.write_text(json.dumps(sample))
                with self.assertRaises(ValueError):
                    build_sample(source, root / "video.mp4", root / "out")
            run.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_predictions_and_inferences_excluded_before_ffmpeg(self):
        with tempfile.TemporaryDirectory() as directory, patch("scripts.build_evidence_samples.subprocess.run") as run:
            root = Path(directory)
            for ability in ("prospective_reasoning", "causal_reasoning", "emotion_recognition"):
                source = root / "sample.json"
                source.write_text(json.dumps({"ability": ability, "question": "Monitor changes.",
                                             "responses": [{"trigger_time": "00:01", "response": "An inference."}]}))
                with self.assertRaisesRegex(ValueError, "not observed"):
                    build_sample(source, root / "video.mp4", root / "out")
            run.assert_not_called()
