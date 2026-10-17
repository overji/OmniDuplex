import json
from pathlib import Path
import tempfile
import unittest

from data_generation.scripts.remove_agent_negatives import clean
from data_generation.src.application.registry import get_task
from data_generation.src.application.pipeline import generated_video_names, historical_counts
from data_generation.src.cli.pipeline import parse_args
from data_generation.src.domain.config import choose_agent_sample_type
from data_generation.src.domain.qa import is_agent_negative
from data_generation.tests.fixtures import saved_agent


class PositiveAgentTests(unittest.TestCase):
    def test_direct_generation_cannot_request_negatives(self):
        task = get_task("object_perception_agent")
        for branch in ("no_context", "missing_tool"):
            with self.assertRaisesRegex(ValueError, "disabled"):
                task.build_request(Path("video.mp4"), 100, [], "", "", sample_type=branch)
        for _ in range(100):
            self.assertEqual(choose_agent_sample_type(), "positive")
            request = task.build_request(Path("video.mp4"), 100, [], "", "")
            self.assertEqual(request.context.sample_type, "positive")
            self.assertIsNone(request.context.required_missing_tool)
            self.assertTrue(request.context.available_tools)

    def test_pipeline_rejects_old_negative_probability(self):
        with self.assertRaises(SystemExit):
            parse_args(["--video", "video.mp4", "--output-dir", "out",
                        "--agent-positive-probability", ".7"])

    def test_negative_only_video_is_not_completed_or_counted_for_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text('{"artifacts": {}}')
            bundle = run / "all_qa.json"
            bundle.write_text(json.dumps([saved_agent(sample_type="no_context")]))
            self.assertEqual(generated_video_names(root), set())
            self.assertEqual(historical_counts(root), {})
            bundle.write_text(json.dumps([saved_agent()]))
            self.assertTrue(generated_video_names(root))
            self.assertEqual(sum(historical_counts(root).values()), 1)

    def test_cleanup_preserves_other_samples_backups_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data"
            source.mkdir()
            positive = saved_agent()
            negatives = [saved_agent(sample_type=branch) for branch in ("no_context", "missing_tool")]
            # The context flag alone is sufficient for legacy source files.
            negatives[0].pop("sample_type")
            other = {"task": "object_perception_pr", "sample_type": "negative"}
            bundle = source / "all_qa.json"
            original = json.dumps([positive, *negatives, other])
            bundle.write_text(original)
            raw = source / "negative.json"
            raw.write_text(json.dumps(negatives[1]))
            changes = clean([source])
            self.assertEqual(sum(c["removed"] for c in changes), 3)
            self.assertEqual(bundle.read_text(), original)
            self.assertTrue(raw.exists())
            backup = root / "backup"
            clean([source], apply=True, backup_dir=backup)
            self.assertEqual(json.loads(bundle.read_text()), [json.loads(json.dumps(positive)), other])
            self.assertFalse(raw.exists())
            copied = backup / bundle.resolve().relative_to(bundle.anchor)
            self.assertEqual(copied.read_text(), original)
            self.assertEqual(clean([source], apply=True, backup_dir=backup), [])
            self.assertFalse(is_agent_negative(other))
