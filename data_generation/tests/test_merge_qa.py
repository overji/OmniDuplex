import copy
import json
import tempfile
import unittest
from pathlib import Path

from data_generation.src.application.merge_qa import build_bundle
from data_generation.tests.fixtures import saved_agent


class MergeQaTests(unittest.TestCase):
    def test_merges_positive_and_both_refusals_without_losing_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "data" / "test_video"
            folder.mkdir(parents=True)
            samples = [saved_agent(sample_type=kind) for kind in ("positive", "no_context", "missing_tool")]
            for index, sample in enumerate(samples):
                (folder / f"sample{index}.json").write_text(json.dumps(sample))
            (folder / "caption.json").write_text('[]')
            (folder / "route.json").write_text(json.dumps({"schema": "dense-caption-v1", "task": "route", "status": "ok"}))
            bundle = build_bundle(root / "data", root / "videos")
            self.assertEqual(len(bundle), 3)
            for source, merged in zip(samples, bundle):
                for key in ("frontend_context", "sample_type", "available_tools", "required_tools", "generation_context"):
                    self.assertEqual(json.loads(json.dumps(source[key])), merged[key])
            self.assertEqual([r["role"] for r in bundle[0]["responses"]], ["assistant", "tool", "assistant"])
            self.assertEqual([r["trigger_time"] for r in bundle[0]["responses"]], ["00:20", "00:40", "00:40"])
            self.assertEqual(bundle[0]["responses"][0]["source_trigger_time"], "00:20")
            for index in (1, 2):
                self.assertEqual(bundle[index]["responses"], samples[index]["responses"])
            for index, original in enumerate(samples):
                self.assertEqual(json.loads((folder / f"sample{index}.json").read_text()), json.loads(json.dumps(original)))

    def test_rejects_old_schema_or_tampered_capabilities(self):
        valid = saved_agent()
        tampered = copy.deepcopy(valid)
        tampered["available_tools"] = ["email_send"]
        for sample in ({"status": "ok", "question": "Old sample", "responses": []}, tampered):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder = root / "test_video"
                folder.mkdir()
                (folder / "sample.json").write_text(json.dumps(sample))
                with self.assertRaises(ValueError):
                    build_bundle(root, root / "videos")

    def test_missing_video_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "test_video"
            folder.mkdir()
            sample = saved_agent()
            del sample["video"]
            (folder / "sample.json").write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError, "no video found"):
                build_bundle(root, root / "videos")

    def test_nested_layout_ignores_summaries_state_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "test_video"
            qa = folder / "qa"
            qa.mkdir(parents=True)
            (root / ".state").mkdir()
            (root / ".state" / "manifest.json").write_text('{}')
            (folder / "caption.json").write_text('[]')
            (folder / "all_qa.json").write_text('[{}]')
            (qa / "sample.json").write_text(json.dumps(saved_agent()))
            (qa / "sample.usage.json").write_text('{}')
            bundle = build_bundle(root, root / "videos")
            self.assertEqual(len(bundle), 1)
            self.assertEqual(bundle[0]["video_folder"], "test_video")

    def test_legacy_layout_ignores_summaries_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "test_video"
            folder.mkdir()
            (folder / "sample.json").write_text(json.dumps(saved_agent()))
            (folder / "sample.usage.json").write_text('{}')
            (folder / "all_qa.json").write_text('[{}]')
            self.assertEqual(len(build_bundle(root, root / "videos")), 1)
