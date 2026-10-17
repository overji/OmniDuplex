import copy
import json
import tempfile
import unittest
from pathlib import Path

from data_generation.src.application.merge_qa import build_bundle
from data_generation.tests.fixtures import saved_agent


class MergeQaTests(unittest.TestCase):
    def test_merges_positive_and_excludes_both_refusals_without_mutating_sources(self):
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
            self.assertEqual(len(bundle), 1)
            for source, merged in zip(samples, bundle):
                for key in ("frontend_context", "sample_type", "available_tools", "required_tools", "generation_context"):
                    self.assertEqual(json.loads(json.dumps(source[key])), merged[key])
            self.assertEqual([r["role"] for r in bundle[0]["responses"]], ["assistant", "tool", "assistant"])
            self.assertEqual([r["trigger_time"] for r in bundle[0]["responses"]], ["00:20", "00:40", "00:41"])
            self.assertEqual(bundle[0]["responses"][2]["window_start"], "00:41")
            self.assertEqual(bundle[0]["responses"][2]["agent_result_time"], "00:40")
            self.assertEqual(bundle[0]["responses"][0]["source_trigger_time"], "00:20")
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

    def test_legacy_files_merge_without_synthesizing_windows_but_new_generation_rejects_them(self):
        from data_generation.src.application.registry import get_task
        for kind in ('positive', 'missing_tool', 'no_context'):
            sample = saved_agent(sample_type=kind)
            if kind == 'positive':
                final = sample['responses'][0]['messages'][2]
                final.pop('window_start')
                final.pop('window_end')
                final['timestamp'] = '00:40'
            else:
                sample['responses'][0]['window_end'] = '00:00'
            task = get_task(sample['task'])
            context = task.GenerationContext(**sample['generation_context'])
            with self.assertRaises(ValueError):
                task.validate_result(sample, context)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder = root / 'test_video'
                folder.mkdir()
                (folder / 'sample.json').write_text(json.dumps(sample))
                bundle = build_bundle(root, root / 'videos')
                if kind == 'positive':
                    self.assertNotIn('window_start', bundle[0]['responses'][2])
                else:
                    self.assertEqual(bundle, [])

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
