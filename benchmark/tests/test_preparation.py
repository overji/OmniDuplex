"""Preparation must preserve source data and resolve legacy/split media safely."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.runtime.preparation import checkpoint_status, media_aliases


class PreparationTests(unittest.TestCase):
    def test_split_uuid_and_original_names_resolve_without_copying_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / 'videos'
            videos.mkdir()
            (videos / 'uuid.mp4').write_bytes(b'video')
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps({'videos': [{'file': 'uuid.mp4', 'original_file': 'old.mp4'}]}))
            rows = [{'video': 'A/videos/uuid.mp4'}, {'video': 'old.mp4'}, {'video': 'missing.mp4'}]
            result = media_aliases(rows, videos, metadata, root / 'aliases')
            self.assertEqual(result['missing_qa'], 1)
            self.assertEqual(result['unique_resolved_media'], 1)
            self.assertEqual(result['missing_videos'], ['missing.mp4'])
            self.assertTrue((root / 'aliases/A/videos/uuid.mp4').is_symlink())
            self.assertEqual((root / 'aliases/old.mp4').resolve(), (videos / 'uuid.mp4').resolve())
            self.assertEqual(rows[0]['video'], 'A/videos/uuid.mp4')

    def test_ambiguous_legacy_video_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('one.mp4', 'two.mp4'):
                (root / name).write_bytes(b'video')
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps({'videos': [{'file': n, 'original_file': 'old.mp4'} for n in ('one.mp4', 'two.mp4')]}))
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                media_aliases([{'video': 'old.mp4'}], root, metadata, root / 'aliases')

    def test_nested_component_weight_shards_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            (root / 'llm').mkdir()
            (root / 'llm/model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'missing.safetensors'}}))
            status = checkpoint_status(root)
            self.assertTrue(any('missing.safetensors' in issue for issue in status['issues']))


if __name__ == '__main__':
    unittest.main()
