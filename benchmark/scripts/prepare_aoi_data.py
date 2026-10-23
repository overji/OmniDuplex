"""Validate AOI sources and prepare local video aliases without changing QA files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.config.paths import DATA, VIDEOS, ROOT
from benchmark.data.files import write_json
from benchmark.runtime.preparation import media_aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    train_config = json.loads((ROOT.parent / 'train/configs/minicpmo_lora.json').read_text())
    parser.add_argument('--dataset-dir', type=Path, default=Path(train_config['data']).parent)
    parser.add_argument('--bench-dir', type=Path, default=DATA.parent)
    parser.add_argument('--output-dir', type=Path, default=VIDEOS)
    args = parser.parse_args()
    training = json.loads((args.dataset_dir / 'all_qa.json').read_text())
    missing = sorted({r['video'] for r in training if not (args.dataset_dir / r['video']).is_file()})
    if missing:
        raise ValueError(f'AOI-Dataset missing {len(missing)} videos: {missing[:5]}')
    rows = json.loads((args.bench_dir / 'all_qa.json').read_text())
    media = media_aliases(rows, args.bench_dir / 'videos', args.bench_dir / 'video_manifest.json', args.output_dir)
    report = {
        'training': {'root': str(args.dataset_dir.resolve()), 'qa': len(training),
                     'unique_videos': len({r['video'] for r in training}), 'missing_videos': missing},
        'benchmark': {'root': str(args.bench_dir.resolve()), 'qa': len(rows),
                      'alias_root': str(args.output_dir.resolve()), **media},
        'source_sha256': {name: hashlib.sha256((root / 'all_qa.json').read_bytes()).hexdigest()
                          for name, root in [('training', args.dataset_dir), ('benchmark', args.bench_dir)]},
    }
    write_json(args.output_dir / 'validation.json', report)
    if media['missing_videos']:
        raise ValueError(f"AOI-Bench missing {len(media['missing_videos'])} videos; see validation.json")
    print(json.dumps({k: {key: value for key, value in v.items() if key != 'aliases'}
                      for k, v in report.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
