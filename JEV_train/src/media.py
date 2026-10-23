from dataclasses import dataclass
import bisect
from pathlib import Path


@dataclass
class MediaUnit:
    second: int
    frame: object = None
    audio: object = None


def units(row, config):
    from PIL import Image
    import numpy as np
    from benchmark.data.media import media_cache
    cache, meta, audio = media_cache(Path(config['videos_dir']) / row['sample']['video'], Path(config['cache_dir']) / 'media')
    for t in range(1, row['horizon'] + 1):
        fi = bisect.bisect_left(meta['frame_times'], t) - 1
        frame = None
        if fi >= 0:
            with Image.open(cache / meta['frames'][fi]) as image:
                frame = image.convert('RGB').copy()
        chunk = None
        if meta['has_audio']:
            chunk = audio[(t-1)*16000:t*16000]
            chunk = np.pad(chunk, (0, max(0, 16000-len(chunk))))
        yield MediaUnit(t, frame, chunk)
