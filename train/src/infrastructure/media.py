"""带真实PTS的媒体输入及CPU预处理缓存。不缓存全参编码器输出，避免切断媒体梯度。"""

import bisect
import copy
import math
from contextlib import contextmanager
from pathlib import Path
from benchmark.data.timestamps import seconds
from benchmark.data.media import media_cache


class Clip:
    """Only raw/preprocessed inputs are cached; encoder outputs are never cached."""
    def __init__(self, record, config, horizon):
        import numpy as np
        from PIL import Image
        cache, metadata, audio = media_cache(Path(config['videos_dir']) / record['sample']['video'], Path(config['media_cache']))
        self.horizon = math.floor(metadata['duration']) if horizon is None else min(horizon, math.floor(metadata['duration']))
        self.duration = metadata['duration']
        self.chunks, self.processed = [], {}
        self.question_time = seconds(record['sample']['question_time'])
        for second in range(1, self.horizon + 1):
            fi = bisect.bisect_left(metadata['frame_times'], second) - 1
            frames = []
            if fi >= 0:
                with Image.open(cache / metadata['frames'][fi]) as image:
                    frames = [image.convert('RGB')]
            chunk = audio[(second - 1) * 16000:second * 16000]
            if len(chunk) < 16000:
                chunk = np.pad(chunk, (0, 16000 - len(chunk)))
            self.chunks.append(dict(audio_waveform=chunk, frame_list=frames,
                                    max_slice_nums=1, batch_vision_feed=False))

    @contextmanager
    def preprocessing_cache(self, processor):
        """Cache per-call CPU BatchFeatures; .to() mutates them, so return copies."""
        names = ('process_image', 'process_audio_streaming')
        originals = {name: getattr(processor, name) for name in names}
        counters = {name: 0 for name in names}
        def wrap(name):
            def call(*args, **kwargs):
                key = (name, counters[name]); counters[name] += 1
                if key not in self.processed:
                    feature = copy.deepcopy(originals[name](*args, **kwargs))
                    snapshot = (copy.deepcopy(processor.get_streaming_snapshot())
                                if name == 'process_audio_streaming' else None)
                    self.processed[key] = (feature, snapshot)
                feature, snapshot = self.processed[key]
                if snapshot is not None:
                    processor.restore_streaming_snapshot(copy.deepcopy(snapshot))
                return copy.deepcopy(feature)
            return call
        try:
            for name in names:
                setattr(processor, name, wrap(name))
            yield
        finally:
            for name, fn in originals.items():
                setattr(processor, name, fn)
