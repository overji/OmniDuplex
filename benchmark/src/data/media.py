"""Shared timestamped media cache; model-specific preprocessing stays in adapters."""
import json
import math
import re
import subprocess
import fcntl

from benchmark.data.files import write_json


def media_cache(video, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / (video.stem + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _media_cache_unlocked(video, cache_dir)


def _media_cache_unlocked(video, cache_dir):
    """Extract all frames at approximately 1 Hz, keeping their actual PTS."""
    import numpy as np
    import soundfile as sf

    target = cache_dir / video.stem
    target.mkdir(parents=True, exist_ok=True)
    fingerprint = {"path": str(video.resolve()), "size": video.stat().st_size,
                   "mtime_ns": video.stat().st_mtime_ns}
    metadata_path = target / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["source"] != fingerprint:
            raise ValueError(f"Media cache is stale: {target}")
    else:
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video),
        ]))
        duration = float(probe["format"]["duration"])
        frame_dir = target / "frames"
        frame_dir.mkdir(exist_ok=True)
        extracted = subprocess.run([
            "ffmpeg", "-y", "-threads", "2", "-i", str(video), "-an", "-vf",
            "select='isnan(prev_selected_t)+gte(t-prev_selected_t,1)',showinfo",
            "-vsync", "0", str(frame_dir / "frame_%06d.jpg"),
        ], capture_output=True, text=True, check=True)
        frame_times = [float(value) for value in re.findall(r"\bn:\s*\d+.*?pts_time:([\d.e+-]+)", extracted.stderr)]
        frames = sorted(frame_dir.glob("frame_*.jpg"))
        if len(frames) != len(frame_times) or not frames:
            raise ValueError("Cannot align extracted frames with their timestamps")
        has_audio = any(stream["codec_type"] == "audio" for stream in probe["streams"])
        if has_audio:
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-threads", "2", "-i", str(video),
                "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(target / "audio.wav"),
            ], check=True)
        metadata = {"source": fingerprint, "duration": duration, "has_audio": has_audio,
                    "frame_times": frame_times, "frames": [str(frame.relative_to(target)) for frame in frames]}
        write_json(metadata_path, metadata)
    audio = sf.read(target / "audio.wav", dtype="float32")[0] if metadata["has_audio"] else np.zeros(
        math.ceil(metadata["duration"] * 16000), dtype=np.float32)
    return target, metadata, audio
