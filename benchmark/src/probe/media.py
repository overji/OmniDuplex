"""Strict audiovisual prefix clipping with a reusable source-aware cache."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from benchmark.data.files import write_json


def source_identity(video):
    stat = video.stat()
    return {'path': str(video.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def scan_video(video):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                                 '-of', 'json', str(video)], check=True, capture_output=True, text=True)
        body = json.loads(result.stdout)
        duration = float(body['format']['duration'])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Video duration must be positive and finite')
        if not any(stream['codec_type'] == 'video' for stream in body['streams']):
            raise ValueError('No video stream')
        return {'duration': duration,
                'has_audio': any(stream['codec_type'] == 'audio' for stream in body['streams'])}
    except subprocess.CalledProcessError as exc:
        raise ValueError(f'Cannot probe video {video}: {exc.stderr}') from exc


def prefix_command(video, output, cutoff, has_audio):
    command = ['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-copyts', '-start_at_zero', '-i', str(video),
               '-map', '0:v:0', '-vf', f'trim=start=0:end={cutoff:.9f}',
               '-c:v', 'libx264', '-preset', 'fast', '-vsync', '0']
    if has_audio:
        command += ['-map', '0:a:0', '-af', f'atrim=start=0:end={cutoff:.9f}',
                    '-c:a', 'aac']
    else:
        command += ['-an']
    return command + ['-t', f'{cutoff:.9f}', '-movflags', '+faststart', str(output)]


def validate_prefix(path, cutoff, has_audio):
    info = scan_video(path)
    if info['has_audio'] != has_audio:
        raise ValueError('Prefix audio stream does not match source')
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True)
    times = [float(frame['best_effort_timestamp_time'])
             for frame in json.loads(result.stdout)['frames'] if 'best_effort_timestamp_time' in frame]
    if not times or min(times) < 0 or max(times) >= cutoff:
        raise ValueError('Prefix frame timestamps cross the cutoff')
    return info


def prefix_clip(row, cache_dir):
    from benchmark.probe.data import digest
    video = Path(row['video'])
    if source_identity(video) != row['source']:
        raise ValueError('Source video changed after probe preparation')
    identity = {'source': row['source'], 'cutoff': row['prefix_end_seconds'],
                'has_audio': row['has_audio'], 'policy': 'strict_trim_h264_aac_v1',
                'command': prefix_command(video, Path('OUTPUT.mp4'), row['prefix_end_seconds'], row['has_audio'])}
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = digest(identity)
    output = cache_dir / (key + '.mp4')
    metadata = cache_dir / (key + '.json')
    with (cache_dir / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.is_file() and metadata.is_file():
            try:
                saved = json.loads(metadata.read_text())
                if saved['identity'] == identity and saved['clip'] == source_identity(output):
                    return output
            except (OSError, ValueError, KeyError):
                pass
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.mp4', delete=False) as temporary:
            staging = Path(temporary.name)
        try:
            subprocess.run(prefix_command(video, staging, row['prefix_end_seconds'], row['has_audio']),
                           check=True, capture_output=True, text=True)
            validate_prefix(staging, row['prefix_end_seconds'], row['has_audio'])
            os.replace(staging, output)
            write_json(metadata, {'identity': identity, 'clip': source_identity(output)})
        finally:
            staging.unlink(missing_ok=True)
    return output
