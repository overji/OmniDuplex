"""Directory pipeline and canonical historical QA counts."""
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from . import batch
from ..domain.catalog import TASK_CATALOG
from ..domain.common import SAMPLE_SCHEMA
from ..domain.qa import is_agent_negative
from ..domain.config import validate_probability, validate_agent_positive_probability
from ..infrastructure.paths import DATA_ROOT


def historical_counts(output_parent, *, exclude=None):
    counts = Counter()
    for folder in sorted(output_parent.iterdir()) if output_parent.is_dir() else []:
        if not folder.is_dir() or folder.resolve() == exclude:
            continue
        if not ((folder / ".state/manifest.json").is_file() or (folder / "manifest.json").is_file()):
            continue
        source = folder / "all_qa.json"
        if not source.is_file():
            continue
        samples = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(samples, list):
            raise ValueError(f"invalid historical QA bundle: {source}")
        for sample in samples:
            if isinstance(sample, dict) and not is_agent_negative(sample) and sample.get("schema") == SAMPLE_SCHEMA and sample.get("status") == "ok" and sample.get("task") in TASK_CATALOG:
                counts[sample["task"]] += 1
    return dict(counts)


def source_videos(args):
    if args.video is not None:
        return [args.video]
    if not args.videos_dir.is_dir():
        raise ValueError(f"video directory does not exist: {args.videos_dir}")
    videos = sorted(p for p in args.videos_dir.rglob('*') if p.is_file()
                    and p.suffix.lower() in {'.mp4', '.mov', '.mkv', '.webm', '.avi'})
    if args.split is not None:
        metadata = json.loads((args.videos_dir.parent / "video_metadata.json").read_text(encoding="utf-8"))
        listed = [row["file"] for row in metadata["videos"]]
        if (metadata["video_count"] != len(listed) or len(set(listed)) != len(listed)
                or set(listed) != {video.name for video in videos}):
            raise ValueError(f"split {args.split} videos do not match video_metadata.json")
    return videos


def generated_video_names(*roots):
    """Collect videos with saved, accepted QA data; ignore attempts and empty bundles."""
    names = set()
    seen = set()
    for root in roots:
        root = Path(root).resolve()
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        for path in root.rglob("*.json"):
            if path.name != "all_qa.json" and path.parent.parent != root:
                continue
            if any(part in {"attempts", "aborted_attempts", "workers", ".state"}
                   for part in path.relative_to(root).parts[:-1]):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            samples = data if isinstance(data, list) else [data]
            for sample in samples:
                if (isinstance(sample, dict) and not is_agent_negative(sample) and sample.get("schema") == SAMPLE_SCHEMA
                        and sample.get("status") == "ok" and isinstance(sample.get("video"), str)
                        and isinstance(sample.get("responses"), list) and sample["responses"]):
                    names.add(Path(sample["video"]).as_posix())
    return names


def resume_videos(args):
    manifest = args.output_dir / ".state" / "manifest.json"
    if not manifest.is_file():
        raise ValueError("resume requires --output-dir pointing to a run with a manifest")
    settings = json.loads(manifest.read_text(encoding="utf-8")).get("pipeline_settings", {})
    sources = settings.get("sources")
    if not isinstance(sources, list):
        raise ValueError("resume manifest has no pipeline sources")
    return [args.videos_dir / relative for relative, _ in sources]


def video_urls(args, videos):
    mapping = json.loads(args.video_url_map.read_text()) if args.video_url_map else {}
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise ValueError('video URL map must map relative video paths to URL strings')
    if args.provider == 'gemini' and mapping:
        raise ValueError('Gemini uploads local videos; omit --video-url-map')
    for video in videos:
        relative = video.relative_to(args.videos_dir).as_posix()
        url = mapping.get(relative)
        if args.provider == 'openai' and url is None:
            raise ValueError(f'OpenAI-compatible pipeline requires a video URL mapping for {relative}')
        if url is not None and (urlsplit(url).scheme not in {'http', 'https'} or not urlsplit(url).netloc):
            raise ValueError(f'invalid remote video URL for {relative}')
    return mapping


def run(args, *, progress=None):
    if args.video is not None and not args.video.is_file():
        raise ValueError(f"video does not exist: {args.video}")
    for name in ("pr_probability", "agent_probability", "agent_positive_probability"):
        validate_probability(name, getattr(args, name))
    validate_agent_positive_probability(args.agent_positive_probability)
    if args.num_attempts is not None and args.num_attempts < 1:
        raise ValueError('num-attempts must be positive')
    videos = source_videos(args)
    if not videos:
        raise ValueError('no source videos found')
    if args.resume:
        videos = resume_videos(args)
    else:
        existing = generated_video_names(DATA_ROOT, args.output_dir)
        pending = [video for video in videos if video.relative_to(args.videos_dir).as_posix() not in existing]
        if progress is not None:
            progress(f"Skipping {len(videos) - len(pending)} videos with saved QA data; {len(pending)} remaining.", False)
        videos = pending
        if not videos:
            return 0
    if not args.api_key or not args.model:
        raise ValueError("pipeline requires an API key and model; configure the provider environment or CLI arguments")
    video_urls(args, videos)  # Validate all source mappings before any billable call.
    from ..infrastructure.gemini import video_session
    with video_session():
        return batch.run(args, progress=progress, videos=videos)
