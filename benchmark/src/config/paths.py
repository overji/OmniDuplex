"""Stable paths independent of the caller's working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
MANIFESTS = CONFIGS / "manifests"
SYSTEM_PROMPT = CONFIGS / "prompts/system_prompt.txt"
DATA = Path("/remote-home/yushengliao/speech/jimodels/datasets/AOI-Bench/all_qa.json")
VIDEOS = ROOT / "cache/aoi_bench_videos"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
MODEL_ROOT = Path("/remote-home/yushengliao/speech/jimodels/openmodels")
