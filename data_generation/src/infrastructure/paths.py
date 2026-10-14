"""Stable resource locations, independent of command working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parent
PROMPT_ROOT = ROOT / "prompts"
DATA_ROOT = ROOT / "data"
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"
