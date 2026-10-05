"""Shared CLI configuration for OpenAI-compatible providers."""

import os
from pathlib import Path


from ..infrastructure.paths import DEFAULT_ENV_FILE
QWEN_MODEL = "qwen3.8-omni-flash"
QWEN_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"


def add_api_arguments(parser):
    parser.add_argument("--provider", choices=("openai", "qwen"), default="openai")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE,
                        help="Qwen dotenv file (default: repository .env)")
    parser.add_argument("--model", help="model name")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", help="API key (prefer environment or .env)")
    parser.add_argument("--upload-url", help="Qwen upload policy endpoint (default: inferred from API region)")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)


def resolve_api_arguments(args):
    if args.provider == "qwen":
        from dotenv import dotenv_values

        # Read configuration without exporting secrets or overriding process env.
        environment = {**dotenv_values(args.env_file, interpolate=False), **os.environ}
        defaults = {
            "model": environment.get("QWEN_MODEL") or QWEN_MODEL,
            "base_url": environment.get("QWEN_URL") or QWEN_BASE_URL,
            "api_key": environment.get("QWEN_API_KEY"),
            "upload_url": environment.get("QWEN_UPLOAD_URL"),
        }
    else:
        defaults = {
            "model": os.getenv("OPENAI_MODEL"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "upload_url": None,
        }
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    return args
