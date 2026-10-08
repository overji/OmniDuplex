"""CLI options; each model keeps its existing flags and defaults."""
from pathlib import Path

from benchmark.config.defaults import DEFAULTS


def _configure_common(parser, model):
    defaults = DEFAULTS[model]
    parser.add_argument("--data", type=Path, default=defaults["data"])
    parser.add_argument("--videos-dir", type=Path, default=defaults["videos_dir"])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, default=defaults["system_prompt"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--max-samples", type=int, default=defaults["max_samples"])


def configure_minicpmo(parser):
    _configure_common(parser, 'minicpmo')
    parser.add_argument("--cache-dir", type=Path, default=DEFAULTS["minicpmo"]["cache_dir"])
    parser.add_argument("--device", default=DEFAULTS["minicpmo"]["device"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["minicpmo"]["seed"])
    parser.add_argument("--sample-start", type=int, default=DEFAULTS["minicpmo"]["sample_start"])
    parser.add_argument("--sample-step", type=int, default=DEFAULTS["minicpmo"]["sample_step"])
    parser.add_argument("--max-seconds", type=float, default=DEFAULTS["minicpmo"]["max_seconds"])
    parser.add_argument("--decode-mode", choices=["sampling", "greedy"], default=DEFAULTS["minicpmo"]["decode_mode"])
    parser.add_argument("--max-speak-tokens", type=int, default=DEFAULTS["minicpmo"]["max_speak_tokens"])
    parser.add_argument("--sliding-window-mode", choices=["off", "basic", "context"], default=DEFAULTS["minicpmo"]["sliding_window_mode"])
    parser.add_argument("--window-high-tokens", type=int, default=DEFAULTS["minicpmo"]["window_high_tokens"])
    parser.add_argument("--window-low-tokens", type=int, default=DEFAULTS["minicpmo"]["window_low_tokens"])
    parser.add_argument("--cpu-threads", type=int, default=DEFAULTS["minicpmo"]["cpu_threads"])
    parser.add_argument("--unused-tts-device", choices=["cpu", "model"], default=DEFAULTS["minicpmo"]["unused_tts_device"])


def configure_mmduet2(p):
    _configure_common(p, 'mmduet2')
    p.add_argument("--cache-dir", type=Path, default=DEFAULTS["mmduet2"]["cache_dir"])
    p.add_argument("--official-code", type=Path, required=True,
                   help="Pinned MMDuet2 checkout's proactive_eval directory")
    p.add_argument("--device", default=DEFAULTS["mmduet2"]["device"])


def configure_livestar(p):
    _configure_common(p, 'livestar')
    p.add_argument("--cache-dir", type=Path, default=DEFAULTS["livestar"]["cache_dir"])
    p.add_argument("--official-code", type=Path, required=True,
                   help="Pinned LiveStar checkout")


def configure_roma(p):
    _configure_common(p, 'roma')
    p.add_argument("--roma-code", type=Path, required=True, help="Pinned ROMA checkout")


def configure_joyai(p):
    _configure_common(p, 'joyai')
    p.add_argument("--cache-dir", type=Path, default=DEFAULTS["joyai"]["cache_dir"])
    p.add_argument("--adapter-url", default=DEFAULTS["joyai"]["adapter_url"])
    p.add_argument("--model", default=DEFAULTS["joyai"]["model"])


def configure_collect(parser):
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULTS["collect"]["data"])
    parser.add_argument('--output-dir', type=Path, default=DEFAULTS["collect"]["output_dir"])


def configure_compare(parser):
    p = parser
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=DEFAULTS["compare"]["output"])


def configure_download(parser):
    parser.add_argument("--repo", default=DEFAULTS["download"]["repo"])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULTS["download"]["manifest"])


def configure_parallel(parser, *, joyai=False):
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--worker-python", type=Path,
                        help="Worker interpreter; defaults to the current Python")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULTS["parallel"]["poll_seconds"])
    if joyai:
        parser.add_argument("--joyai-services", type=Path,
                            help="JSON configuration to launch one native service stack per GPU")


def configure_queue(parser):
    parser.add_argument("--config", type=Path, required=True)
