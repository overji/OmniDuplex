"""CLI options; each model keeps its existing flags and defaults."""
import argparse
from pathlib import Path

from benchmark.config.defaults import DEFAULTS, PROBE_ADAPTER_DEFAULTS


def _configure_common(parser, model, *, local_model=True):
    parser.add_argument('--agent-call-schema', choices=('task-start', 'legacy'), default='task-start',
                        help='Benchmark call schema; legacy preserves the original agent_call contract')
    protocols = (('text', 'structured', 'agent-token', 'task-text') if model == 'minicpmo'
                 else ('text', 'structured', 'task-text') if model in ('joyai', 'livestar') else ('text', 'structured'))
    parser.add_argument("--agent-output-protocol", choices=protocols, default=argparse.SUPPRESS,
                        help="Optional structured Agent output; default preserves existing text protocol")
    defaults = DEFAULTS[model]
    parser.add_argument("--data", type=Path, default=defaults["data"])
    parser.add_argument("--videos-dir", type=Path, default=defaults["videos_dir"])
    if local_model:
        parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, default=defaults["system_prompt"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--max-samples", type=int, default=defaults["max_samples"])


def configure_minicpmo(parser):
    _configure_common(parser, 'minicpmo')
    parser.add_argument('--lora-adapter', type=Path, default=argparse.SUPPRESS,
                        help='Agent-token adapter directory; requires --agent-output-protocol agent-token')
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
    p.add_argument("--logits-mode", choices=["full", "last"], default=DEFAULTS["mmduet2"]["logits_mode"],
                   help="Last computes only the next-token vocabulary projection during generation")
    p.add_argument("--mlp-chunk-tokens", type=int, default=DEFAULTS["mmduet2"]["mlp_chunk_tokens"],
                   help="Inference-only tokenwise decoder MLP chunking; 0 disables it")
    p.add_argument("--cache-dir", type=Path, default=DEFAULTS["mmduet2"]["cache_dir"])
    p.add_argument("--official-code", type=Path, required=True,
                   help="Pinned MMDuet2 checkout's proactive_eval directory")
    p.add_argument("--device", default=DEFAULTS["mmduet2"]["device"])


def configure_gander(parser):
    configure_minicpmo(parser)
    for action in parser._actions:
        if action.dest == 'agent_output_protocol':
            action.choices = ('text', 'task-text')
            action.help = 'Gander keeps its native tool-call action grammar; generic JSON/agent-token modes are incompatible'
    parser.set_defaults(output_dir=DEFAULTS['minicpmo']['output_dir'].parent / 'gander',
                        max_speak_tokens=11)
    parser.add_argument('--base-model-path', type=Path, required=True,
                        help='Frozen MiniCPM-o-4.5 base checkpoint used by released Gander Thinker')
    parser.add_argument('--official-code', type=Path, required=True,
                        help='Gander official repository root containing minicpm_ft/mcpmft')
    parser.add_argument('--max-tool-tokens', type=int, default=96,
                        help='Native tool JSON token budget, independent of the per-second speech budget')


def configure_livestar(p):
    _configure_common(p, 'livestar')
    p.add_argument("--cache-dir", type=Path, default=DEFAULTS["livestar"]["cache_dir"])
    p.add_argument("--official-code", type=Path, required=True,
                   help="Pinned LiveStar checkout")
    p.add_argument('--reuse-non-agent', type=Path, default=argparse.SUPPRESS,
                   help='Import compatible completed RTD/PR from an old LiveStar parallel run; Agent always regenerates')


def configure_roma(p):
    _configure_common(p, 'roma')
    p.add_argument("--roma-code", type=Path, required=True, help="Pinned ROMA checkout")


def configure_joyai(p):
    _configure_common(p, 'joyai')
    p.add_argument("--cache-dir", type=Path, default=DEFAULTS["joyai"]["cache_dir"])
    p.add_argument("--adapter-url", default=DEFAULTS["joyai"]["adapter_url"])
    p.add_argument("--model", default=DEFAULTS["joyai"]["model"])


def _configure_cloud_realtime(parser, provider):
    _configure_common(parser, provider, local_model=False)
    defaults = DEFAULTS[provider]
    common = DEFAULTS['cloud_realtime']
    parser.add_argument('--model', default=defaults['model'])
    parser.add_argument('--cache-dir', type=Path, default=defaults['cache_dir'])
    parser.add_argument('--api-key-env', default=defaults['api_key_env'],
                        help='Environment variable holding the API key; never saved to results')
    parser.add_argument('--base-url', help='Endpoint override; otherwise read ' + defaults['base_url_env'])
    parser.add_argument('--api-timeout', type=float, default=common['api_timeout'],
                        help='Maximum seconds for session setup or one complete media/response step')
    parser.add_argument('--max-seconds', type=float, default=common['max_seconds'])
    parser.add_argument('--sample-start', type=int, default=common['sample_start'])
    parser.add_argument('--sample-step', type=int, default=common['sample_step'])
    parser.add_argument('--frame-max-edge', type=int, default=common['frame_max_edge'])
    parser.add_argument('--jpeg-quality', type=int, default=common['jpeg_quality'])
    parser.add_argument('--max-output-tokens', type=int,
                        help='Optional per-step output limit; omitted preserves the provider default')


def configure_qwen_realtime(parser):
    _configure_cloud_realtime(parser, 'qwen_realtime')


def configure_gemini_live(parser):
    _configure_cloud_realtime(parser, 'gemini_live')
    parser.add_argument('--api-version', default=DEFAULTS['gemini_live']['api_version'])


def configure_probe(parser):
    defaults = DEFAULTS['probe']
    commands = parser.add_subparsers(dest='probe_command', required=True)
    prepare = commands.add_parser('prepare', help='Build paired pre/post prefix probes without loading models')
    prepare.add_argument('--data', type=Path, default=defaults['data'])
    prepare.add_argument('--videos-dir', type=Path, default=defaults['videos_dir'])
    prepare.add_argument('--output-dir', type=Path, default=defaults['manifest_dir'])
    run = commands.add_parser('run', help='Run ordinary, non-duplex Omni inference')
    models = run.add_subparsers(dest='probe_model', required=True)
    from benchmark.probe.models import PROBE_MODELS
    for name in PROBE_MODELS:
        model = models.add_parser(name)
        model.add_argument('--manifest', type=Path, required=True)
        model.add_argument('--model-path', type=Path, required=True)
        model.add_argument('--official-code', type=Path,
                           help='Native source checkout, required by Ming and optional for other custom models')
        model.add_argument('--output-dir', type=Path, required=True)
        model.add_argument('--cache-dir', type=Path, default=defaults['cache_dir'])
        model.add_argument('--system-prompt', type=Path, default=defaults['system_prompt'])
        model.add_argument('--fps', type=float, default=defaults['fps'])
        model.add_argument('--max-pixels', type=int, default=defaults['max_pixels'])
        model.add_argument('--max-new-tokens', type=int, default=defaults['max_new_tokens'])
        model.add_argument('--retry-errors-once', action='store_true',
                           help='Finish unattempted requests, then retry failures once with a persistent budget')
        model.add_argument('--output-protocol', choices=('yes-no', 'reply-task-text'), default='yes-no',
                           help='Reply/silent generation or historical YES/NO decision probes')
        model.add_argument('--prompt-layout', choices=('shared', 'task-specific'), default='shared',
                           help='Shared historical prompt or common + per-task prompts and examples')
        model.add_argument('--dtype', choices=('auto', 'bfloat16', 'float16'), default=defaults['dtype'])
        model.add_argument('--device-map', default=defaults['device_map'])
        model.add_argument('--attn-implementation', choices=('sdpa', 'flash_attention_2', 'eager'),
                           default=PROBE_ADAPTER_DEFAULTS.get(name, {}).get('attn_implementation', defaults['attn_implementation']))
        if name == 'baichuan_omni':
            model.add_argument('--attention-query-chunk-size', type=int, default=0,
                               help='Bound native eager attention query memory; 0 keeps original computation')
        model.add_argument('--max-pairs', type=int, default=defaults['max_pairs'],
                           help='Limit pairs before skipping completed probes')
        if name == 'qwen25_omni':
            model.add_argument('--input-modality', choices=('audio-video', 'video-only', 'audio-only'),
                               default=argparse.SUPPRESS, help='Explicit input modality ablation')
    score = commands.add_parser('score', help='Recompute probe metrics without model dependencies')
    score.add_argument('--manifest', type=Path, required=True)
    score.add_argument('--predictions', type=Path, required=True)
    score.add_argument('--output-dir', type=Path, required=True)


def configure_prepare(parser):
    from benchmark.config.paths import CONFIGS, ROOT
    parser.add_argument('--suite', type=Path, default=CONFIGS / 'suites/full_2110.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'probes/full_2110')
    parser.add_argument('--skip-probes', action='store_true',
                        help='Only inventory files/environments; skip ffprobe and paired-manifest creation')


def configure_collect(parser):
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, default=DEFAULTS["collect"]["data"])
    parser.add_argument('--output-dir', type=Path, default=DEFAULTS["collect"]["output_dir"])


def configure_compare(parser):
    p = parser
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=DEFAULTS["compare"]["output"])
    p.add_argument("--judge-dir", type=Path, help="Optional judge output root; timing scores remain unchanged")


def configure_download(parser):
    parser.add_argument("--repo", default=DEFAULTS["download"]["repo"])
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULTS["download"]["manifest"])


def configure_parallel(parser, *, joyai=False):
    parser.add_argument("--sample-indices", type=Path,
                        help="JSON array of original sample indices to benchmark")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--worker-python", type=Path,
                        help="Worker interpreter; defaults to the current Python")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULTS["parallel"]["poll_seconds"])
    if joyai:
        parser.add_argument("--joyai-services", type=Path,
                            help="JSON configuration to launch one native service stack per GPU")


def configure_queue(parser):
    parser.add_argument("--config", type=Path, required=True)


def configure_migrate_agent_windows(parser):
    from benchmark.config.paths import MANIFESTS
    parser.add_argument("--data", type=Path, default=DEFAULTS["collect"]["data"])
    parser.add_argument("--manifest", type=Path, default=MANIFESTS / "data_manifest.json")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")


def configure_judge(parser):
    from benchmark.config.paths import CONFIGS
    parser.add_argument("runs", nargs="+", type=Path, help="Directories containing predictions.jsonl")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("gemini", "openai"), required=True)
    parser.add_argument("--model", required=True, help="Explicit judge model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", help="Prefer GEMINI_API_KEY or OPENAI_API_KEY")
    parser.add_argument("--env-file", type=Path, help="Optional dotenv configuration")
    parser.add_argument("--system-prompt", type=Path, default=CONFIGS / "prompts/judge.txt")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)


def configure_probe_judge(parser):
    configure_judge(parser)
    for action in parser._actions:
        if action.dest in ('provider', 'model'):
            action.required = False  # dry-run validates inputs without credentials or a judge model.
        if action.dest == 'runs':
            action.help = 'Directories containing new-protocol probe_predictions.jsonl'
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Validate and count TP requests without API calls')
